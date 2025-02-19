import gc
import torch
import numpy as np
import pandas as pd
from time import time
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
# TODO: cleanup yadisk to other place
import yadisk
import posixpath
import shutil
import os


def recursive_upload(client: yadisk.Client, from_dir: str, to_dir: str):
    for root, dirs, files in os.walk(from_dir):
        p = root.split(from_dir)[1].strip(os.path.sep)
        dir_path = posixpath.join(to_dir, p)

        try:
            client.mkdir(dir_path)
        except yadisk.exceptions.PathExistsError:
            pass

        for file in files:
            file_path = posixpath.join(dir_path, file)
            p_sys = p.replace("/", os.path.sep)
            in_path = os.path.join(from_dir, p_sys, file)

            try:
                client.upload(in_path, file_path)
            except yadisk.exceptions.PathExistsError:
                pass


class BaseModel:
    def translate_single(self, text: str, src_lang: str, dst_lang: str, *args, **kwargs) -> str:
        return self.translate_batch([text], src_lang, dst_lang, *args, **kwargs)[0]

    def translate_batch(self, texts: list[str] or list[dict], src_lang: str = None, dst_lang: str = None, *args, **kwargs) -> list[str]:
        if not bool(texts):  # if empty
            return []

        # check if input type is correct
        assert (type(texts[0]) is dict) or (type(texts[0]) is str and src_lang is not None and dst_lang is not None), 'Input parameter `texts` should be either list of dicts, or list of strings with `src_lang` and `dst_lang` specified.'

        # if list of strings -> convert to list of dicts
        if type(texts[0]) is str:
            texts = self.add_lang_info_columns(texts, src_lang, dst_lang)

        # check that all dict columns are present
        assert 'text' in texts[0] and 'src_lang' in texts[0] and 'dst_lang' in texts[0], 'Input parameter `texts` should explicitly have `text`, `src_lang` and `dst_lang` specified.'

        # transform -> tokenize -> translate
        return self.translate(self.tokenize(self.transform(texts)))

    def transform(self, texts: list[dict]) -> list[str]:
        return [row['text'] for row in texts]

    def tokenize(self, texts: list[str], *args, **kwargs) -> list[list[int]]:
        raise NotImplementedError('`.tokenize` method should be implemented in derived class.')

    def translate(self, tokens: list[list[int]], *args, **kwargs) -> list[str]:
        raise NotImplementedError('`.translate` method should be implemented in derived class.')

    @staticmethod
    def add_lang_info_columns(texts: list[str], src_lang: str, dst_lang: str) -> list[dict]:
        output = []
        for text in texts:
            output.append({
                'text': text,
                'src_lang': src_lang,
                'dst_lang': dst_lang
            })
        return output
    
    @staticmethod
    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()


class HFModel(BaseModel):
    def __init__(self, hf_repo: str, convert_to_float16: bool = False, device: str = 'auto'):
        dtype = torch.float32 if not convert_to_float16 else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_repo, torch_dtype=dtype, device_map=device)

    def tokenize(self, texts: list[str], *args, **kwargs) -> dict:
        return self.tokenizer(texts, return_tensors="pt", padding=True, *args, **kwargs).to(self.model.device)

    def translate(self, input_ids: dict, max_tokens: int = 100, *args, **kwargs) -> list[str]:
        outputs = self.model.generate(**input_ids, max_new_tokens=max_tokens)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def train(
        self, 
        data: pd.DataFrame,
        src_lang: str, 
        dst_lang: str, 
        experiment_name: str,
        ya_client: yadisk.Client = None,
        batch_size: int = 16, 
        max_length : int = 128,
        training_steps: int = 100,
        save_every: int = 1000,
        learn_both_direction: bool = False,
        ttl: int = 4*60*60,
    ):
        # sanity check on data
        assert data[src_lang].shape[0] > 0, f"Column `{src_lang}` is empty"
        assert data[dst_lang].shape[0] > 0, f"Column `{dst_lang}` is empty"
        assert data[src_lang].shape == data[dst_lang].shape, f"Length mismatch in columns `{src_lang}` and `{dst_lang}`"
        
        # prepare ya-disk env
        if ya_client is not None:
            ya_client.mkdir(f"/bs-diploma/experiments/kaggle/{experiment_name}")
        
        # TODO: move it to outer space
        def get_batch_pairs(batch_size: int):
            idx = 0
            straight = True
            n = data.shape[0]
            while True:
                srcs, dsts = [], []
                for _ in range(batch_size):
                    row = data.iloc[idx%n]
                    srcs.append(row[src_lang])
                    dsts.append(row[dst_lang])
                    idx += 1
                if straight:
                    yield (srcs, dsts, src_lang, dst_lang)
                else:
                    yield (dsts, srcs, dst_lang, src_lang)
                if learn_both_direction:
                    straight = not straight
        
        # scheduler & optimizer
        self.model.cuda()
        optimizer = Adafactor(
            [p for p in self.model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
        
        # set model to training mode (enable grad)
        self.model.train()
        
        # some extra variables to get track of what is happening
        losses = []
        start_time = time()
        it_number = 0
        
        x, y, loss = None, None, None
        self.cleanup()

        tq = trange(len(losses), training_steps)
        for i in tq:
            if (time() - start_time) >= ttl:
                print(f'Triggered time to leave! Current time: {time()}. Start time: {start_time}. TTL: {ttl}.')
                break
            batch = next(get_batch_pairs(batch_size))
            xx, yy, lang1, lang2 = batch
            try:
                self.tokenizer.src_lang = lang1
                x = self.tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(self.model.device)
                self.tokenizer.src_lang = lang2
                y = self.tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(self.model.device)
                # -100 is a magic value ignored in the loss function
                # because we don't want the model to learn to predict padding ids
                y.input_ids[y.input_ids == self.tokenizer.pad_token_id] = -100

                loss = self.model(**x, labels=y.input_ids).loss
                loss.backward()
                losses.append(loss.item())

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            except RuntimeError as e:  # usually, it is out-of-memory
                optimizer.zero_grad(set_to_none=True)
                x, y, loss = None, None, None
                self.cleanup()
                print('error', max(len(s) for s in xx + yy), e)
                # continue i don't want it due to the risk of loosing checkpoint
            
            it_number += 1
            if i % save_every == 0:
                # each `save_every` steps, report average loss at these steps
                print(i, np.mean(losses[-save_every:]))

            if i % save_every == 0:
                self.model.save_pretrained(f"model.sft.{src_lang}_{dst_lang}.{i}")
                self.tokenizer.save_pretrained(f"tokenizer.sft.{src_lang}_{dst_lang}.{i}")
                if ya_client is not None:
                    try:
                        ya_client.mkdir(f"/bs-diploma/experiments/kaggle/{experiment_name}/model.sft.{src_lang}_{dst_lang}.{i}")
                        ya_client.mkdir(f"/bs-diploma/experiments/kaggle/{experiment_name}/tokenizer.sft.{src_lang}_{dst_lang}.{i}")
                    
                        recursive_upload(ya_client, f"model.sft.{src_lang}_{dst_lang}.{i}", f"/bs-diploma/experiments/kaggle/{experiment_name}/model.sft.{src_lang}_{dst_lang}.{i}")
                        recursive_upload(ya_client, f"tokenizer.sft.{src_lang}_{dst_lang}.{i}", f"/bs-diploma/experiments/kaggle/{experiment_name}/tokenizer.sft.{src_lang}_{dst_lang}.{i}")
                        
                        shutil.rmtree(f"model.sft.{src_lang}_{dst_lang}.{i}")
                        shutil.rmtree(f"tokenizer.sft.{src_lang}_{dst_lang}.{i}")
                    except Exception as ex:
                        print('Failed to backup on ya disk')
                        print(ex)
    
        # save final model
        
        self.model.save_pretrained(f"model.sft.{src_lang}_{dst_lang}.{it_number}.final")
        self.tokenizer.save_pretrained(f"tokenizer.sft.{src_lang}_{dst_lang}.{it_number}.final")
        if ya_client is not None:
            try:
                ya_client.mkdir(f"/bs-diploma/experiments/kaggle/{experiment_name}/model.sft.{src_lang}_{dst_lang}.{it_number}.final")
                ya_client.mkdir(f"/bs-diploma/experiments/kaggle/{experiment_name}/tokenizer.sft.{src_lang}_{dst_lang}.{it_number}.final")

                recursive_upload(ya_client, f"model.sft.{src_lang}_{dst_lang}.{it_number}.final", f"/bs-diploma/experiments/kaggle/{experiment_name}/model.sft.{src_lang}_{dst_lang}.{it_number}.final")
                recursive_upload(ya_client, f"tokenizer.sft.{src_lang}_{dst_lang}.{it_number}.final", f"/bs-diploma/experiments/kaggle/{experiment_name}/tokenizer.sft.{src_lang}_{dst_lang}.{it_number}.final")
            except Exception as ex:
                print('Failed to backup on ya disk')
                print(ex)


class Madlad400Model(HFModel):
    def __init__(self, convert_to_float16: bool = True, device: str = 'auto'):
        super(Madlad400Model, self).__init__(
            'google/madlad400-3b-mt',
            convert_to_float16=convert_to_float16,
            device=device
        )

    def transform(self, texts: list[dict]) -> list[str]:
        # madlad support translation in form "<2xx> text"
        return [f"<2{row['dst_lang']}> {row['text']}" for row in texts]


class NLLB200Model(HFModel):
    def __init__(self, convert_to_float16: bool = True, device: str = 'auto'):
        super(NLLB200Model, self).__init__(
            'facebook/nllb-200-distilled-1.3B',
            convert_to_float16=convert_to_float16,
            device=device,
        )
        self.dst_lang: str = None

    def transform(self, texts: list[dict]) -> list[str]:
        self.dst_lang = self.__from_iso(texts[0]['dst_lang'])
        return [f"{row['text']}" for row in texts]

    def translate(self, input_ids: dict, max_tokens: int = 100, *args, **kwargs) -> list[str]:
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.dst_lang)
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def __from_iso(self, lang: str):
        if lang == 'en':
            return 'eng_Latn'
        elif lang == 'ru':
            return 'rus_Cyrl'
        elif lang == 'tat' or lang == 'tt':
            return 'tat_Cycl'
        elif lang == 'kaz' or lang == 'kk':
            return 'kaz_Cyrl'
        elif lang == 'mhr':
            raise ValueError('Language: mhr is not supported by NLLB-200')
        else:
            raise NotImplementedError(f'Language `{lang}` not supported yet.')

