import gc
import torch
import yadisk
import shutil
import numpy as np
import pandas as pd
from time import time
from tqdm.auto import trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, NllbTokenizer
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from scripts.helpers.yadisk_manager import recursive_upload
from scripts.helpers.batch_processor import get_batch_pairs


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
    def from_iso(lang: str):
        if lang in {'en', 'eng', 'en_Latn', 'eng_Latn'}:
            return 'eng_Latn'
        elif lang in {'ru', 'rus', 'ru_Cyrl', 'rus_Cyrl'}:
            return 'rus_Cyrl'
        elif lang in {'tt', 'tat', 'tt_Cyrl', 'tat_Cyrl'}:
            return 'tat_Cycl'
        elif lang in {'kk', 'kaz', 'kk_Cyrl', 'kaz_Cyrl'}:
            return 'kaz_Cyrl'
        elif lang in {'mhr', 'chm', 'mhr_Cyrl', 'chm_Cyrl'}:
            return 'mhr_Cyrl'
        else:
            raise NotImplementedError(f'Language `{lang}` not supported yet.')

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
    def __init__(self, hf_repo: str, convert_to_float16: bool = False, device: str = 'auto', skip_initialization: bool = False):
        dtype = torch.float32 if not convert_to_float16 else torch.float16
        if not skip_initialization:
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
        yadisk_dir: str = '/bs-diploma/experiments/kaggle',
        ttl: int = 4*60*60,
        lr: float = 1e-4,
        clip_threshold: float = 1.0,
        weight_decay: float = 1e-3,
        num_warmup_steps: int = 1000,
    ):
        # sanity check on data
        assert data[src_lang].shape[0] > 0, f"Column `{src_lang}` is empty"
        assert data[dst_lang].shape[0] > 0, f"Column `{dst_lang}` is empty"
        assert data[src_lang].shape == data[dst_lang].shape, f"Length mismatch in columns `{src_lang}` and `{dst_lang}`"
        
        # prepare ya-disk env
        if ya_client is not None:
            try:
                ya_client.mkdir(f"{yadisk_dir}/{experiment_name}")
            except yadisk.exceptions.PathExistsError as ex:
                print('Experiment with this name already exists in yaDis')
                raise ex
        
        # scheduler & optimizer
        self.model.cuda()
        optimizer = Adafactor(
            [p for p in self.model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=lr,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
        )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
        
        # set model to training mode (enable grad)
        self.model.train()
        
        # some extra variables to get track of what is happening
        losses = []
        start_time = time()
        it_number = 0
        batch_generator = get_batch_pairs(
            data=data, batch_size=batch_size,
            src_lang=src_lang, dst_lang=dst_lang,
            learn_both_direction=learn_both_direction,
        )
        self.cleanup()

        tq = trange(len(losses), training_steps)
        for i in tq:
            if (time() - start_time) >= ttl:
                print(f'Triggered time to leave! Current time: {time()}. Start time: {start_time}. TTL: {ttl}.')
                break
            batch = next(batch_generator)
            xx, yy, lang1, lang2 = batch
            # normalize language token, otherwise model would learn different 2 token (fixed 16.03!!!)
            lang1, lang2 = self.from_iso(lang1), self.from_iso(lang2)
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
                self.cleanup()
                print('error', max(len(s) for s in xx + yy), e)
                # continue  # I don't want it due to the risk of loosing checkpoint
            
            it_number += 1
            if i % save_every == 0:
                # each `save_every` steps, report average loss at these steps
                print(i, np.mean(losses[-save_every:]))

            if i % save_every == 0:
                self.model.save_pretrained(f"model.sft.{src_lang}_{dst_lang}.{i}")
                self.tokenizer.save_pretrained(f"tokenizer.sft.{src_lang}_{dst_lang}.{i}")
                if ya_client is not None:
                    try:
                        ya_client.mkdir(f"{yadisk_dir}/{experiment_name}/model.sft.{src_lang}_{dst_lang}.{i}")
                        ya_client.mkdir(f"{yadisk_dir}/{experiment_name}/tokenizer.sft.{src_lang}_{dst_lang}.{i}")
                    
                        recursive_upload(ya_client, f"model.sft.{src_lang}_{dst_lang}.{i}", f"{yadisk_dir}/{experiment_name}/model.sft.{src_lang}_{dst_lang}.{i}")
                        recursive_upload(ya_client, f"tokenizer.sft.{src_lang}_{dst_lang}.{i}", f"{yadisk_dir}/{experiment_name}/tokenizer.sft.{src_lang}_{dst_lang}.{i}")
                        
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
                ya_client.mkdir(f"{yadisk_dir}/{experiment_name}/model.sft.{src_lang}_{dst_lang}.{it_number}.final")
                ya_client.mkdir(f"{yadisk_dir}/{experiment_name}/tokenizer.sft.{src_lang}_{dst_lang}.{it_number}.final")

                recursive_upload(ya_client, f"model.sft.{src_lang}_{dst_lang}.{it_number}.final", f"{yadisk_dir}/{experiment_name}/model.sft.{src_lang}_{dst_lang}.{it_number}.final")
                recursive_upload(ya_client, f"tokenizer.sft.{src_lang}_{dst_lang}.{it_number}.final", f"{yadisk_dir}/{experiment_name}/tokenizer.sft.{src_lang}_{dst_lang}.{it_number}.final")
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
    def __init__(self, convert_to_float16: bool = True, device: str = 'auto', skip_initialization: bool = False):
        super(NLLB200Model, self).__init__(
            'facebook/nllb-200-distilled-1.3B',
            convert_to_float16=convert_to_float16,
            device=device,
        )
        self.dst_lang: str = None

    @staticmethod
    def from_folder(model_path: str, tokenizer_path: str, convert_to_float16: bool = False, device: str = 'auto'):
        dtype = torch.float32 if not convert_to_float16 else torch.float16
        tokenizer = NllbTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)
        nllb_model = NLLB200Model(convert_to_float16=convert_to_float16, device=device, skip_initialization=True)
        nllb_model.model = model
        nllb_model.tokenizer = tokenizer
        return nllb_model


    def transform(self, texts: list[dict]) -> list[str]:
        if 'break_bos_token' in dir(self) and self.break_bos_token:
            self.dst_lang = texts[0]['dst_lang']
        else:
            self.dst_lang = self.from_iso(texts[0]['dst_lang'])
        return [f"{row['text']}" for row in texts]

    def translate(self, input_ids: dict, max_tokens: int = 100, *args, **kwargs) -> list[str]:
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.dst_lang)
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
