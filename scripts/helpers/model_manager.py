import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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


class Madlad400Model(HFModel):
    def __init__(self, convert_to_float16: bool = False, device: str = 'auto'):
        super(Madlad400Model, self).__init__(
            'google/madlad400-3b-mt',
            convert_to_float16=convert_to_float16,
            device=device
        )

    def transform(self, texts: list[dict]) -> list[str]:
        # madlad support translation in form "<2xx> text"
        return [f"<2{row['dst_lang']}> {row['text']}" for row in texts]


class NLLB200Model(HFModel):
    def __init__(self, convert_to_float16: bool = False, device: str = 'auto'):
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

