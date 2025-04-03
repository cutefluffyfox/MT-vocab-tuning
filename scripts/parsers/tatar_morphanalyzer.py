import os
import json
import requests
import sentencepiece as spm
from transformers import NllbTokenizer

from scripts.helpers.path_manager import ToeknizerManager
from scripts.parsers.base_tokenizer import BaseTokenizer


class TurkLandMorphTokenizer(BaseTokenizer):
    BASE_URL = 'http://modmorph.turklang.net/ru/platform/morph_analyzer'
    API_URL = 'http://modmorph.turklang.net/ru/platform/morph_analyzer/process_text'

    def __init__(self, file_name: str = 'turk-morph-analyzer.json', pretokenizer_model: str = 'facebook/nllb-200-distilled-1.3B', go_to_api_for_new_word: bool = True):
        self.dm = ToeknizerManager()

        tokenizer = NllbTokenizer.from_pretrained(pretokenizer_model)

        self.spm = spm.SentencePieceProcessor()
        self.spm.LoadFromSerializedProto(tokenizer.sp_model.serialized_model_proto())

        self.file_name: str = file_name
        self.word_to_tokens_map = dict()
        self.load_learned_map(self.file_name)
        self.use_api = go_to_api_for_new_word
        
        self.session = requests.session()
        self.session.get(self.BASE_URL)
        self.session.headers.update({'X-CSRFToken': self.session.cookies['csrftoken']})

    def __pretokenize(self, text: str) -> list[str]:
        return self.spm.EncodeAsPieces(text)

    def tokenize(self, text: str):
        tokens = []
        was_fully_tokenizer: bool = False
        for piece in self.__pretokenize(text):
            piece_tokens, api_status = self.__tokenize_word(piece)
            tokens.extend(piece_tokens)

            if self.use_api and api_status and not was_fully_tokenizer:
                # because API splits text by default, it is faster to make "raw" version beforehand
                self.__send_api_request(text)
                was_fully_tokenizer = True

        return tokens

    @staticmethod
    def __normalize_token(token: str) -> str:
        return token.lower().strip().strip('▁')

    def __send_api_request(self, text: str):
        data = {
            'language': 'TAT',
            'results': 'tree',
            'text': text,
            'cache': 'none',
        }
        response = self.session.post(self.API_URL, data=data, allow_redirects=True)
        token_info = response.json()

        if len(token_info['words']) == 0:
            self.word_to_tokens_map[self.__normalize_token(text)] = [self.__normalize_token(text)]
            return

        for word in token_info['words']:
            if not token_info['words'][word]['recognized']:
                self.word_to_tokens_map[self.__normalize_token(word)] = [self.__normalize_token(word)]
                continue

            # sometimes api returns several options
            tokenization_options = []
            for child in token_info['words'][word].get('children', list()):
                tokenization_options.append(list(TurkLandMorphTokenizer.construct_tokens_from_inner_html(child)))

            # We need to select single option, which is context dependent
            # As it is difficult to extract context, we will just take the longest
            # chain of morphemes to balance amount of tokens in vocab
            long_chain = max(tokenization_options, key=len)

            if long_chain:
                tokens_only = [p[0] for p in long_chain]
                self.word_to_tokens_map[self.__normalize_token(word)] = list(map(self.__normalize_token, tokens_only))

    @staticmethod
    def construct_tokens_from_inner_html(res: dict):
        res['innerHTML']: str
        yield res['innerHTML'].strip('</p>').split(' : ')
        for child in res.get('children', list()):
            yield from TurkLandMorphTokenizer.construct_tokens_from_inner_html(child)
            return  # TODO: return all options, but for now ignore branching

    def load_learned_map(self, file_name):
        file_name = self.dm.get_path(file_name)
        if not os.path.exists(file_name):
            return 
        
        with open(file_name, 'r') as file:
            self.word_to_tokens_map = json.load(file)

    def save_learned_map(self, file_name: str = None):
        if file_name is None:
            file_name = self.file_name

        with open(self.dm.get_path(file_name), 'w') as file:
            json.dump(self.word_to_tokens_map, file, ensure_ascii=True, indent=4)
    
    def __tokenize_word(self, word: str) -> (list[str], bool):
        was_api_request: bool = False
        if self.use_api and self.word_to_tokens_map.get(self.__normalize_token(word)) is None:
            self.__send_api_request(self.__normalize_token(word))
            was_api_request = True

        if self.word_to_tokens_map.get(self.__normalize_token(word)) is None:
            tokens = [self.__normalize_token(word)]
        else:
            tokens = self.word_to_tokens_map[self.__normalize_token(word)]
        token_separator = '\uF000'  # private use area (1st symbol)
        tokens_combined = token_separator.join(tokens)
        ptr_formatted = ptr_tokens = 0

        if ''.join(tokens).lower() != self.__normalize_token(word) or not self.__normalize_token(word):
            # print(f'WARN: API did not return correct format for the word `{word}` : `{tokens}`')
            return [word], was_api_request

        word_len = len(word)
        output = ['']
        while ptr_formatted < word_len:
            if tokens_combined[ptr_tokens] == token_separator:
                output.append('')
                ptr_tokens += 1
            elif tokens_combined[ptr_tokens].lower() == word[ptr_formatted].lower():
                output[-1] += word[ptr_formatted]
                ptr_tokens += 1
                ptr_formatted += 1
            else:
                output[-1] += word[ptr_formatted]
                ptr_formatted += 1

        return output, was_api_request
