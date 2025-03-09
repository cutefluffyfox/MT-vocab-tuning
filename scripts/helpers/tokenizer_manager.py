import os
import re
import sys
import json
import shutil
import unicodedata
from collections import Counter

import pandas as pd
from tqdm import tqdm
from sacremoses import MosesPunctNormalizer

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model

from transformers import NllbTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES


def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]
replace_nonprint = get_non_printing_char_replacer(" ")


def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean


def preprocess_corpora(data: pd.DataFrame, lang: str):
    all_texts = data[lang].dropna().drop_duplicates().tolist()
    all_text_normalized = [preproc(t) for t in tqdm(all_texts)]
    chars_cnt = Counter(c for t in all_text_normalized for c in t)
    required_chars = ''.join([
        k for k, v in chars_cnt.most_common()
        if v >= 3 and k not in ' '
    ])

    return all_texts, required_chars


def train_sentencepiece(all_texts, required_chars, tokenizer_prefix: str = 'smp_tyvan_16k'):
    all_texts_file = 'myv_texts_plain.txt'

    with open(all_texts_file, 'w', encoding='UTF-8') as f:
        for i, text in enumerate(all_texts):
            print(text, file=f)

    spm.SentencePieceTrainer.train(
        input=all_texts_file,
        model_prefix=tokenizer_prefix,
        vocab_size=2 ** 14,  # 16K
        character_coverage=1,
        num_threads=16,
        train_extremely_large_corpus=False,
        add_dummy_prefix=False,
        max_sentencepiece_length=128,
        max_sentence_length=4192 * 4,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        required_chars=required_chars,
    )


def merge_nllb_new_tokenizers(model_name: str, tokenizer_prefix: str, new_tokenizer_name: str):
    # reading the NLLB and the New sentencepiece models into a native format
    sp_trained = spm.SentencePieceProcessor(model_file=f'{tokenizer_prefix}.model')
    added_spm = sp_pb2_model.ModelProto()
    added_spm.ParseFromString(sp_trained.serialized_model_proto())

    tokenizer = NllbTokenizer.from_pretrained(model_name)

    old_spm = sp_pb2_model.ModelProto()
    old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

    # adding the missing tokens to the NLLB sentencepiece model
    nllb_tokens_set = {p.piece for p in old_spm.pieces}
    prev_min_score = old_spm.pieces[-1].score

    for p in added_spm.pieces:
        piece = p.piece
        if piece not in nllb_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            # for all new tokens, I'll set a lower score (priority)
            new_p.score = p.score + prev_min_score
            old_spm.pieces.append(new_p)

    # saving the result to disk
    with open(f'spm_{new_tokenizer_name}.model', 'wb') as f:
        f.write(old_spm.SerializeToString())


def update_nllb_tokenizer(
    old_tokenizer: NllbTokenizer,
    new_spm_path: str,
    new_lang_codes: list[str],
) -> NllbTokenizer:

    TKN_DIR = "old_tokenizer"  # todo: make it temporary
    old_tokenizer.save_pretrained(TKN_DIR)

    with open(f"{TKN_DIR}/tokenizer_config.json", "r") as f:
        cfg = json.load(f)
    cfg["added_tokens_decoder"] = {
        k: v
        for k, v in cfg["added_tokens_decoder"].items()
        if k in ["0", "1", "2", "3"]
    }
    cfg["additional_special_tokens"] = []
    with open(f"{TKN_DIR}/tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # this contains added tokens: language codes and mask
    os.remove(f"{TKN_DIR}/added_tokens.json")
    os.remove(f"{TKN_DIR}/special_tokens_map.json")
    os.remove(f"{TKN_DIR}/sentencepiece.bpe.model")
    shutil.copy(new_spm_path, f"{TKN_DIR}/sentencepiece.bpe.model")

    new_tokenizer = NllbTokenizer.from_pretrained(
        TKN_DIR,
        additional_special_tokens=sorted(FAIRSEQ_LANGUAGE_CODES + new_lang_codes),
    )

    # TODO: clean-up dir
    # shutil.rmtree(TKN_DIR)
    return new_tokenizer


def add_new_full_language(data: pd.DataFrame, lang: str, model_name: str):
    all_texts, required_chars = preprocess_corpora(
        data=data,
        lang=lang,
    )
    train_sentencepiece(
        all_texts=all_texts,
        required_chars=required_chars,
        tokenizer_prefix=f'smp_{lang}_16k'
    )
    merge_nllb_new_tokenizers(
        model_name=model_name,
        tokenizer_prefix=f'smp_{lang}_16k',
        new_tokenizer_name=f'nllb_{lang}'
    )

    # load old tokenizer for sanity check
    tokenizer_old = NllbTokenizer.from_pretrained(model_name)
    tokenizer = update_nllb_tokenizer(
        old_tokenizer=tokenizer_old,
        new_spm_path=f'spm_nllb_{lang}.model',
        new_lang_codes=[f'{lang}_Cyrl']
    )

    # print len and last 2 tokens (should be some lang and <mask>)
    print(f'Old tokenizer length: {len(tokenizer_old)}')
    print(f'New tokenizer length: {len(tokenizer)}')

    # sanity check that everything loaded correctly
    new_code = tokenizer.convert_tokens_to_ids(f'{lang}_Cyrl')
    mask_code = tokenizer.convert_tokens_to_ids('<mask>')
    print(f"Code of `{lang}_Cyrl`: {new_code}")
    print(f"Code of `<mask>`: {mask_code}")
    print(f"Decoded `{lang}_Cyrl` and `<mask>`:", tokenizer.convert_ids_to_tokens([new_code, mask_code]))
    return tokenizer


def initialize_new_model_emb(model_name: str, model, tokenizer, lang: str, similar_lang: str):
    tokenizer_old = NllbTokenizer.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    moved_tokens = list(tokenizer_old.lang_code_to_id) + ['<mask>']
    model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(moved_tokens)] = model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids(moved_tokens)]
    model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(f'{lang}_Cyrl')] = model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids(f'{similar_lang}_Cyrl')]
    added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
    print('Amount of new tokens:', len(added_vocab))
    for t in tqdm(added_vocab):
        if t == f'{lang}_Cyrl':
            continue
        tt = tokenizer_old(t, add_special_tokens=False).input_ids
        if len(tt) == 0:
            print(f'empty token "{t}"/{tokenizer.convert_tokens_to_ids(t)}')
            tt = [tokenizer_old.unk_token_id]
        model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(t)] = model.model.shared.weight.data[tt].mean(0)

