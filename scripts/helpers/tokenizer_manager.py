import re
import sys
import unicodedata
from collections import Counter

import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
from transformers import NllbTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacremoses import MosesPunctNormalizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


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


def merge_nllb_new_tokenizers(tokenizer_prefix: str, new_tokenizer_name: str):
    # reading the NLLB and the New sentencepiece models into a native format
    tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    sp_trained = spm.SentencePieceProcessor(model_file=f'{tokenizer_prefix}.model')
    added_spm = sp_pb2_model.ModelProto()
    added_spm.ParseFromString(sp_trained.serialized_model_proto())
    old_spm = sp_pb2_model.ModelProto()
    old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

    # adding the missing tokens to the NLLB sentencepiece model
    nllb_tokens_set = {p.piece for p in old_spm.pieces}
    prev_min_score = old_spm.pieces[-1].score
    for p in added_spm.pieces:
        piece = p.piece
        # !!! THIS FIX WAS ADDED LATER; it is required for CT2 compatibility !!!
        # 1 is ordinary token, non-1 is special token; we don't want to copy the special tokens
        if p.type != 1:
            continue
        if piece not in nllb_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            # for all new tokens, I'll set a lower score (priority)
            new_p.score = p.score + prev_min_score
            old_spm.pieces.append(new_p)

    # saving the result to disk
    NEW_SPM_NAME = f'spm_{new_tokenizer_name}.model'
    with open(NEW_SPM_NAME, 'wb') as f:
        f.write(old_spm.SerializeToString())


def reinitialize_weights(new_tokenizer_name: str):
    model_name = 'facebook/nllb-200-distilled-1.3B'

    # loading the tokenizers
    tokenizer_old = NllbTokenizer.from_pretrained(model_name)
    tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=new_tokenizer_name)
    print(len(tokenizer_old), len(tokenizer))  # must see difference
    added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
    print(len(added_vocab))  # new vocab size

    # loading and resizing the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # re-initializing the new embeddings
    for t in tqdm(added_vocab):
        tt = tokenizer_old(t, add_special_tokens=False).input_ids
        if len(tt) == 0:
            tt = [tokenizer_old.unk_token_id]
        idx = tokenizer.convert_tokens_to_ids(t)
        model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)  # WARN: mean initialization


def fix_tokenizer(tokenizer, new_lang='tyv_Cyrl'):
    """
    Add a new language token to the tokenizer vocabulary
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)

    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}


def add_new_full_language(data: pd.DataFrame, lang: str, similar_lang: str):
    all_texts, required_chars = preprocess_corpora(
        data=data,
        lang=lang,
    )
    train_sentencepiece(
        all_texts=all_texts,
        required_chars=required_chars,
        tokenizer_prefix=f'smp_{lang}_16k'
    )
    merge_nllb_new_tokenizers(f'smp_{lang}_16k', f'nllb_{lang}')

    model_name = "facebook/nllb-200-distilled-1.3B"
    # loading the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, vocab_file=f'nllb_{lang}')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # patching them
    fix_tokenizer(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    # fixing the new/moved token embeddings in the model
    added_token_id = tokenizer.convert_tokens_to_ids(f'{lang}_Cyrl')
    similar_lang_id = tokenizer.convert_tokens_to_ids(f'{similar_lang}_Cyrl')
    embeds = model.model.shared.weight.data
    # moving the embedding for "mask" to its new position
    embeds[added_token_id + 1] = embeds[added_token_id]
    # initializing new language token with a token of a similar language
    embeds[added_token_id] = embeds[similar_lang_id]

    return model, tokenizer
