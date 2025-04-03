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

from scripts.helpers.model_manager import BaseModel
from scripts.parsers.base_tokenizer import BaseTokenizer
from scripts.parsers.tatar_morphanalyzer import TurkLandMorphTokenizer



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


def train_sentencepiece(all_texts: list[str], required_chars: str, model_type: str, tokenizer_prefix: str = 'smp_tyvan_16k'):
    all_texts_file = 'tokenizer_texts_plain.txt'

    with open(all_texts_file, 'w', encoding='UTF-8') as f:
        for i, text in enumerate(all_texts):
            print(text, file=f)

    assert model_type in {'bpe', 'unigram', 'char', 'word'}, "Invalid model type: expected {bpe, unigram, char, word}"
    spm.SentencePieceTrainer.train(
        input=all_texts_file,
        model_prefix=tokenizer_prefix,
        model_type=model_type,  # default unigram
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


def train_morph_analyzer(all_texts: list[str], pretrained_tokenizer: BaseTokenizer, output_file: str = 'tt_tokens.json'):
    # calculate each token frequency
    cnt = Counter()
    for line in tqdm(all_texts, desc='Tokenizing texts'):
        tokens = pretrained_tokenizer.tokenize(line)
        for token in tokens:
            cnt[token] += 1

    # get top frequent
    vocab_size = 2 ** 14
    most_common_tokens = [token for token, freq in cnt.most_common(vocab_size)]

    with open(output_file, 'w') as file:
        json.dump(most_common_tokens, fp=file)


def merge_nllb_new_tokenizers(model_name: str, tokenizer_prefix: str, new_tokenizer_name: str):
    # if it is raw tokens -> create new SentencePieceProcessor
    if tokenizer_prefix.endswith('.json'):
        added_spm = sp_pb2_model.ModelProto()

        with open(tokenizer_prefix, 'r') as file:
            tokens = json.load(fp=file)

        for piece in tokens:
            new_p = sp_pb2_model.ModelProto().SentencePiece()

            # set new score to zero
            new_p.piece = piece
            new_p.score = 0

            added_spm.pieces.append(new_p)
    # otherwise read in native format
    else:
        # reading the NLLB and the New sentencepiece models into a native format
        sp_trained = spm.SentencePieceProcessor(model_file=f'{tokenizer_prefix}.model')
        added_spm = sp_pb2_model.ModelProto()
        added_spm.ParseFromString(sp_trained.serialized_model_proto())

    tokenizer = NllbTokenizer.from_pretrained(model_name)

    old_spm = sp_pb2_model.ModelProto()
    old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

    # adding the missing tokens to the NLLB sentencepiece model
    nllb_tokens_set = {p.piece for p in old_spm.pieces}
    print('Nllb tokens set len:', len(nllb_tokens_set))
    prev_min_score = old_spm.pieces[-1].score

    for p in added_spm.pieces:
        piece = p.piece
        if piece not in nllb_tokens_set:
            if piece in {'<s>', '</s>', '<unk>', '<pad>', '<mask>'}:
                print(f'{piece} not in nllb tokens set?')
                print('This is UNEXPECTED BEHAVIOR that cause to BREAK tokenizer')
                print('We have hotfix to just skip, but IT IS BAD if you are seeing this')
                continue
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
    new_lang_codes: list[str],
    new_spm_path: str = None,
) -> NllbTokenizer:

    TKN_DIR = "old_tokenizer"  # todo: make it temporary
    old_tokenizer.save_pretrained(TKN_DIR)
    
    with open(f"{TKN_DIR}/tokenizer_config.json", "r") as f:
        cfg = json.load(f)
    cfg["added_tokens_decoder"] = {
        k: v
        for k, v in cfg["added_tokens_decoder"].items()
        if k in ["0", "1", "2", "3"]  # we remove <mask> and <2 lang> tokens
    }
    cfg["additional_special_tokens"] = []  # <2 lang> tokens
    with open(f"{TKN_DIR}/tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # this contains added tokens: language codes and mask
    os.remove(f"{TKN_DIR}/added_tokens.json")
    os.remove(f"{TKN_DIR}/special_tokens_map.json")
    if new_spm_path is not None:
        os.remove(f"{TKN_DIR}/sentencepiece.bpe.model")
        shutil.copy(new_spm_path, f"{TKN_DIR}/sentencepiece.bpe.model")

    new_tokenizer = NllbTokenizer.from_pretrained(
        TKN_DIR,
        additional_special_tokens=sorted(FAIRSEQ_LANGUAGE_CODES + new_lang_codes),
    )

    # TODO: clean-up dir
    # shutil.rmtree(TKN_DIR)
    return new_tokenizer


def add_new_full_language(lang: str, model_name: str, tokenization_type: str, data: pd.DataFrame = None, add_token_lang: bool = True):
    if data is not None:
        tokenization_type = tokenization_type.lower()
        all_texts, required_chars = preprocess_corpora(
            data=data,
            lang=lang,
        )

        if tokenization_type in {'bpe', 'unigram'}:
            tokenizer_prefix = f'smp_{lang}_16k'
            train_sentencepiece(
                all_texts=all_texts,
                required_chars=required_chars,
                tokenizer_prefix=tokenizer_prefix,
                model_type=tokenization_type,
            )
        elif tokenization_type == 'morph':
            if lang.lower() in {'tt', 'tat'}:
                tokenizer = TurkLandMorphTokenizer(
                    pretokenizer_model=model_name,
                    go_to_api_for_new_word=False,
                )
            else:
                raise NotImplementedError(f'Language `{lang}` does not have MorphTokenization yet')

            tokenizer_prefix = f'{lang}-raw_tokens.json'
            train_morph_analyzer(
                all_texts=all_texts,
                pretrained_tokenizer=tokenizer,
                output_file=tokenizer_prefix
            )
        else:
            tokenizer_prefix = 'unknown'
            raise ValueError('Unsupported tokenization type')

    # merge tokens
    merge_nllb_new_tokenizers(
        model_name=model_name,
        tokenizer_prefix=tokenizer_prefix,
        new_tokenizer_name=f'nllb_{lang}'
    )

    # load old tokenizer for sanity check
    tokenizer_old = NllbTokenizer.from_pretrained(model_name)
    tokenizer = update_nllb_tokenizer(
        old_tokenizer=tokenizer_old,
        new_spm_path=None if (data is None) else f'spm_nllb_{lang}.model',
        new_lang_codes=[f'{lang}_Cyrl'] if add_token_lang else [],
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


def initialize_new_model_emb(model_name: str, model, tokenizer: NllbTokenizer, lang: str, similar_lang: str, tactic: str):
    # sanity check on input parameters
    if similar_lang.startswith(lang):
        similar_lang = f"{lang}_Cyrl"
    assert '_' not in lang, 'Parameter `lang` should be Cyrillic language without language script'
    assert '_' in similar_lang, 'Parameter `similar_lang` should specify language script'

    # unload model from cuda (otherwise code will fail due to device map)
    device = model.device
    model.to('cpu')
    BaseModel.cleanup()

    tokenizer_old = NllbTokenizer.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # a bit more universal way to get moved tokens
    new_tokens = set(tokenizer.get_vocab())
    old_tokens = set(tokenizer_old.get_vocab())
    added_vocab = new_tokens - old_tokens
    moved_tokens = [token for token in old_tokens if tokenizer_old.convert_tokens_to_ids(token) != tokenizer.convert_tokens_to_ids(token)]

    # copy embeddings from old token positions to new
    new_weights = model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(moved_tokens)]
    model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(moved_tokens)] = model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids(moved_tokens)]
    model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids(moved_tokens)] = new_weights  # copy uninitialized, otherwise weights won't be random anymore

    model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(f'{lang}_Cyrl')] = model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids(similar_lang)]

    # sanity check on index position
    added_tokens_idx = tokenizer.convert_tokens_to_ids(list(added_vocab))
    print(f'Amount of moved tokens: {len(moved_tokens)} {moved_tokens[:10]}')
    print(f'Previous tokenizer length: {len(tokenizer_old)}')
    print(f'New tokenizer length: {len(tokenizer)}')
    print(f'Ids of new tokens: min [{min(added_tokens_idx)}], max [{max(added_tokens_idx)}], mean [{sum(added_tokens_idx)//len(added_tokens_idx)}]')

    print('Amount of new tokens:', len(added_vocab))
    print(list(added_vocab)[:15])

    if tactic == 'random':
        pass  # done automatically
    elif tactic == 'FVT':  # fast vocabulary transfer
        for t in tqdm(added_vocab):
            if t == f'{lang}_Cyrl':
                continue
            tt = tokenizer_old(t, add_special_tokens=False).input_ids
            if len(tt) == 0:
                print(f'empty token "{t}"/{tokenizer.convert_tokens_to_ids(t)}')
                tt = [tokenizer_old.unk_token_id]
            model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(t)] = model.model.shared.weight.data[tt].mean(0)
    elif tactic == 'TransTokenization':  # Trans-tokenization (again based on similar lang)
        raise NotImplementedError('TransTokenization nor supported yet')
    else:
        raise ValueError('Unknown tokenization method')

    # load model back to needed device
    model.to(device)


"""


[
'ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 
'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 
'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 
'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 
'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 
'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 
'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 
'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 
'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 
'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 
'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 
'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 
'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn', 'lin_Latn', 
'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 
'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn', 
'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 
'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 
'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 
'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Beng', 'scn_Latn', 'shn_Mymr', 
'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 
'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 
'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 
'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 
'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 
'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 
'zho_Hant', 'zul_Latn'
]


"""
