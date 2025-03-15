import yadisk
import posixpath
import os
from tqdm import tqdm


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


def download_model(client: yadisk.Client, experiment: str, folder: str, checkpoint: str='final'):
    model_files = {'config.json', 'generation_config.json', 'model.safetensors'}
    tokenizer_files_new = {'special_tokens_map.json', 'sentencepiece.bpe.model', 'tokenizer_config.json', 'added_tokens.json'}
    tokenizer_files_old = {'special_tokens_map.json', 'sentencepiece.bpe.model', 'tokenizer_config.json', 'tokenizer.json'}

    assert client.exists(experiment), 'Experiment not found on Ya.Disk'

    # get all checkpoints & process them
    checkpoints = list(client.listdir(experiment))
    parsed = dict()
    for checkpoint_dir_object in tqdm(checkpoints, desc='Preprocessing checkpoints'):
        checkpoint_dir = checkpoint_dir_object.name
        checkpoint_type, train_type, direction, iteration, *extra = checkpoint_dir.split('.')

        if checkpoint_type == 'tokenizer':
            continue

        model_path = f"{experiment}/model.{train_type}.{direction}.{iteration}" + (f'.{extra}' if extra else '')
        tokenizer_path = f"{experiment}/tokenizer.{train_type}.{direction}.{iteration}" + (f'.{extra}' if extra else '')

        # print(client.exists(model_path))
        # print(len(set([file_obj.name for file_obj in client.listdir(model_path)]) & model_files) == len(model_files))
        # print(client.exists(tokenizer_path))
        # print(len(set([file_obj.name for file_obj in client.listdir(tokenizer_path)]) & tokenizer_files) == len(tokenizer_files))
        if not client.exists(model_path):
            continue
        if not len(set([file_obj.name for file_obj in client.listdir(model_path)]) & model_files) == len(model_files):
            continue
        if not client.exists(tokenizer_path):
            continue
        if (
            not len(set([file_obj.name for file_obj in client.listdir(tokenizer_path)]) & tokenizer_files_old) == len(tokenizer_files_old)
            and not len(set([file_obj.name for file_obj in client.listdir(tokenizer_path)]) & tokenizer_files_new) == len(tokenizer_files_new)
        ):
            continue

        parsed[int(iteration)] = {
            'iteration': int(iteration),
            'model': model_path,
            'tokenizer': tokenizer_path,
            'tokenizer_type': 'old' if 'tokenizer.json' in set([file_obj.name for file_obj in client.listdir(tokenizer_path)]) else 'new',
        }

    if checkpoint in {'final', 'last', 'full'}:
        checkpoint = max(parsed.keys())
    else:
        checkpoint = int(checkpoint)

    assert checkpoint in parsed, f'Failed to find checkpoint {checkpoint}'

    # make local folder
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, 'model'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'tokenizer'), exist_ok=True)

    for file in os.listdir(os.path.join(folder, 'model')):
        os.remove(os.path.join(folder, 'model', file))
    for file in os.listdir(os.path.join(folder, 'tokenizer')):
        os.remove(os.path.join(folder, 'tokenizer', file))

    # download all files
    tokenizer_files = tokenizer_files_old if parsed[checkpoint]['tokenizer_type'] == 'old' else tokenizer_files_new
    for file_type in tqdm(tokenizer_files, desc='Downloading tokenizer'):
        client.download(f"{parsed[checkpoint]['tokenizer']}/{file_type}", os.path.join(folder, 'tokenizer', file_type))
    # download all tokenizer files
    for file_type in tqdm(model_files, desc='Downloading model'):
        client.download(f"{parsed[checkpoint]['model']}/{file_type}", os.path.join(folder, 'model', file_type))



