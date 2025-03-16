import os

from tqdm import tqdm

from scripts.helpers.model_manager import BaseModel
from scripts.helpers.dataset_manager import FloresDataset
from scripts.helpers.path_manager import DataManager
from scripts.helpers.batch_processor import batch_split


def benchmark(model: BaseModel, direction: str, logs_name: str = 'flores-dev.txt'):
    # make logs folder & file
    dm = DataManager()
    logs_file = dm.get_path('benchmark', direction, logs_name)
    with open(logs_file, 'w', encoding='UTF-8') as file:  # remove previous results (if exist)
        pass

    # read dataset &
    df = FloresDataset(direction, 'flores-dev').get_df()

    src_lang, dst_lang = direction.split('-')
    df['text'] = df[src_lang]
    df['src_lang'] = src_lang.replace('mhr', 'mhr').replace('tat', 'tt').replace('kaz', 'kk')  # TODO: make it customizable for all models
    df['dst_lang'] = dst_lang.replace('mhr', 'mhr').replace('tat', 'tt').replace('kaz', 'kk')

    batches = [batch.copy() for batch in batch_split(df, batch_size=8)]
    for i, batch in enumerate(tqdm(batches)):
        try:
            results = model.translate_batch(batch)
            # cache results
            with open(logs_file, 'a', encoding='UTF-8') as file:
                for row, translation in zip(batch, results):
                    row['translation'] = translation
                    print(row, file=file)
        except Exception as ex:
            print(f'WARN: Failed to process batch {i}. Failed with exception: {ex}')
