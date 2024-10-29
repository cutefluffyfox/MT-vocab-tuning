import os
import shutil
import requests
from zipfile import ZipFile

import pandas as pd
from datasets import load_dataset as load_hf_dataset

from scripts.helpers.path_manager import DatasetManager


class MTDataset:
    def __init__(self, direction: str, name: str):
        self.direction = direction
        self.name = name
        self.csv_name = name + '.csv'
        self.dm = DatasetManager()
        os.makedirs(os.path.dirname(self.dm.get_path(direction, 'DELETE_ME')), exist_ok=True)

    def get_df(self) -> pd.DataFrame:
        return pd.read_csv(self.dm.get_path(self.direction, self.csv_name))

    def save_df(self, df: pd.DataFrame):
        file_path = self.dm.get_path(self.direction, self.csv_name)
        df.to_csv(file_path)

    def is_cached(self) -> bool:
        return os.path.exists(self.dm.get_path(self.direction, self.csv_name))


class HFDataset(MTDataset):
    def __init__(self, direction: str, name: str):
        super().__init__(direction, name)

    def download_parquet(self, link: str, use_cache: bool = True):
        if not use_cache or not self.is_cached():
            self.save_df(pd.read_parquet(link))

    def download_json(self, link: str, use_cache: bool = True):
        if not use_cache or not self.is_cached():
            self.save_df(pd.read_json(link, lines=True))

    def download_csv(self, link: str, use_cache: bool = True):
        if not use_cache or not self.is_cached():
            self.save_df(pd.read_csv(link))

    def download_datasets(self, repo: str, table: str, use_cache: bool = True):
        if not use_cache or not self.is_cached():
            ds = load_hf_dataset(repo)
            self.save_df(ds[table].to_pandas())


class FloresDataset(MTDataset):
    def __init__(self, direction: str, name: str):
        super().__init__(direction, name)

    def download_from_github(self, eval_type: str, langs: list[str], release: str = 'v2.0-rc.3', use_cache: bool = True):
        if use_cache and self.is_cached():
            return

        # download release from GitHub
        res = requests.get(f'https://github.com/openlanguagedata/flores/releases/download/{release}/floresp-{release}.zip')
        unzip_path = self.dm.get_path(self.direction, self.name)
        zip_file_path = unzip_path + '.zip'

        # save zip archive
        with open(zip_file_path, 'wb') as file:
            file.write(res.content)

        # extract data from archive
        with ZipFile(zip_file_path) as zf:
            zf.extractall(path=unzip_path, pwd=b'multilingual machine translation')

        # read files we interested in
        parsed = dict()
        for lang in langs:
            with open(os.path.join(unzip_path, f'floresp-{release}', eval_type, f"{eval_type}.{lang}"), encoding='utf-8') as file:
                parsed[lang.split('_')[0]] = [line.strip() for line in file.readlines()]

        # combine and save
        df = pd.DataFrame(parsed)
        self.save_df(df)

        # remove all tmp files
        os.remove(zip_file_path)
        shutil.rmtree(unzip_path)
