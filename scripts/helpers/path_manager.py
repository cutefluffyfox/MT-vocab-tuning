import os


class PathHolder:
    root_path: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dir_path: str

    @staticmethod
    def create_folder(file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


class DatasetManager(PathHolder):
    dir_path = 'datasets'

    def get_path(self, lang: str, dataset_name: str) -> str:
        return os.path.join(self.root_path, self.dir_path, lang, dataset_name)


class DataManager(PathHolder):
    dir_path = 'data'

    def get_path(self, exp_type: str, lang: str, file_name: str, create: bool = True) -> str:
        path = os.path.join(self.root_path, self.dir_path, exp_type, lang, file_name)
        if create:
            self.create_folder(path)
        return path
