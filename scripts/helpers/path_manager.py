import os


class PathHolder:
    root_path: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dir_path: str


class DatasetManager(PathHolder):
    dir_path = 'datasets'

    def get_path(self, lang: str, dataset_name: str):
        return os.path.join(self.root_path, self.dir_path, lang, dataset_name)


