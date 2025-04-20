class BaseTokenizer:
    def pretokenize(self, text: str):
        raise NotImplementedError('Tokenizers must have `.pretokenize` method implemented')

    def tokenize(self, text: str):
        raise NotImplementedError('Tokenizers must have `.tokenize` method implemented')

    def save(self, file_name: str = None):
        raise NotImplementedError('Tokenizers must have `.save` method implemented')

    def load(self, file_name: str):
        raise NotImplementedError('Tokenizers must have `.load` method implemented')
