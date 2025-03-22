import numpy as np
import scipy.stats as st
import evaluate
from transformers import logging
from transformers import pipeline
from collections import defaultdict

from scripts.helpers.model_manager import BaseModel


class Metric:
    metric_name: str = 'base_metric_class'
    supported_langs: set or str = 'all'

    def __call__(self, sources: list[str], targets: list[str], translations: list[str], bootstrap: int or None = None, confidence: float = 0.95, *args, **kwargs):
        pass

    def lang_is_supported(self, lang: str) -> bool:
        if self.supported_langs == 'all':
            return True
        return lang in self.supported_langs

    @staticmethod
    def bootstrap_input_data(sources: list[str], targets: list[str], translations: list[str], bootstrap_n: int or None = 300):
        if bootstrap_n is None or not isinstance(bootstrap_n, int) or bootstrap_n <= 1:
            yield sources, targets, translations
            return

        assert len(sources) == len(targets) == len(translations), 'Input data is not equal length'
        input_size = len(sources)

        for _ in range(bootstrap_n):
            batch = defaultdict(list)
            for sample in np.random.randint(0, input_size, input_size):
                batch['sources'].append(sources[sample])
                batch['targets'].append(targets[sample])
                batch['translations'].append(translations[sample])
            yield batch['sources'], batch['targets'], batch['translations']
        return



class HFMetric(Metric):
    def __init__(self, metric_name: str, **kwargs):
        self.metric = evaluate.load(metric_name, **kwargs)
        self.metric_name = metric_name

    def __call__(self, sources: list[str], targets: list[str], translations: list[str], score_only: bool = True, bootstrap: int or None = None, confidence: float = 0.95, *args, **kwargs):
        max_len = max(
            0 if sources is None else len(sources),
            0 if targets is None else len(targets),
            0 if translations is None else len(translations),
        )
        sources = [None] * max_len if sources is None else sources
        targets = [None] * max_len if targets is None else targets
        translations = [None] * max_len if translations is None else translations

        results = []
        for srcs, trgs, trns in self.bootstrap_input_data(sources, targets, translations, bootstrap_n=bootstrap):
            res = self.metric.compute(predictions=trns, references=trgs)
            results.append(self.extract_score(res) if score_only else res)
        if len(results) == 1:
            return results[0]
        # calculate "expected" value (without bootstrap)
        res = self.metric.compute(predictions=translations, references=targets)
        score = self.extract_score(res) if score_only else res
        return score, st.t.interval(confidence, len(results)-1, loc=np.mean(results), scale=st.sem(results))

    def extract_score(self, result: dict) -> float:
        raise NotImplementedError('Each HFMetric should specify custom extract_score function, otherwise set `score_only = False`')


class HFMetricModel(Metric):
    def __init__(self, model_repo: str, model_pipeline: str, keep_loaded: bool = False):
        self.metric_name = self.__class__.__name__

        self.model_repo = model_repo
        self.model_pipeline = model_pipeline
        self.keep_loaded = keep_loaded
        self.pipe = None
        logging.set_verbosity_error()

    def load_model(self):
        self.pipe = pipeline(self.model_pipeline, model=self.model_repo)

    def unload_model(self):
        del self.pipe
        self.pipe = None
        BaseModel.cleanup()

    def __call__(self, sources: list[str], targets: list[str], translations: list[str], score_only: bool = True, *args, **kwargs):
        if self.pipe is None:
            self.load_model()

        res = self.pipe(translations)

        if not self.keep_loaded:
            self.unload_model()

        return self.extract_score(res) if score_only else res

    def extract_score(self, result: dict) -> float:
        raise NotImplementedError('Each HFMetricModel should specify custom extract_score function, otherwise set `score_only = False`')


class FluencyRU(HFMetricModel):
    supported_langs = {'ru', 'rus'}

    def __init__(self, keep_loaded: bool = False):
        super().__init__(model_repo='RussianNLP/ruRoBERTa-large-rucola', model_pipeline='text-classification', keep_loaded=keep_loaded)
        self.metric_name = 'fluency-ru'

    def extract_score(self, result: list[dict]) -> float:
        return float(np.mean([row['score'] for row in result]))


class BLEU(HFMetric):
    def __init__(self):
        super().__init__("bleu")

    def extract_score(self, result: dict) -> float:
        return result['bleu']


class BLEUConf(HFMetric):
    def __init__(self):
        super().__init__('bleu')


class BLEURT(HFMetric):
    def __init__(self):
        super().__init__("bleurt", module_type="metric")

    def extract_score(self, result: dict) -> float:
        return np.mean(result['scores'])


class GoogleBLEU(HFMetric):
    def __init__(self):
        super().__init__('google_bleu')

    def extract_score(self, result: dict) -> float:
        return result['google_bleu']


class BertScore(HFMetric):
    def __init__(self):
        super().__init__("bertscore")

    def __call__(self, sources: list[str], targets: list[str], translations: list[str], score_only: bool = True, *args, **kwargs):
        assert 'lang' in kwargs, 'You MUST specify `lang: str = "iso-code" as parameter`'
        res = self.metric.compute(predictions=targets, references=translations, lang=kwargs['lang'])
        return self.extract_score(res) if score_only else res

    def extract_score(self, result: dict) -> float:
        return np.mean(result['f1'])


class CHRF(HFMetric):
    def __init__(self):
        super().__init__("chrf")

    def extract_score(self, result: dict) -> float:
        return result['score']

