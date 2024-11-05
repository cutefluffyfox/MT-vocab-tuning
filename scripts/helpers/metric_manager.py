import numpy as np
import evaluate


class Metric:
    def __call__(self, sources: list[str], references: list[str], translations: list[str], *args, **kwargs):
        pass


class HFMetric(Metric):
    def __init__(self, metric_name: str, **kwargs):
        self.metric = evaluate.load(metric_name, **kwargs)
        self.metric_name = metric_name

    def __call__(self, sources: list[str], targets: list[str], translations: list[str], score_only: bool = True, *args, **kwargs):
        res = self.metric.compute(predictions=targets, references=translations)
        return self.extract_score(res) if score_only else res

    def extract_score(self, result: dict) -> float:
        raise NotImplementedError('Each HFMetric should specify custom extract_score function, otherwise set `score_only = False`')


class BLEU(HFMetric):
    def __init__(self):
        super().__init__("bleu")

    def extract_score(self, result: dict) -> float:
        return result['bleu']


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

