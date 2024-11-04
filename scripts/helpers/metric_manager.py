import evaluate


class Metric:
    def __call__(self, sources: list[str], references: list[str], translations: list[str]):
        pass


class HFMetric(Metric):
    def __init__(self, metric_name: str, **kwargs):
        self.metric = evaluate.load(metric_name, **kwargs)
        self.metric_name = metric_name

    def __call__(self, sources: list[str], targets: list[str], translations: list[str]):
        return self.metric.compute(predictions=targets, references=translations, sources=sources)


class BLEU(HFMetric):
    def __init__(self):
        super().__init__("bleu")


class BLEURT(HFMetric):
    def __init__(self):
        super().__init__("bleurt", module_type="metric")


class GoogleBLEU(HFMetric):
    def __init__(self):
        super().__init__('google_bleu')


class BertScore(HFMetric):
    def __init__(self):
        super().__init__("bertscore")


class CHRF(HFMetric):
    def __init__(self):
        super().__init__("chrf")

