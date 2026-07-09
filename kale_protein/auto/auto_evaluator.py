import kale_protein  # noqa
from kale_protein.registry import EVALUATOR_REGISTRY
class AutoProteinEvaluator:
    @classmethod
    def from_config(cls, config): return MultiMetricEvaluator(config)
class MultiMetricEvaluator:
    def __init__(self, config):
        self.config=config; self.metrics=[]
        for name in config.get('evaluation',{}).get('metrics',[]): self.metrics.append((name, EVALUATOR_REGISTRY.get((config['task'], name))()))
    def evaluate(self, outputs, data): return {n:m(outputs,data) for n,m in self.metrics}
