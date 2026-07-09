import kale_protein  # noqa
from kale_protein.registry import INTERPRETER_REGISTRY
class AutoProteinInterpreter:
    @classmethod
    def from_config(cls, config): return INTERPRETER_REGISTRY.get((config['task'], config['interpretation']['method']))(config)
