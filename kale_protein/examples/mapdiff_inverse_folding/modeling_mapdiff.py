from kale_protein.auto.auto_predictor import MultiStreamProteinModel
from kale_protein.auto.weights import resolve_pretrained_weight
from configuration_mapdiff import MapDiffConfig
from kale_protein.registry import RUNNER_REGISTRY


class MapDiffModel:
    config_class = MapDiffConfig

    def __init__(self, config, pretrain=False):
        self.config = config
        self.pretrain = pretrain
        self.weight_path = resolve_pretrained_weight(config) if pretrain else None
        self.model = MultiStreamProteinModel(config)

    def component(self, stream_name):
        return self.model.component(stream_name)

    def embed(self, stream_data):
        return self.model.embed(stream_data)

    def __call__(self, batch):
        return self.model(batch)

    def sample(self, batch, sampling_config):
        return self.model.sample(batch, sampling_config)


class MapDiffGenerator:
    config_class = MapDiffConfig

    def __init__(self, config, pretrain=False):
        self.config = config
        self.pretrain = pretrain
        self.weight_path = resolve_pretrained_weight(config) if pretrain else None
        self.model = MultiStreamProteinModel(config)
        self.runner = RUNNER_REGISTRY.get(config["runner"])(model=self.model, config=config)

    def __call__(self, structure_embedding):
        return self.generate(structure_embedding)

    def generate(self, structure_embedding):
        return self.runner.generate(structure_embedding)
