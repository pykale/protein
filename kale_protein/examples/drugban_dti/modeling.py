from kale_protein.auto.auto_predictor import MultiStreamProteinModel
from kale_protein.auto.weights import resolve_pretrained_weight
from kale_protein.registry import RUNNER_REGISTRY


class DrugBANModel:
    config_class = None

    def __init__(self, config, pretrain=False):
        self.config = config
        self.pretrain = pretrain
        self.weight_path = resolve_pretrained_weight(config) if pretrain else None
        self.model = MultiStreamProteinModel(config)

    def __iter__(self):
        yield self.component("target")
        yield self.component("drug")

    def component(self, stream_name):
        return self.model.component(stream_name)

    def embed(self, stream_data):
        return self.model.embed(stream_data)

    def __call__(self, batch):
        return self.model(batch)


class DrugBANInteractionPredictor:
    config_class = None

    def __init__(self, config, pretrain=False):
        self.config = config
        self.pretrain = pretrain
        self.weight_path = resolve_pretrained_weight(config) if pretrain else None
        self.model = MultiStreamProteinModel(config)
        self.runner = RUNNER_REGISTRY.get(config["runner"])(model=self.model, config=config)

    def __call__(self, protein_embedding, molecule_embedding):
        return self.runner(protein_embedding, molecule_embedding)

    def predict(self, batch_or_dataset):
        return self.runner.predict(batch_or_dataset)

    def fit(self, train_data, valid_data=None):
        return self.runner.fit(train_data, valid_data)
