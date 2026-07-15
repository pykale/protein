# Extending KaleProtein

KaleProtein separates generic Auto dispatch, reusable core components, and
complete named models. Adding a model normally changes only one model-card
directory.

## Add Reusable Data

Task datasets belong in `core/data/tasks/<task>/datasets.py` and register a stable
id:

```python
from kale_protein.core.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("DTI/MyDataset")
def load_my_dataset(path, split="train"):
    return MyDataset(path=path, split=split)
```

Every compatible model can then use:

```python
data = AutoProteinData("DTI/MyDataset", path="data.csv", split="test")
```

Dataset loaders should normalize task fields and preserve provenance. They
must not import a concrete model.

## Add A Reusable Preprocessor

Modality processors belong in `core/data/modalities/<modality>/processors.py`:

```python
from kale_protein.core.registry import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register(
    "sequence/my_tokenizer",
    aliases=("protein/my_sequence",),
)
class MyTokenizer:
    default_input_key = "sequence"

    def __init__(self, input_key="sequence", max_length=None):
        self.input_key = input_key
        self.max_length = max_length

    def transform(self, sample):
        sequence = sample[self.input_key]
        if self.max_length is not None:
            sequence = sequence[: self.max_length]
        return {"sequence": sequence, "tokens": encode(sequence)}
```

## Add Reusable Model Components

Reusable modality encoders register as embedders. Reusable task fusion and
heads register as predictors. Put them in `core/modeling/modalities/` and
`core/modeling/tasks/`, respectively:

```python
from torch import nn

from kale_protein.auto import AutoProteinEmbedder, AutoProteinPredictor


@AutoProteinEmbedder.register("sequence/my_encoder")
class MySequenceEncoder(nn.Module):
    def __init__(self, config=None, hidden_dim=128):
        super().__init__()
        self.encoder = build_encoder(hidden_dim)

    def embed(self, tokens, mask, **metadata):
        return {
            "sequence_embedding": self.encoder(tokens),
            "sequence_mask": mask,
            **metadata,
        }


@AutoProteinPredictor.register("classification/my_head")
class MyClassificationHead(nn.Module):
    def __init__(self, config=None, hidden_dim=128, classes=2):
        super().__init__()
        self.output = nn.Linear(hidden_dim, classes)

    def forward(self, sequence_embedding, sequence_mask=None, **metadata):
        return {
            "logits": self.output(sequence_embedding),
            "sequence_mask": sequence_mask,
            **metadata,
        }
```

Put a component in core only when it is genuinely useful to more than one
named model. Model-specific layers can register from the example's
`modeling.py` instead.

## Add A Complete Model Card

Use the same simple layout for every model:

```text
kale_protein/examples/my_model/
  __init__.py
  config.yaml
  configuration.py
  modeling.py
  README.md
  data/
  maps/
  weights/
```

### Configuration

```yaml
model_id: MyTask/MyModel
model_type: my_model
name: my_model
task: classification
objective: discriminative

auto_map:
  AutoProteinConfig: configuration.MyModelConfig
  AutoProteinModel: modeling.MyModel

pretrained:
  local_dir: weights
  filename: my_model.pt
  url: ""

streams:
  sequence:
    modality: sequence
    input_key: sequence
    processor: my_tokenizer
    processor_kwargs:
      max_length: 512

components:
  embedders:
    sequence:
      id: sequence/my_encoder
      kwargs:
        hidden_dim: 128
  predictor:
    id: classification/my_head
    kwargs:
      hidden_dim: 128
      classes: 2
```

### Configuration Class

```python
from kale_protein.auto import AutoProteinConfig


class MyModelConfig(AutoProteinConfig):
    model_type = "my_model"
```

### Complete Model

The full model is the composition root and sole full-checkpoint owner:

```python
from torch import nn

from kale_protein.auto import AutoProteinEmbedder, AutoProteinPredictor
from kale_protein.core.weights import load_checkpoint_state_dict, resolve_pretrained_weight


class MyModel(nn.Module):
    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = config
        self.embedder = AutoProteinEmbedder.from_config(
            config.get_embedders()["sequence"], config=config
        )
        self.predictor = AutoProteinPredictor.from_config(
            config.get_predictor(), config=config
        )
        if pretrain:
            path = resolve_pretrained_weight(config)
            state = load_checkpoint_state_dict(path)
            self.load_state_dict(adapt_checkpoint_keys(state), strict=True)

    def embed(self, tokens, mask, **metadata):
        return self.embedder.embed(tokens=tokens, mask=mask, **metadata)

    def forward(self, **batch):
        embeddings = self.embed(**batch)
        return self.predictor(**embeddings)
```

Every stage should return a dictionary. The next stage consumes it with
`**mapping`, so its field names form a normal, inspectable Python API. Choose
semantic names such as `sequence_embedding`, `structure_mask`, `labels`, or
`sample_ids`; Auto performs discovery and construction but does not hardcode a
model's intermediate tensor schema.

For a generative model, the predictor implements `generate()` and the full
model may expose a convenience `generate()` method. Do not add a second Auto
class that constructs another complete generator model.

## Register An External Card

Bundled cards are discovered from `config.yaml`. Register an external card
without editing Auto:

```python
from kale_protein.core.registry import register_model_card

register_model_card("path/to/my_model/config.yaml")
model = AutoProteinModel("MyTask/MyModel")
```

Never add model ids, model-name branches, or architecture tables to `auto/`.

## Checkpoints

Use `core.weights` for local-first resolution, optional checksum verification,
atomic downloads, and common checkpoint extraction. Keep key conversion in the
model card and require strict loading. Nested embedders and predictors should
not independently resolve the complete model checkpoint.

## Scripts And Tests

Each runnable script should show its real stages directly:

```text
load -> preprocess -> collate -> embed -> train/predict/generate -> metric/interpret
```

Tests should use temporary files, fake URLs and downloaders, tiny tensors, and
small model dimensions. Cover component construction, one optimizer step,
checkpoint round trips, incompatible checkpoints, metrics, generation, and
script import safety.

```bash
python -m pytest -q
python -m compileall -q kale_protein
python -m build
```
