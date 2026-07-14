# Extending KaleProtein

KaleProtein separates reusable data and modality behavior from named model
implementations. The public pipeline remains explicit:

```text
load -> preprocess -> collate -> embed/model -> predict or generate -> evaluate -> interpret
```

## Add a Dataset

Register task-level loaders with `DATASET_REGISTRY`. A loader should normalize
records to stable task fields and accept paths or options as ordinary keyword
arguments.

```python
from kale_protein.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("DTI/MyDataset")
def load_my_dataset(path, split="train"):
    return MyDataset(path=path, split=split)
```

It is then available to every model for that task:

```python
from kale_protein.auto import AutoProteinData

data = AutoProteinData("DTI/MyDataset", path="data.csv", split="test")
```

Dataset code must not import a particular model card. Pair construction, split
semantics, labels, and task metadata belong here; neural featurization does not.

## Add a Reusable Preprocessor

Processors own one modality and register a canonical key plus optional friendly
aliases.

```python
from kale_protein.registry import MODALITY_PROCESSOR_REGISTRY


@MODALITY_PROCESSOR_REGISTRY.register(
    ("protein_sequence", "my_tokenizer"),
    aliases=(("protein/my_sequence", "sequence"),),
)
class MyTokenizer:
    def __init__(self, input_key="sequence", max_length=None):
        self.input_key = input_key
        self.max_length = max_length

    def transform(self, sample):
        sequence = sample[self.input_key]
        if self.max_length is not None:
            sequence = sequence[: self.max_length]
        return {"sequence": sequence, "tokens": encode(sequence)}
```

```python
processor = AutoProteinPreprocessor("protein/my_sequence", max_length=512)
tokens = processor.tokenize({"sequence": "MKT..."})
```

A modality processor must not read another stream or construct a complete model
batch. Cross-stream collation belongs to task or model-card code.

## Add a Model Card

Named model definitions belong together in one self-contained directory:

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

The entry files use the same names across model cards. Supporting layer modules
may sit beside `modeling.py` when an architecture is large.

### Configuration

Declare a globally unique `model_id` and map Auto APIs to card-local classes:

```yaml
model_id: MyTask/MyModel
model_type: my_model
name: my_model
task: my_task
objective: discriminative
runner: predict

auto_map:
  AutoProteinConfig: configuration.MyModelConfig
  AutoProteinModel: modeling.MyModel
  AutoProteinPredictor: modeling.MyPredictor

pretrained:
  local_dir: weights
  filename: my_model.pt
  url: ""

streams:
  sequence:
    modality: protein_sequence
    input_key: sequence
    processor: my_tokenizer
    encoder: my_sequence_encoder

fusion:
  type: model_owned
head:
  type: model_owned
```

The `encoder`, `fusion`, and `head` fields describe the card for inspection; they
do not require named architecture classes to be registered in the shared Auto
layer. `modeling.py` constructs the real learnable network.

### Configuration class

```python
from kale_protein.auto import AutoProteinConfig


class MyModelConfig(AutoProteinConfig):
    model_type = "my_model"
```

### Model classes

```python
from torch import nn


class MyModel(nn.Module):
    config_class = MyModelConfig

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = config
        self.network = build_network(config)
        if pretrain:
            self.load_pretrained(config)

    def embed(self, processed):
        return self.network.encode(processed)


class MyPredictor(nn.Module):
    config_class = MyModelConfig

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.model = MyModel(config, pretrain=pretrain)

    def forward(self, embeddings):
        return self.model.predict(embeddings)
```

The exact component API should fit the task. A multimodal card may expose
separate embedders; a generative card should map `AutoProteinGenerator` and
provide `generate()`.

## Register a Card

Bundled cards under `kale_protein/` are discovered from their `config.yaml`
metadata. External cards can be registered without editing Auto code:

```python
from kale_protein.registry import register_model_card

register_model_card("path/to/my_model/config.yaml")
model = AutoProteinModel("MyTask/MyModel")
```

Do not add model IDs, model-name branches, or architecture dictionaries to
`kale_protein.auto`.

## Pretrained Weights

Use the generic helpers from `kale_protein.auto.weights` to resolve assets and
extract common checkpoint containers:

```python
from kale_protein.auto.weights import (
    load_pretrained_state_dict,
    resolve_pretrained_weight,
)

path = resolve_pretrained_weight(config)
state_dict = load_pretrained_state_dict(config)
model.load_state_dict(state_dict, strict=True)
```

Keep architecture-specific key conversion in the model card. Never use
`strict=False` merely to make an unrelated released checkpoint appear to load.

## Workflow Scripts

Provide separate, import-safe scripts with `main(argv=None)` and a main guard.
Each script should show its own real stages directly. For example, evaluation
loads held-out data, preprocesses and collates it, embeds inputs, predicts, and
computes metrics. It should not invoke a generic workflow wrapper or perform a
training loop.

## Tests

Use tiny temporary files, fake URLs/downloaders, fake chemistry objects, and
small model dimensions. Cover the boundaries rather than downloading large
assets:

- dynamic `auto_map` dispatch
- preprocessing and collation schemas
- one optimizer step changes a parameter
- checkpoint round trips and incompatibility errors
- known metric values
- iterative generation trajectories
- script import safety
- absence of upstream runtime imports

Run:

```bash
python -m pytest -q
python -m compileall -q kale_protein
python -m build
```
