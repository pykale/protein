下面是一份可以直接给 Codex 的完整执行计划。目标是把 **KaleProtein** 做成基于 PyKale 六步 workflow 的 **Auto/registry 框架**，并把 **DrugBAN** 和 **MapDiff** 作为两个首批 example/preset。

---

# KaleProtein implementation plan for Codex

## 0. Overall goal

Implement a modular KaleProtein framework inside the existing PyKale-style repository.

The framework should follow the PyKale six-step workflow:

```text
Load Data → Preprocess Data → Embed → Predict/Generate → Evaluate → Interpret
```

But unlike a model-centric design, KaleProtein should be **stream-based** and **registry-driven**.

The first version should support two example workflows:

```text
1. DrugBAN
   Task: drug–target interaction prediction
   Type: discriminative / predictive
   Modalities:
       - small molecule stream
       - protein sequence stream
   Fusion:
       - bilinear attention
   Output:
       - binary DTI prediction or affinity score

2. MapDiff
   Task: structure-conditioned inverse folding / protein sequence generation
   Type: generative / diffusion-like
   Modalities:
       - protein structure stream
       - noisy/masked protein sequence stream
   Conditioning:
       - structure-conditioned denoising
   Output:
       - generated protein sequence
```

The framework should not hard-code DrugBAN or MapDiff as monolithic models. They should be implemented as **presets/recipes** composed of stream processors, stream encoders, fusion or conditioner modules, task heads, runners, evaluators, and interpreters.

---

# 1. Core design principle

Use a two-level abstraction.

```text
Level 1: PyKale-style Auto workflow classes

    AutoProteinDataLoader
    AutoProteinPreprocessor
    AutoProteinEmbedder
    AutoProteinPredictor
    AutoProteinEvaluator
    AutoProteinInterpreter

Level 2: Modality/task registries

    modality processor registry
    modality encoder registry
    fusion registry
    conditioner registry
    head registry
    runner registry
    evaluator registry
    interpreter registry
    preset registry
```

The top-level API should be simple:

```python
from kale_protein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinPreprocessor,
    AutoProteinPredictor,
    AutoProteinEvaluator,
    AutoProteinInterpreter,
)

config = AutoProteinConfig.from_preset("drugban")

loader = AutoProteinDataLoader.from_config(config)
dataset = loader.load()

preprocessor = AutoProteinPreprocessor.from_config(config)
dataset = preprocessor.transform_dataset(dataset)

predictor = AutoProteinPredictor.from_config(config)
outputs = predictor.predict(dataset)

evaluator = AutoProteinEvaluator.from_config(config)
metrics = evaluator.evaluate(outputs, dataset)

interpreter = AutoProteinInterpreter.from_config(config)
explanations = interpreter.explain(predictor, dataset)
```

For MapDiff:

```python
config = AutoProteinConfig.from_preset("mapdiff")

loader = AutoProteinDataLoader.from_config(config)
dataset = loader.load()

preprocessor = AutoProteinPreprocessor.from_config(config)
dataset = preprocessor.transform_dataset(dataset)

predictor = AutoProteinPredictor.from_config(config)
generated = predictor.generate(dataset)

evaluator = AutoProteinEvaluator.from_config(config)
metrics = evaluator.evaluate(generated, dataset)
```

---

# 2. Recommended directory structure

Create the following package structure. Adjust the root package name according to the existing repo convention.

```text
kale_protein/
  __init__.py

  auto/
    __init__.py
    config.py
    auto_loader.py
    auto_preprocessor.py
    auto_embedder.py
    auto_predictor.py
    auto_evaluator.py
    auto_interpreter.py

  registry/
    __init__.py
    base.py
    presets.py
    modality_processors.py
    modality_encoders.py
    fusion.py
    conditioners.py
    heads.py
    runners.py
    evaluators.py
    interpreters.py

  modalities/
    __init__.py

    small_molecule/
      __init__.py
      processors.py
      encoders.py
      collators.py

    protein_sequence/
      __init__.py
      processors.py
      encoders.py
      collators.py

    protein_structure/
      __init__.py
      processors.py
      encoders.py
      collators.py

  fusion/
    __init__.py
    concat.py
    bilinear_attention.py
    cross_attention.py

  conditioners/
    __init__.py
    structure_conditioned_denoising.py

  heads/
    __init__.py
    classification.py
    regression.py
    diffusion_sequence_decoder.py

  runners/
    __init__.py
    predict_runner.py
    train_predict_runner.py
    generate_runner.py
    diffusion_generate_runner.py
    structure_predict_runner.py

  tasks/
    __init__.py

    drug_target_interaction/
      __init__.py
      datasets.py
      metrics.py
      losses.py
      samplers.py
      interpreters.py

    inverse_folding/
      __init__.py
      datasets.py
      metrics.py
      losses.py
      samplers.py
      interpreters.py

  presets/
    drugban.yaml
    mapdiff.yaml

  examples/
    drugban_dti/
      config.yaml
      train.py
      evaluate.py
      interpret.py
      README.md

    mapdiff_inverse_folding/
      config.yaml
      generate.py
      evaluate.py
      README.md

  tests/
    test_registry.py
    test_config.py
    test_auto_preprocessor.py
    test_auto_predictor_drugban.py
    test_auto_predictor_mapdiff.py
```

If the existing repo already has a package layout such as `kale/`, place `kale_protein` under the appropriate PyKale namespace, for example:

```text
kale/protein/
```

Do not create a separate top-level package if the repository convention expects modules under `kale/`.

---

# 3. Config system

Implement `AutoProteinConfig`.

File:

```text
kale_protein/auto/config.py
```

Required features:

```python
class AutoProteinConfig:
    @classmethod
    def from_yaml(cls, path): ...
    
    @classmethod
    def from_dict(cls, config_dict): ...
    
    @classmethod
    def from_preset(cls, preset_name): ...
    
    def to_dict(self): ...
    
    def validate(self): ...
```

The config should support this schema:

```yaml
name: drugban
task: drug_target_interaction
objective: discriminative
runner: predict

streams:
  drug:
    modality: small_molecule
    input_key: smiles
    processor: rdkit_graph
    encoder: drugban_molecule_gnn
    processor_kwargs: {}
    encoder_kwargs: {}

  target:
    modality: protein_sequence
    input_key: sequence
    processor: amino_acid_tokenizer
    encoder: drugban_protein_cnn
    processor_kwargs: {}
    encoder_kwargs: {}

fusion:
  type: bilinear_attention
  kwargs: {}

head:
  type: binary_classifier
  kwargs:
    input_dim: 256
    num_classes: 1

loss:
  type: binary_cross_entropy

evaluation:
  metrics:
    - auroc
    - auprc
    - accuracy
    - f1

interpretation:
  method: bilinear_attention_map
```

MapDiff schema:

```yaml
name: mapdiff
task: inverse_folding
objective: generative
runner: diffusion_generate

streams:
  structure:
    modality: protein_structure
    input_key: backbone_coords
    processor: backbone_coordinate_processor
    encoder: mapdiff_structure_encoder
    processor_kwargs: {}
    encoder_kwargs: {}

  noisy_sequence:
    modality: protein_sequence
    input_key: sequence
    processor: masked_sequence_tokenizer
    encoder: residue_token_embedding
    processor_kwargs: {}
    encoder_kwargs: {}

conditioner:
  type: structure_conditioned_denoising
  kwargs: {}

head:
  type: diffusion_sequence_decoder
  kwargs: {}

sampling:
  steps: 100
  num_samples: 8
  temperature: 1.0

evaluation:
  metrics:
    - sequence_recovery
    - perplexity
    - diversity
    - novelty

interpretation:
  method: denoising_trajectory
```

Validation rules:

```text
1. config must have task
2. config must have objective
3. config must have runner
4. config must have streams
5. every stream must have:
   - modality
   - input_key
   - processor
   - encoder
6. discriminative tasks must have fusion or conditioner and head
7. generative tasks must have head and runner
8. if runner == diffusion_generate, sampling config should exist
```

---

# 4. Registry system

Implement a lightweight registry class.

File:

```text
kale_protein/registry/base.py
```

Expected API:

```python
class Registry:
    def __init__(self, name):
        self.name = name
        self._mapping = {}

    def register(self, key, value=None):
        ...

    def get(self, key):
        ...

    def has(self, key):
        ...

    def available_keys(self):
        ...
```

It should support both direct registration:

```python
MODALITY_ENCODER_REGISTRY.register(
    ("protein_sequence", "drugban_protein_cnn"),
    DrugBANProteinCNNEncoder,
)
```

and decorator registration:

```python
@MODALITY_ENCODER_REGISTRY.register(("protein_sequence", "esm2"))
class ESM2Encoder:
    ...
```

Create registry instances:

```text
kale_protein/registry/modality_processors.py
kale_protein/registry/modality_encoders.py
kale_protein/registry/fusion.py
kale_protein/registry/conditioners.py
kale_protein/registry/heads.py
kale_protein/registry/runners.py
kale_protein/registry/evaluators.py
kale_protein/registry/interpreters.py
kale_protein/registry/presets.py
```

Suggested registry keys:

```python
# Processor registry
(modality, processor_name)

# Encoder registry
(modality, encoder_name)

# Fusion registry
fusion_name

# Conditioner registry
conditioner_name

# Head registry
(task, head_name)

# Runner registry
runner_name

# Evaluator registry
(task, metric_name)

# Interpreter registry
(task, method_name)

# Preset registry
preset_name
```

---

# 5. Stream abstraction

Implement a `StreamSpec` dataclass.

File:

```text
kale_protein/auto/config.py
```

Suggested structure:

```python
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class StreamSpec:
    name: str
    modality: str
    input_key: str
    processor: str
    encoder: str
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False
```

Implement helper method:

```python
config.get_streams()
```

This should return:

```python
Dict[str, StreamSpec]
```

---

# 6. AutoProteinPreprocessor

File:

```text
kale_protein/auto/auto_preprocessor.py
```

Responsibilities:

```text
1. Read streams from config.
2. For each stream:
   - find processor class by (modality, processor)
   - instantiate processor
3. Apply each processor independently.
4. Preserve task-level fields such as label, domain, id, metadata.
```

Expected implementation:

```python
class AutoProteinPreprocessor:
    @classmethod
    def from_config(cls, config):
        return MultiStreamPreprocessor(config)


class MultiStreamPreprocessor:
    def __init__(self, config):
        self.config = config
        self.processors = {}

        for stream_name, stream in config.get_streams().items():
            processor_cls = MODALITY_PROCESSOR_REGISTRY.get(
                (stream.modality, stream.processor)
            )
            self.processors[stream_name] = processor_cls(
                input_key=stream.input_key,
                **stream.processor_kwargs,
            )

    def transform_sample(self, sample):
        output = {}

        for stream_name, processor in self.processors.items():
            output[stream_name] = processor.transform(sample)

        for key in ["label", "domain", "id", "metadata"]:
            if key in sample:
                output[key] = sample[key]

        return output

    def transform_dataset(self, dataset):
        return [self.transform_sample(sample) for sample in dataset]
```

Important boundary:

```text
Stream processors should not know other streams.

Small molecule processor should not know protein sequence.
Protein sequence processor should not know small molecule graph.
Pair construction, negative sampling, and cold split are task-level operations, not modality-level operations.
```

---

# 7. Modality processors

## 7.1 Small molecule processors

File:

```text
kale_protein/modalities/small_molecule/processors.py
```

Implement at least:

```python
class SMILESTokenizer:
    def __init__(self, input_key="smiles", **kwargs): ...
    def transform(self, sample): ...
```

```python
class RDKitGraphProcessor:
    def __init__(self, input_key="smiles", **kwargs): ...
    def transform(self, sample): ...
```

First version can be lightweight. If RDKit is unavailable, provide a clear error message.

Output schema for molecule graph:

```python
{
    "smiles": "...",
    "node_features": ...,
    "edge_index": ...,
    "edge_features": ...,
}
```

For early testability, allow a dummy fallback processor:

```python
class DummyMoleculeGraphProcessor:
    ...
```

## 7.2 Protein sequence processors

File:

```text
kale_protein/modalities/protein_sequence/processors.py
```

Implement:

```python
class AminoAcidTokenizer:
    ...
```

Use standard amino acid vocabulary:

```text
A C D E F G H I K L M N P Q R S T V W Y
```

Also include special tokens if needed:

```text
<pad>, <unk>, <mask>, <bos>, <eos>
```

Output schema:

```python
{
    "sequence": "...",
    "tokens": LongTensor,
    "attention_mask": LongTensor,
}
```

Also implement:

```python
class MaskedSequenceTokenizer:
    ...
```

For MapDiff-like workflows.

## 7.3 Protein structure processors

File:

```text
kale_protein/modalities/protein_structure/processors.py
```

Implement:

```python
class BackboneCoordinateProcessor:
    ...
```

Expected input:

```python
sample["backbone_coords"]
```

Expected output:

```python
{
    "coords": Tensor,       # shape [L, A, 3] or [L, 3]
    "coord_mask": Tensor,   # shape [L]
}
```

Do not implement a full PDB parser in v0.1 unless the repo already has one. For v0.1, support pre-extracted coordinate arrays.

---

# 8. Modality encoders

## 8.1 Base interface

Each encoder should be a `torch.nn.Module`.

Expected method:

```python
def forward(self, batch):
    ...
```

The encoder receives only one stream batch.

## 8.2 DrugBAN molecule encoder

File:

```text
kale_protein/modalities/small_molecule/encoders.py
```

Implement:

```python
class DrugBANMoleculeGNNEncoder(nn.Module):
    ...
```

For v0.1, this may be a minimal GNN/MLP placeholder if full DrugBAN code migration is too large. Make the interface stable.

Expected output:

```python
{
    "embedding": Tensor,
    "token_embeddings": Optional[Tensor],
    "mask": Optional[Tensor],
}
```

## 8.3 DrugBAN protein encoder

File:

```text
kale_protein/modalities/protein_sequence/encoders.py
```

Implement:

```python
class DrugBANProteinCNNEncoder(nn.Module):
    ...
```

Expected output:

```python
{
    "embedding": Tensor,
    "token_embeddings": Tensor,
    "mask": Tensor,
}
```

## 8.4 MapDiff structure encoder

File:

```text
kale_protein/modalities/protein_structure/encoders.py
```

Implement:

```python
class MapDiffStructureEncoder(nn.Module):
    ...
```

For v0.1, this can be an adapter/wrapper if MapDiff code already exists, or a minimal placeholder with correct input/output schema.

## 8.5 Residue token embedding

File:

```text
kale_protein/modalities/protein_sequence/encoders.py
```

Implement:

```python
class ResidueTokenEmbedding(nn.Module):
    ...
```

Used for noisy/masked sequence stream in MapDiff.

---

# 9. Fusion and conditioner modules

## 9.1 Fusion modules

File:

```text
kale_protein/fusion/bilinear_attention.py
```

Implement:

```python
class BilinearAttentionFusion(nn.Module):
    def forward(self, stream_outputs):
        ...
```

Input:

```python
stream_outputs = {
    "drug": {
        "embedding": ...,
        "token_embeddings": ...,
        "mask": ...
    },
    "target": {
        "embedding": ...,
        "token_embeddings": ...,
        "mask": ...
    }
}
```

Output:

```python
{
    "embedding": fused_embedding,
    "attention": attention_map,
}
```

Also implement simple fallback fusions:

```python
class ConcatFusion(nn.Module): ...
class CrossAttentionFusion(nn.Module): ...
```

## 9.2 Conditioners

File:

```text
kale_protein/conditioners/structure_conditioned_denoising.py
```

Implement:

```python
class StructureConditionedDenoising(nn.Module):
    def forward(self, stream_outputs, timestep=None):
        ...
```

Used by MapDiff-like models.

Input:

```python
stream_outputs = {
    "structure": {...},
    "noisy_sequence": {...}
}
```

Output:

```python
{
    "hidden": hidden_representation,
    "conditioning": structure_conditioning,
}
```

---

# 10. Heads

## 10.1 Binary classifier

File:

```text
kale_protein/heads/classification.py
```

Implement:

```python
class BinaryClassificationHead(nn.Module):
    def forward(self, features):
        ...
```

Output:

```python
{
    "logits": logits,
    "probabilities": probs,
}
```

## 10.2 Regression head

File:

```text
kale_protein/heads/regression.py
```

Implement:

```python
class RegressionHead(nn.Module):
    ...
```

## 10.3 Diffusion sequence decoder

File:

```text
kale_protein/heads/diffusion_sequence_decoder.py
```

Implement interface:

```python
class DiffusionSequenceDecoder(nn.Module):
    def forward(self, hidden, timestep=None):
        ...
    
    def sample(self, hidden, sampling_config):
        ...
```

For v0.1, this may be an adapter/wrapper around MapDiff, or a minimal placeholder with deterministic toy sampling.

---

# 11. Predictor and runners

## 11.1 AutoProteinPredictor

File:

```text
kale_protein/auto/auto_predictor.py
```

Expected behavior:

```python
class AutoProteinPredictor:
    @classmethod
    def from_config(cls, config):
        runner_name = config["runner"]
        runner_cls = RUNNER_REGISTRY.get(runner_name)
        model = MultiStreamProteinModel(config)
        return runner_cls(model=model, config=config)
```

## 11.2 MultiStreamProteinModel

Implement:

```python
class MultiStreamProteinModel(nn.Module):
    def __init__(self, config):
        ...
    
    def encode_streams(self, batch):
        ...
    
    def forward(self, batch):
        ...
```

Construction logic:

```text
1. Build one encoder per stream.
2. If config has fusion, build fusion module.
3. If config has conditioner, build conditioner module.
4. Build task head.
```

Pseudo-code:

```python
class MultiStreamProteinModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict()

        for stream_name, stream in config.get_streams().items():
            encoder_cls = MODALITY_ENCODER_REGISTRY.get(
                (stream.modality, stream.encoder)
            )
            self.encoders[stream_name] = encoder_cls(**stream.encoder_kwargs)

        self.fusion = None
        if config.get("fusion") is not None:
            fusion_cls = FUSION_REGISTRY.get(config["fusion"]["type"])
            self.fusion = fusion_cls(**config["fusion"].get("kwargs", {}))

        self.conditioner = None
        if config.get("conditioner") is not None:
            conditioner_cls = CONDITIONER_REGISTRY.get(config["conditioner"]["type"])
            self.conditioner = conditioner_cls(**config["conditioner"].get("kwargs", {}))

        head_cls = HEAD_REGISTRY.get((config["task"], config["head"]["type"]))
        self.head = head_cls(**config["head"].get("kwargs", {}))

    def encode_streams(self, batch):
        outputs = {}
        for stream_name, encoder in self.encoders.items():
            outputs[stream_name] = encoder(batch[stream_name])
        return outputs

    def forward(self, batch):
        stream_outputs = self.encode_streams(batch)

        if self.fusion is not None:
            features = self.fusion(stream_outputs)
            return self.head(features)

        if self.conditioner is not None:
            features = self.conditioner(stream_outputs, timestep=batch.get("timestep"))
            return self.head(features)

        return self.head(stream_outputs)
```

## 11.3 Runners

Implement files:

```text
kale_protein/runners/predict_runner.py
kale_protein/runners/train_predict_runner.py
kale_protein/runners/generate_runner.py
kale_protein/runners/diffusion_generate_runner.py
kale_protein/runners/structure_predict_runner.py
```

Minimum v0.1:

```python
class PredictRunner:
    def __init__(self, model, config): ...
    def predict(self, batch_or_dataset): ...
    def fit(self, train_data, valid_data=None): ...
```

```python
class DiffusionGenerateRunner:
    def __init__(self, model, config): ...
    def generate(self, batch_or_dataset): ...
```

The unified predictor object should expose:

```python
predictor.predict(...)
predictor.generate(...)
predictor.score(...)
predictor.fit(...)
```

Unsupported methods should raise a clear error:

```text
This runner does not support generate().
Available methods: predict(), fit().
```

---

# 12. Evaluators

## 12.1 AutoProteinEvaluator

File:

```text
kale_protein/auto/auto_evaluator.py
```

Behavior:

```python
class AutoProteinEvaluator:
    @classmethod
    def from_config(cls, config):
        return MultiMetricEvaluator(config)
```

```python
class MultiMetricEvaluator:
    def __init__(self, config):
        self.metrics = []
        for metric_name in config["evaluation"]["metrics"]:
            metric_cls = EVALUATOR_REGISTRY.get((config["task"], metric_name))
            self.metrics.append((metric_name, metric_cls()))

    def evaluate(self, outputs, data):
        results = {}
        for name, metric in self.metrics:
            results[name] = metric(outputs, data)
        return results
```

## 12.2 DrugBAN metrics

File:

```text
kale_protein/tasks/drug_target_interaction/metrics.py
```

Implement:

```text
accuracy
f1
auroc
auprc
```

Use sklearn if available. If sklearn is missing, return clear error.

## 12.3 MapDiff metrics

File:

```text
kale_protein/tasks/inverse_folding/metrics.py
```

Implement:

```text
sequence_recovery
diversity
novelty
perplexity placeholder
```

Do not implement TM-score/RMSD in v0.1 unless dependencies already exist.

Add TODO hooks for:

```text
refolding_tm_score
rmsd
designability
self_consistency_tm
```

---

# 13. Interpreters

## 13.1 AutoProteinInterpreter

File:

```text
kale_protein/auto/auto_interpreter.py
```

Behavior:

```python
class AutoProteinInterpreter:
    @classmethod
    def from_config(cls, config):
        method = config["interpretation"]["method"]
        interpreter_cls = INTERPRETER_REGISTRY.get((config["task"], method))
        return interpreter_cls(config)
```

## 13.2 DrugBAN interpreter

File:

```text
kale_protein/tasks/drug_target_interaction/interpreters.py
```

Implement:

```python
class BilinearAttentionMapInterpreter:
    def explain(self, predictor, data):
        ...
```

Return attention map if the model exposes it.

## 13.3 MapDiff interpreter

File:

```text
kale_protein/tasks/inverse_folding/interpreters.py
```

Implement:

```python
class DenoisingTrajectoryInterpreter:
    def explain(self, predictor, data):
        ...
```

For v0.1, return saved trajectory only if runner/model stores it. Otherwise return a clear message or empty dict.

---

# 14. Presets

## 14.1 DrugBAN preset

File:

```text
kale_protein/presets/drugban.yaml
```

Content:

```yaml
name: drugban
task: drug_target_interaction
objective: discriminative
runner: predict

streams:
  drug:
    modality: small_molecule
    input_key: smiles
    processor: rdkit_graph
    encoder: drugban_molecule_gnn
    processor_kwargs: {}
    encoder_kwargs:
      hidden_dim: 128
      output_dim: 128

  target:
    modality: protein_sequence
    input_key: sequence
    processor: amino_acid_tokenizer
    encoder: drugban_protein_cnn
    processor_kwargs:
      max_length: 1000
    encoder_kwargs:
      hidden_dim: 128
      output_dim: 128

fusion:
  type: bilinear_attention
  kwargs:
    hidden_dim: 256

head:
  type: binary_classifier
  kwargs:
    input_dim: 256
    hidden_dim: 128
    output_dim: 1

loss:
  type: binary_cross_entropy

evaluation:
  metrics:
    - auroc
    - auprc
    - accuracy
    - f1

interpretation:
  method: bilinear_attention_map
```

## 14.2 MapDiff preset

File:

```text
kale_protein/presets/mapdiff.yaml
```

Content:

```yaml
name: mapdiff
task: inverse_folding
objective: generative
runner: diffusion_generate

streams:
  structure:
    modality: protein_structure
    input_key: backbone_coords
    processor: backbone_coordinate_processor
    encoder: mapdiff_structure_encoder
    processor_kwargs: {}
    encoder_kwargs:
      hidden_dim: 128

  noisy_sequence:
    modality: protein_sequence
    input_key: sequence
    processor: masked_sequence_tokenizer
    encoder: residue_token_embedding
    processor_kwargs:
      max_length: 512
      mask_token: "<mask>"
    encoder_kwargs:
      hidden_dim: 128

conditioner:
  type: structure_conditioned_denoising
  kwargs:
    hidden_dim: 128

head:
  type: diffusion_sequence_decoder
  kwargs:
    hidden_dim: 128
    vocab_size: 25

sampling:
  steps: 100
  num_samples: 8
  temperature: 1.0

evaluation:
  metrics:
    - sequence_recovery
    - diversity
    - novelty

interpretation:
  method: denoising_trajectory
```

---

# 15. Examples

## 15.1 DrugBAN example

Directory:

```text
kale_protein/examples/drugban_dti/
```

Files:

```text
config.yaml
train.py
evaluate.py
interpret.py
README.md
```

The example should show:

```python
config = AutoProteinConfig.from_yaml("config.yaml")
loader = AutoProteinDataLoader.from_config(config)
dataset = loader.load()
preprocessor = AutoProteinPreprocessor.from_config(config)
dataset = preprocessor.transform_dataset(dataset)
predictor = AutoProteinPredictor.from_config(config)
outputs = predictor.predict(dataset)
evaluator = AutoProteinEvaluator.from_config(config)
metrics = evaluator.evaluate(outputs, dataset)
```

Use a tiny toy dataset in code or YAML so the example can run in CI.

Toy sample:

```python
[
    {
        "id": "sample_1",
        "smiles": "CCO",
        "sequence": "MKTFFVLLL",
        "label": 1,
    },
    {
        "id": "sample_2",
        "smiles": "CCN",
        "sequence": "GAVLIPFWY",
        "label": 0,
    },
]
```

## 15.2 MapDiff example

Directory:

```text
kale_protein/examples/mapdiff_inverse_folding/
```

Files:

```text
config.yaml
generate.py
evaluate.py
README.md
```

Use toy backbone coordinates.

Toy sample:

```python
[
    {
        "id": "protein_1",
        "backbone_coords": [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        "sequence": "MA",
    }
]
```

---

# 16. Tests

Implement minimum tests.

## 16.1 Registry tests

File:

```text
kale_protein/tests/test_registry.py
```

Test:

```text
1. register class by key
2. get class by key
3. available keys
4. missing key gives clear error
```

## 16.2 Config tests

File:

```text
kale_protein/tests/test_config.py
```

Test:

```text
1. load drugban preset
2. load mapdiff preset
3. validate required fields
4. parse streams into StreamSpec
```

## 16.3 Preprocessor tests

File:

```text
kale_protein/tests/test_auto_preprocessor.py
```

Test:

```text
1. DrugBAN config builds two processors
2. sample with smiles + sequence becomes drug stream + target stream
3. label is preserved
4. MapDiff config builds structure + noisy_sequence processors
```

## 16.4 Predictor tests

Files:

```text
kale_protein/tests/test_auto_predictor_drugban.py
kale_protein/tests/test_auto_predictor_mapdiff.py
```

Test:

```text
DrugBAN:
    - from_preset("drugban") builds predictor
    - predictor.predict(toy_batch) returns logits/probabilities

MapDiff:
    - from_preset("mapdiff") builds predictor
    - predictor.generate(toy_batch) returns generated sequence or token ids
```

Use small tensors and do not require real DrugBAN/MapDiff pretrained weights in v0.1.

---

# 17. Registration bootstrap

Ensure all built-in components are registered when importing `kale_protein`.

In:

```text
kale_protein/__init__.py
```

Import registration modules:

```python
from .registry import *
from .modalities.small_molecule import processors as _small_molecule_processors
from .modalities.small_molecule import encoders as _small_molecule_encoders
from .modalities.protein_sequence import processors as _protein_sequence_processors
from .modalities.protein_sequence import encoders as _protein_sequence_encoders
from .modalities.protein_structure import processors as _protein_structure_processors
from .modalities.protein_structure import encoders as _protein_structure_encoders
from .fusion import bilinear_attention as _bilinear_attention
from .fusion import concat as _concat
from .conditioners import structure_conditioned_denoising as _structure_conditioned_denoising
from .heads import classification as _classification
from .heads import diffusion_sequence_decoder as _diffusion_sequence_decoder
from .runners import predict_runner as _predict_runner
from .runners import diffusion_generate_runner as _diffusion_generate_runner
```

Alternatively, implement an explicit:

```python
import kale_protein
kale_protein.register_builtin_components()
```

The explicit registration function is cleaner and easier to debug.

---

# 18. Error handling requirements

Every registry lookup should produce a clear error.

Example:

```text
No modality encoder registered for key:
    ('protein_sequence', 'esm2')

Available encoders for modality 'protein_sequence':
    - drugban_protein_cnn
    - residue_token_embedding
```

For unsupported runner methods:

```text
PredictRunner does not support generate().
Available methods:
    - predict
    - fit
```

For missing optional dependencies:

```text
RDKitGraphProcessor requires rdkit, but rdkit is not installed.
Install rdkit or use processor: smiles_tokenizer.
```

---

# 19. Important design boundaries

Follow these boundaries strictly.

## 19.1 Stream-level modules

Stream processors and encoders should only handle one modality.

Examples:

```text
small molecule processor:
    SMILES → graph or tokens

protein sequence processor:
    sequence → residue tokens

protein structure processor:
    coordinates → coordinate tensor / residue graph
```

They should not perform cross-modal operations.

## 19.2 Fusion/conditioner modules

Any operation involving multiple streams belongs here.

Examples:

```text
DrugBAN bilinear attention:
    drug stream + target stream → fused representation

MapDiff structure-conditioned denoising:
    structure stream + noisy sequence stream → denoising hidden state
```

## 19.3 Task-level modules

Task-level logic includes:

```text
positive/negative pair construction
cold drug / cold target / cold pair split
domain labels
losses
metrics
sampling protocol
evaluation protocol
```

Do not hide task-level logic inside modality processors.

---

# 20. Compatibility with future models

Design the framework so these can be added later without changing the top-level API.

## 20.1 AlphaFold-like structure prediction

Future config should look like:

```yaml
task: structure_prediction
objective: conditional_prediction
runner: structure_predict

streams:
  sequence:
    modality: protein_sequence
    input_key: sequence
    processor: amino_acid_tokenizer
    encoder: sequence_embedding

  msa:
    modality: protein_msa
    input_key: msa
    processor: msa_processor
    encoder: msa_encoder

  template:
    modality: protein_template
    input_key: template
    processor: template_processor
    encoder: template_encoder
    optional: true

fusion:
  type: evoformer_like_trunk

head:
  type: structure_prediction_head

evaluation:
  metrics:
    - tm_score
    - rmsd
    - lddt
```

Do not implement this in v0.1. Just make sure the schema can support it later.

## 20.2 RFDiffusion-like backbone generation

Future config should look like:

```yaml
task: protein_backbone_generation
objective: generative
runner: diffusion_generate

streams:
  constraint:
    modality: protein_structure_constraint
    input_key: motif_or_target
    processor: constraint_processor
    encoder: constraint_encoder

  noisy_backbone:
    modality: protein_structure
    input_key: noisy_backbone
    processor: noisy_backbone_processor
    encoder: structure_noise_embedding

conditioner:
  type: guided_diffusion_conditioner

head:
  type: backbone_denoiser

evaluation:
  metrics:
    - designability
    - diversity
    - novelty
    - motif_rmsd
```

Do not implement this in v0.1.

---

# 21. Suggested implementation order

Follow this order.

## Phase 1 — Foundation

```text
1. Create package structure.
2. Implement Registry.
3. Implement AutoProteinConfig.
4. Add drugban.yaml and mapdiff.yaml presets.
5. Add tests for Registry and Config.
```

## Phase 2 — Stream preprocessing

```text
1. Implement StreamSpec.
2. Implement AutoProteinPreprocessor.
3. Implement small molecule processors:
   - SMILESTokenizer
   - DummyMoleculeGraphProcessor or RDKitGraphProcessor
4. Implement protein sequence processors:
   - AminoAcidTokenizer
   - MaskedSequenceTokenizer
5. Implement protein structure processor:
   - BackboneCoordinateProcessor
6. Add tests for preprocessing.
```

## Phase 3 — Minimal model assembly

```text
1. Implement modality encoder registry.
2. Implement minimal DrugBAN molecule encoder.
3. Implement minimal DrugBAN protein encoder.
4. Implement minimal MapDiff structure encoder.
5. Implement residue token embedding.
6. Implement MultiStreamProteinModel.
7. Implement AutoProteinPredictor.
```

## Phase 4 — Fusion, conditioner, heads, runners

```text
1. Implement ConcatFusion.
2. Implement BilinearAttentionFusion.
3. Implement StructureConditionedDenoising.
4. Implement BinaryClassificationHead.
5. Implement DiffusionSequenceDecoder.
6. Implement PredictRunner.
7. Implement DiffusionGenerateRunner.
8. Add predictor tests for DrugBAN and MapDiff.
```

## Phase 5 — Evaluator and interpreter

```text
1. Implement AutoProteinEvaluator.
2. Implement DTI metrics.
3. Implement inverse folding metrics.
4. Implement AutoProteinInterpreter.
5. Implement DrugBAN attention interpreter.
6. Implement MapDiff trajectory interpreter.
```

## Phase 6 — Examples and documentation

```text
1. Add DrugBAN example with toy data.
2. Add MapDiff example with toy data.
3. Add README explaining:
   - six-step PyKale workflow
   - stream-based modality registry
   - DrugBAN preset
   - MapDiff preset
   - how to add new modality
   - how to add new model preset
4. Ensure examples run without real external datasets.
```

---

# 22. Minimum acceptance criteria

The implementation is acceptable when the following work:

```python
from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor

config = AutoProteinConfig.from_preset("drugban")
preprocessor = AutoProteinPreprocessor.from_config(config)

sample = {
    "smiles": "CCO",
    "sequence": "MKTFFVLLL",
    "label": 1,
}

processed = preprocessor.transform_sample(sample)

assert "drug" in processed
assert "target" in processed
assert "label" in processed
```

And:

```python
from kale_protein.auto import AutoProteinConfig, AutoProteinPredictor

config = AutoProteinConfig.from_preset("drugban")
predictor = AutoProteinPredictor.from_config(config)
```

And:

```python
config = AutoProteinConfig.from_preset("mapdiff")
predictor = AutoProteinPredictor.from_config(config)
```

And tests pass:

```bash
pytest kale_protein/tests
```

---

# 23. Non-goals for v0.1

Do not attempt these in the first implementation unless they already exist in the repo:

```text
1. Full faithful DrugBAN reproduction with pretrained checkpoints.
2. Full faithful MapDiff diffusion implementation.
3. Full PDB/mmCIF parser.
4. AlphaFold-like model implementation.
5. RFDiffusion implementation.
6. Real TM-score/RMSD external binary integration.
7. Large dataset downloading.
8. Training scripts requiring GPU.
```

v0.1 should prioritize:

```text
stable abstractions
clean registry
toy runnable examples
clear extension path
```

---

# 24. Final Codex instruction

Use this instruction directly:

```text
Implement KaleProtein as a stream-based Auto/registry framework following the PyKale six-step workflow.

Do not implement DrugBAN and MapDiff as monolithic hard-coded models. Implement them as presets composed of modality processors, modality encoders, fusion/conditioner modules, heads, runners, evaluators, and interpreters.

Prioritize clean architecture, stable config schema, toy runnable examples, and tests over full reproduction of the original DrugBAN/MapDiff papers.

Implement the work in phases:
1. registry and config
2. stream preprocessors
3. model assembly
4. fusion/conditioner/head/runner
5. evaluator/interpreter
6. examples and tests

The first successful target is:
- AutoProteinConfig.from_preset("drugban")
- AutoProteinConfig.from_preset("mapdiff")
- AutoProteinPreprocessor.from_config(config)
- AutoProteinPredictor.from_config(config)
- toy DrugBAN prediction
- toy MapDiff generation
- pytest passing

Keep future compatibility with AlphaFold-like structure prediction and RFDiffusion-like backbone generation by designing around task, objective, runner, input_schema/output_schema, and modality streams. Do not implement those future models now.
```

This gives Codex a bounded implementation target and avoids the common failure mode: trying to port full DrugBAN/MapDiff before the framework abstraction is stable.
