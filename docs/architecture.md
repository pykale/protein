# KaleProtein Architecture

The repository has three layers: generic Auto dispatch, reusable core
components, and complete model examples. Only the `kaleprotein/` library package
is installed; `examples/`, `tests/`, and `docs/` remain repository-level
resources.

```text
kaleprotein/
  auto/
  core/
examples/
tests/
docs/
```

```mermaid
flowchart TB
    subgraph PACKAGE["installed package: kaleprotein/"]
      subgraph CORE["core: shared and reusable"]
        direction LR
        CORE_DATA["data/<br/>utils + dataset modules + schemas"]
        CORE_PREP["preprocessing/<br/>reusable transforms"]
        CORE_MODELING["modeling/<br/>modalities + tasks"]
        CORE_EVAL["evaluation/<br/>task metrics + interpretation"]
        REGISTRY["registry/"]
        CONFIG["config/"]
        WEIGHTS["weights/"]
      end

      subgraph AUTO["auto: selection and construction"]
        direction LR
        DATA["AutoProteinData"]
        PREP["AutoProteinPreprocessor"]
        MODEL["AutoProteinModel"]
        EMBED["AutoProteinEmbedder"]
        PREDICT["AutoProteinPredictor"]
        EVAL["AutoProteinEvaluator"]
        INTERP["AutoProteinInterpreter"]
        DATA --> PREP --> MODEL --> EVAL --> INTERP
        MODEL --> EMBED
        MODEL --> PREDICT
      end
    end

    subgraph EXAMPLES["repository examples/: complete named models"]
        direction LR
        DRUGBAN["drugban_dti/<br/>config + model + scripts + assets"]
        MAPDIFF["mapdiff_inverse_folding/<br/>config + model + scripts + assets"]
        MORE["new model cards"]
    end

    AUTO -->|"generic registry lookup"| CORE
    MODEL -->|"auto_map"| EXAMPLES
    EXAMPLES -->|"compose reusable components"| CORE
```

## User API

Users normally construct one complete model:

```python
model = AutoProteinModel("DTI/DrugBAN", pretrain=True)
```

`AutoProteinModel` reads the model card and imports its `modeling.py`. It does
not contain a DrugBAN or MapDiff condition.

## Model Composition

The concrete class is the composition root. DrugBAN declares:

```text
DrugBANModel
  protein_embedder  <- AutoProteinEmbedder("sequence/cnn")
  molecule_embedder <- AutoProteinEmbedder("molecule/gcn")
  predictor         <- AutoProteinPredictor("dti/ban")
```

MapDiff declares:

```text
MapDiffModel
  embedder  <- AutoProteinEmbedder("structure/mapdiff_condition")
  predictor <- AutoProteinPredictor("inverse_folding/mapdiff_generator")
```

For a discriminative model, the predictor is the task fusion/head. For a
generative model, the predictor is the generator or denoiser. Component Auto
classes are model-author APIs; they do not resolve full model cards.

## Named Stage Contracts

Public pipeline boundaries exchange ordinary dictionaries. Each next stage
expands that dictionary into named arguments:

```python
processed = preprocessor.transform_dataset(records)
batch = model.collator(**processed)
embeddings = model.embed(**batch)
prediction = model.predictor(**embeddings)
metrics = model.evaluate(**prediction)
```

Auto requires mappings at these boundaries, but it does not prescribe one
universal tensor schema. The concrete model card owns meaningful names. For
example, DrugBAN uses `protein_embedding` and `molecule_embedding`, while
MapDiff uses `structure_embedding`, `conditioning`, and `batch`. Predictors and
generators declare the fields they consume and accept unrelated metadata with
`**kwargs` when it should flow to a later stage. This lets users replace or
insert components using normal Python APIs instead of adapting positional
tuples or framework-specific workflow containers.

## Runtime Pipelines

DrugBAN:

```mermaid
flowchart LR
    CSV["BindingDB / Human / BioSNAP"] --> PREP["SMILES graph + protein sequence"]
    PREP --> COLLATE["DrugBAN collator"]
    COLLATE --> EMBED["molecule GCN + sequence CNN"]
    EMBED --> PREDICT["BAN predictor"]
    PREDICT --> METRICS["metrics"]
    PREDICT --> ATTENTION["optional attention interpretation"]
```

MapDiff:

```mermaid
flowchart LR
    INPUT["CATH graph / PDB"] --> PREP["backbone preprocessing"]
    PREP --> COLLATE["sparse graph + padded IPA batch"]
    COLLATE --> CONDITION["structure condition encoding"]
    CONDITION --> GENERATE["iterative diffusion generator"]
    GENERATE --> METRICS["recovery / perplexity / diversity"]
    GENERATE --> TRAJECTORY["optional trajectory interpretation"]
```

## Ownership Rules

- `auto/` owns generic dispatch only.
- `core/data/utils/` owns fundamental FASTA, tabular, PDB, mmCIF, and file
  parsing functions.
- `core/data/<dataset>.py` owns built-in dataset adapters such as BindingDB,
  BioSNAP, Human, and CATH. Public ids use `Dataset/Task` order.
- `core/data/base.py` and `core/data/schemas.py` own shared dataset classes and
  stable data records.
- `core/preprocessing/` owns reusable transformations that prepare records for
  model inputs without becoming dataset loaders.
- `core/modeling/modalities/` owns reusable modality encoders and neural layers.
- `core/modeling/tasks/` owns reusable fusion layers, heads, predictors, and
  generators.
- `core/evaluation/tasks/` owns task metrics and interpretation methods.
- `examples/<model>/` owns concrete model composition, model-specific layers,
  collators, feature graphs, forward/generate behavior, scripts, and checkpoint
  adapters.
- The complete model is the sole owner of full-model weight resolution and
  loading. Nested components never download the same checkpoint again.

Adding a new model normally adds one example directory and model card. Core is
changed only when the model introduces a genuinely reusable component; Auto is
not changed.

In a source checkout, `kaleprotein` discovers cards from the adjacent
`examples/` directory. Installed wheels do not include those examples; users or
downstream packages register external model cards through
`discover_model_cards(...)` or `register_model_card(...)`.
