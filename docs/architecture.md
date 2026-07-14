# KaleProtein Architecture

KaleProtein uses a model-card-driven Auto layer. Shared Auto classes discover
implementations from `config.yaml` and `auto_map`; architecture-specific code
stays inside each model package.

```mermaid
flowchart TB
    ENTRY["User code or runnable example CLI"]

    subgraph PIPELINE["Explicit runtime pipeline"]
        direction LR
        DATA["1. Load data<br/>AutoProteinData"]
        PREPROCESS["2. Preprocess and collate<br/>AutoProteinPreprocessor<br/>AutoMoleculePreprocessor"]
        MODEL["3. Build and embed<br/>AutoProteinModel<br/>model.embed()"]
        INFERENCE["4. Predict or generate<br/>AutoProteinPredictor<br/>AutoProteinGenerator"]
        OUTPUT["5. Evaluate or interpret<br/>AutoProteinEvaluator<br/>AutoProteinInterpreter"]

        DATA --> PREPROCESS --> MODEL --> INFERENCE --> OUTPUT
    end

    ENTRY --> DATA

    subgraph GENERIC["Generic infrastructure: no model-specific classes"]
        direction LR
        CONFIG["auto/config.py<br/>AutoProteinConfig"]
        CARDS["registry/model_cards.py<br/>filesystem discovery<br/>auto_map import"]
        REGISTRIES["registry/*<br/>datasets, processors<br/>metrics, interpreters"]
        WEIGHTS["auto/weights.py<br/>local first, download<br/>checksum, checkpoint load"]

        CONFIG -->|"reads auto_map"| CARDS
    end

    DATA -.->|"dataset ID"| REGISTRIES
    PREPROCESS -.->|"modality ID"| REGISTRIES
    OUTPUT -.->|"task and method"| REGISTRIES
    MODEL -.->|"model ID"| CONFIG
    INFERENCE -.->|"model ID"| CONFIG
    MODEL -.->|"pretrain=True"| WEIGHTS
    INFERENCE -.->|"pretrain=True"| WEIGHTS

    subgraph SHARED["Reusable shared layer"]
        direction LR
        MODALITIES["modalities/<br/>protein_sequence<br/>protein_structure<br/>small_molecule"]
        TASKS["tasks/<br/>drug_target_interaction<br/>inverse_folding"]
        DTI["Reusable DTI datasets<br/>BindingDB, Human, BioSNAP"]
        INVERSE_DATA["Inverse-folding data<br/>CATH .pt, PDB preprocessing<br/>CollatorIPAPretrain, CollatorDiff"]

        TASKS --> DTI
        TASKS --> INVERSE_DATA
    end

    REGISTRIES --> MODALITIES
    REGISTRIES --> TASKS

    subgraph MODEL_PACKAGES["Self-contained model-card packages"]
        direction LR

        subgraph DRUGBAN["examples/drugban_dti/"]
            DB_CONFIG["config.yaml<br/>configuration.py"]
            DB_MODEL["modeling.py<br/>molecular GCN<br/>protein CNN<br/>BAN and MLP"]
            DB_SCRIPTS["train.py, evaluate.py<br/>predict.py, interpret.py"]
            DB_ASSETS["data/<br/>weights/"]

            DB_CONFIG --> DB_MODEL
            DB_ASSETS --> DB_MODEL
        end

        subgraph MAPDIFF["examples/mapdiff_inverse_folding/"]
            MD_CONFIG["config.yaml<br/>configuration.py"]
            MD_MODEL["modeling.py<br/>egnn.py, ipa.py<br/>diffusion.py"]
            MD_COMPAT["upstream_compat.py<br/>MapDiff v1 checkpoint layout"]
            MD_SCRIPTS["pretrain_ipa.py<br/>train_diffusion.py<br/>evaluate.py, generate.py"]
            MD_ASSETS["data/<br/>weights/<br/>maps/"]

            MD_CONFIG --> MD_MODEL
            MD_COMPAT --> MD_MODEL
            MD_ASSETS --> MD_MODEL
        end
    end

    ENTRY --> DB_SCRIPTS
    ENTRY --> MD_SCRIPTS
    DB_SCRIPTS -.->|"uses Auto APIs"| DATA
    MD_SCRIPTS -.->|"uses Auto APIs"| DATA

    CARDS -->|"DTI/DrugBAN"| DB_CONFIG
    CARDS -->|"InverseFolding/MapDiff"| MD_CONFIG
    WEIGHTS -->|"DrugBAN checkpoint"| DB_MODEL
    WEIGHTS -->|"MapDiff checkpoint"| MD_MODEL

    QUALITY["tests/ and GitHub Actions<br/>fake data, fake URLs, mocked dependencies<br/>pytest, compileall, package build"]
    QUALITY -.->|"validates Auto dispatch"| REGISTRIES
    QUALITY -.->|"validates pipeline wiring"| DATA
    QUALITY -.->|"validates implementations"| DB_MODEL
    QUALITY -.->|"validates implementations"| MD_MODEL
```

## Ownership Boundaries

- `auto/` resolves public identifiers and checkpoints but does not define
  DrugBAN, MapDiff, or another concrete architecture.
- `registry/`, `modalities/`, and `tasks/` contain reusable discovery,
  preprocessing, dataset, metric, and interpretation components.
- `examples/<model>/` owns the model card, configuration class, model classes,
  workflow scripts, maps, and weight location for that architecture.
- Every workflow exposes the same stages while retaining model-specific train,
  evaluation, prediction, interpretation, or generation behavior.
