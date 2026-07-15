# KaleProtein Architecture Plan

## Objective

Provide one Hugging Face-style complete-model entry point while preserving an
explicit scientific pipeline:

```python
model = AutoProteinModel("DTI/DrugBAN", pretrain=True)
```

```text
load -> preprocess -> collate -> embed -> predict/generate -> evaluate -> interpret
```

## Boundaries

- `auto/` performs generic registration, selection, and construction. It knows
  no concrete model names.
- `core/` contains reusable registry, config, weight, modality, and task code.
- `examples/<model>/` contains concrete full-model assembly, model-specific
  layers, scripts, assets, and checkpoint adapters.
- `AutoProteinModel` returns a complete model. Inside that class,
  `AutoProteinEmbedder` resolves modality or condition encoders and
  `AutoProteinPredictor` resolves a discriminative head or generative model.
- The complete model resolves and loads a full checkpoint exactly once.

## Acceptance Criteria

- Adding a model card does not require changing Auto code.
- DrugBAN composes molecule GCN, sequence CNN, and BAN predictor components.
- MapDiff composes a condition embedder and diffusion generator without
  duplicating parameters.
- BindingDB, Human, and BioSNAP use one reusable DTI loader.
- DrugBAN training, evaluation, prediction, and interpretation are distinct
  complete workflows.
- MapDiff IPA pretraining, diffusion training, evaluation, and generation are
  distinct complete workflows.
- Missing DrugBAN pretrained weights fail clearly; MapDiff release weights use
  local-first atomic download and strict compatibility loading.
- Pytest uses fake data, URLs, chemistry objects, and tiny models only.

## Verification

```bash
python -m pytest -q
python -m compileall -q kale_protein
python -m build
```
