"""Auto classes for KaleProtein models and datasets."""

from __future__ import annotations

from typing import Any

from kaleprotein.drugban.data import DrugBANDataModule
from kaleprotein.drugban.modeling import DrugBANForDTI, DrugBANInteractionPredictor
from kaleprotein.drugban.tokenization import DrugBANMoleculePreprocessor, DrugBANProteinPreprocessor
from kaleprotein.hub import load_model_config
from kaleprotein.mapdiff.data import MapDiffDataModule
from kaleprotein.mapdiff.modeling import MapDiffForInverseFolding, MapDiffGenerator, MapDiffStructurePreprocessor


MODEL_REGISTRY = {
    "drugban": DrugBANForDTI,
    "mapdiff": MapDiffForInverseFolding,
}

DATA_REGISTRY = {
    "drugban": DrugBANDataModule,
    "mapdiff": MapDiffDataModule,
}

PROTEIN_PREPROCESSOR_REGISTRY = {
    "protein/sequence": DrugBANProteinPreprocessor,
    "protein/structure": MapDiffStructurePreprocessor,
}

MOLECULE_PREPROCESSOR_REGISTRY = {
    "molecule/smiles": DrugBANMoleculePreprocessor,
    "molecule/smile": DrugBANMoleculePreprocessor,
}

PREDICTOR_REGISTRY = {
    "drugban": DrugBANInteractionPredictor,
}

GENERATOR_REGISTRY = {
    "mapdiff": MapDiffGenerator,
}


class AutoProteinModel:
    """Instantiate a protein model by model id or local model folder."""

    def __new__(cls, model_id: str, *args: Any, **kwargs: Any):
        return cls.from_pretrained(model_id, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id: str, *args: Any, **kwargs: Any):
        config = load_model_config(model_id)
        model_type = str(config.get("model_type", "")).lower()
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_type '{model_type}' for model id '{model_id}'")
        return MODEL_REGISTRY[model_type].from_config(config, *args, **kwargs)


class AutoProteinData:
    """Instantiate a data module by dataset/model id or local data folder."""

    def __new__(cls, data_id: str, *args: Any, **kwargs: Any):
        return cls.from_pretrained(data_id, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, data_id: str, *args: Any, **kwargs: Any):
        config = load_model_config(data_id)
        model_type = str(config.get("model_type", "")).lower()
        if model_type not in DATA_REGISTRY:
            raise ValueError(f"Unsupported data model_type '{model_type}' for data id '{data_id}'")
        return DATA_REGISTRY[model_type].from_config(config, *args, **kwargs)


class AutoProteinPreprocessor:
    """Instantiate a protein preprocessor by modality id."""

    def __new__(cls, preprocessor_id: str, *args: Any, **kwargs: Any):
        key = preprocessor_id.lower()
        if key not in PROTEIN_PREPROCESSOR_REGISTRY:
            raise ValueError(f"Unsupported protein preprocessor '{preprocessor_id}'")
        return PROTEIN_PREPROCESSOR_REGISTRY[key](*args, **kwargs)


class AutoMoleculePreprocessor:
    """Instantiate a molecule preprocessor by modality id."""

    def __new__(cls, preprocessor_id: str, *args: Any, **kwargs: Any):
        key = preprocessor_id.lower()
        if key not in MOLECULE_PREPROCESSOR_REGISTRY:
            raise ValueError(f"Unsupported molecule preprocessor '{preprocessor_id}'")
        return MOLECULE_PREPROCESSOR_REGISTRY[key](*args, **kwargs)


class AutoProteinPredictor:
    """Instantiate a task predictor by model id."""

    def __new__(cls, model_id: str, *args: Any, **kwargs: Any):
        return cls.from_pretrained(model_id, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id: str, *args: Any, **kwargs: Any):
        config = load_model_config(model_id)
        model_type = str(config.get("model_type", "")).lower()
        if model_type not in PREDICTOR_REGISTRY:
            raise ValueError(f"Unsupported predictor model_type '{model_type}' for model id '{model_id}'")
        return PREDICTOR_REGISTRY[model_type].from_config(config, *args, **kwargs)


class AutoProteinGenerator:
    """Instantiate a generative head by model id."""

    def __new__(cls, model_id: str, *args: Any, **kwargs: Any):
        return cls.from_pretrained(model_id, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id: str, *args: Any, **kwargs: Any):
        config = load_model_config(model_id)
        model_type = str(config.get("model_type", "")).lower()
        if model_type not in GENERATOR_REGISTRY:
            raise ValueError(f"Unsupported generator model_type '{model_type}' for model id '{model_id}'")
        return GENERATOR_REGISTRY[model_type].from_config(config, *args, **kwargs)
