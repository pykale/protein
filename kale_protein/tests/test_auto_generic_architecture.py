from pathlib import Path

from kale_protein.auto import (
    AutoProteinConfig,
    AutoProteinEmbedder,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
)
from kale_protein.core.registry import MODEL_CARD_REGISTRY, PREPROCESSOR_REGISTRY
from kale_protein.core.registry.base import Registry
from kale_protein.core.registry.model_cards import discover_model_cards


def _fake_card_text(model_id="Architecture/Fake", name="architecture-fake"):
    return f"""\
model_id: {model_id}
name: {name}
task: fake_task
objective: generative
auto_map:
  AutoProteinModel: modeling.RelativeModel
streams:
  input:
    modality: architecture
    input_key: payload
    processor: fake
components:
  embedders:
    input:
      id: architecture/relative_embedder
  predictor:
    id: architecture/relative_predictor
"""


def test_model_card_discovery_registers_ids_and_aliases(tmp_path):
    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(_fake_card_text(), encoding="utf-8")
    cards = Registry("test model cards", normalize_strings=True)
    presets = Registry("test presets", normalize_strings=True)

    discovered = discover_model_cards(
        tmp_path, model_card_registry=cards, preset_registry=presets
    )

    assert discovered == ["Architecture/Fake"]
    assert cards.get("architecture/fake") == config_path.resolve()
    assert cards.get("architecture-fake") == config_path.resolve()
    assert presets.get("architecture-fake") == config_path.resolve()


def test_builtin_alias_resolves_through_model_card_registry():
    config = AutoProteinConfig.from_preset("drugban")
    assert config["model_id"] == "DTI/DrugBAN"


def test_auto_source_has_no_model_specific_tables_or_names():
    auto_dir = Path(__file__).resolve().parents[1] / "auto"
    source = "\n".join(path.read_text(encoding="utf-8") for path in auto_dir.glob("*.py")).casefold()

    assert "drugban" not in source
    assert "mapdiff" not in source
    assert "dti/drugban" not in source


def test_concrete_model_classes_live_only_in_examples():
    package = Path(__file__).resolve().parents[1]
    shared_files = list((package / "auto").rglob("*.py")) + list((package / "core").rglob("*.py"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in shared_files)

    assert "DrugBAN" not in source
    assert "MapDiff" not in source


def test_registered_preprocessor_alias_and_canonical_id():
    class FakeProcessor:
        default_input_key = "payload"

        def __init__(self, input_key="payload", marker=None):
            self.input_key = input_key
            self.marker = marker

        def transform(self, sample):
            return {"value": sample[self.input_key], "marker": self.marker}

    PREPROCESSOR_REGISTRY.register(
        "architecture/fake", FakeProcessor, aliases=("architecture/input",)
    )
    aliased = AutoProteinPreprocessor("architecture/input", marker="alias")
    canonical = AutoProteinPreprocessor("architecture/fake", marker="canonical")

    assert aliased.featurize({"payload": 3}) == {"value": 3, "marker": "alias"}
    assert canonical.featurize({"payload": 4}) == {"value": 4, "marker": "canonical"}


def test_builtin_cards_are_discovered_without_importing_model_modules():
    import kale_protein

    source = Path(kale_protein.__file__).read_text(encoding="utf-8")
    assert ".examples" not in source
    assert MODEL_CARD_REGISTRY.has("DTI/DrugBAN")
    assert MODEL_CARD_REGISTRY.has("InverseFolding/MapDiff")


def test_new_card_composes_registered_components_without_auto_changes(tmp_path):
    class RelativeEmbedder:
        def __init__(self, config=None, marker="embedded"):
            self.marker = marker

        def embed(self, value):
            return f"{self.marker}:{value}"

    class RelativePredictor:
        def __init__(self, config=None, suffix="predicted"):
            self.suffix = suffix

        def __call__(self, value):
            return f"{value}:{self.suffix}"

    AutoProteinEmbedder.register("architecture/relative_embedder", RelativeEmbedder)
    AutoProteinPredictor.register("architecture/relative_predictor", RelativePredictor)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(_fake_card_text("Architecture/Relative", "relative"), encoding="utf-8")
    (tmp_path / "modeling.py").write_text(
        "from kale_protein.auto import AutoProteinEmbedder, AutoProteinPredictor\n"
        "class RelativeModel:\n"
        "    def __init__(self, config, pretrain=False):\n"
        "        self.embedder = AutoProteinEmbedder.from_config(config.get_embedders()['input'], config=config)\n"
        "        self.predictor = AutoProteinPredictor.from_config(config.get_predictor(), config=config)\n"
        "        self.pretrain = pretrain\n"
        "    def __call__(self, value):\n"
        "        return self.predictor(self.embedder.embed(value))\n",
        encoding="utf-8",
    )
    MODEL_CARD_REGISTRY.register("Architecture/Relative", config_path)

    model = AutoProteinModel("Architecture/Relative", pretrain=True)

    assert model("payload") == "embedded:payload:predicted"
    assert model.pretrain is True
