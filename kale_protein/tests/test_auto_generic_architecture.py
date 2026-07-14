from pathlib import Path

from kale_protein.auto import AutoProteinConfig, AutoProteinModel, AutoProteinPreprocessor
from kale_protein.registry import (
    MODEL_CARD_REGISTRY,
    MODALITY_PROCESSOR_REGISTRY,
    PRESET_REGISTRY,
)
from kale_protein.registry.base import Registry
from kale_protein.registry.model_cards import discover_model_cards


def _fake_card_text(model_id="Architecture/Fake", name="architecture-fake"):
    return f"""\
model_id: {model_id}
name: {name}
task: fake_task
objective: generative
runner: fake_runner
streams:
  input:
    modality: fake_modality
    input_key: payload
    processor: fake_processor
    encoder: fake_encoder
head:
  type: fake_head
"""


def test_model_card_discovery_uses_config_metadata(tmp_path):
    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(_fake_card_text(), encoding="utf-8")
    cards = Registry("test model cards")
    presets = Registry("test presets")

    discovered = discover_model_cards(
        tmp_path, model_card_registry=cards, preset_registry=presets
    )

    assert discovered == ["Architecture/Fake"]
    assert cards.get("Architecture/Fake") == config_path.resolve()
    assert presets.get("architecture-fake") == config_path.resolve()


def test_from_preset_resolves_registered_card(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_fake_card_text(), encoding="utf-8")
    monkeypatch.setitem(PRESET_REGISTRY._mapping, "architecture-fake", config_path)

    config = AutoProteinConfig.from_preset("architecture-fake")

    assert config["model_id"] == "Architecture/Fake"
    assert config.get_streams()["input"].processor == "fake_processor"


def test_auto_config_has_no_model_specific_preset_tables():
    auto_dir = Path(__file__).resolve().parents[1] / "auto"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in auto_dir.glob("*.py")
    ).casefold()

    assert "drugban" not in source
    assert "mapdiff" not in source
    assert "multistreamproteinmodel" not in source


def test_named_architectures_are_not_defined_in_shared_model_modules():
    package = Path(__file__).resolve().parents[1]
    shared_roots = [
        package / "auto",
        package / "fusion",
        package / "heads",
        package / "conditioners",
    ]
    shared_files = [path for root in shared_roots for path in root.glob("*.py")]
    shared_files.extend((package / "modalities").glob("*/encoders.py"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in shared_files)

    assert "class DrugBAN" not in source
    assert "class MapDiff" not in source
    assert "drugban_molecule_gnn" not in source
    assert "mapdiff_structure_encoder" not in source


def test_registered_alias_and_canonical_processor_resolution():
    class FakeProcessor:
        def __init__(self, input_key="default", marker=None):
            self.input_key = input_key
            self.marker = marker

        def transform(self, sample):
            return {"value": sample[self.input_key], "marker": self.marker}

    key = ("architecture_modality", "fake_processor")
    MODALITY_PROCESSOR_REGISTRY.register(
        key,
        FakeProcessor,
        aliases=(("architecture/input", "payload"),),
    )

    aliased = AutoProteinPreprocessor("architecture/input", marker="alias")
    canonical = AutoProteinPreprocessor(
        "architecture_modality/fake_processor",
        input_key="payload",
        marker="canonical",
    )

    assert aliased.featurize({"payload": 3}) == {"value": 3, "marker": "alias"}
    assert canonical.featurize({"payload": 4}) == {
        "value": 4,
        "marker": "canonical",
    }


def test_builtin_cards_are_bootstrapped_without_example_imports():
    import kale_protein
    from kale_protein.registry import MODEL_CARD_REGISTRY

    source = Path(kale_protein.__file__).read_text(encoding="utf-8")
    assert ".examples" not in source
    assert MODEL_CARD_REGISTRY.available_keys()


def test_auto_map_card_modules_support_relative_imports(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
model_id: Architecture/Relative
task: fake_task
objective: generative
runner: generate
auto_map:
  AutoProteinConfig: configuration.RelativeConfig
  AutoProteinModel: modeling.RelativeModel
streams:
  input:
    modality: fake
    input_key: value
    processor: fake
    encoder: fake
head:
  type: fake
""",
        encoding="utf-8",
    )
    (tmp_path / "configuration.py").write_text(
        "from kale_protein.auto import AutoProteinConfig\n"
        "class RelativeConfig(AutoProteinConfig):\n"
        "    marker = 'relative'\n",
        encoding="utf-8",
    )
    (tmp_path / "modeling.py").write_text(
        "from .configuration import RelativeConfig\n"
        "class RelativeModel:\n"
        "    def __init__(self, config, pretrain=False):\n"
        "        self.config_class = RelativeConfig\n"
        "        self.pretrain = pretrain\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        MODEL_CARD_REGISTRY._mapping, "Architecture/Relative", config_path
    )

    model = AutoProteinModel("Architecture/Relative", pretrain=True)

    assert model.config_class.marker == "relative"
    assert model.pretrain is True
