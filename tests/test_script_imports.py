from __future__ import annotations

import importlib


def test_example_scripts_import_without_heavy_dependencies() -> None:
    modules = [
        "examples.drugban.train",
        "examples.drugban.eval",
        "examples.drugban.predict",
        "examples.drugban.interpret",
        "examples.mapdiff.pretrain_ipa",
        "examples.mapdiff.train_diffusion",
        "examples.mapdiff.eval",
        "examples.mapdiff.predict",
    ]

    for module_name in modules:
        importlib.import_module(module_name)
