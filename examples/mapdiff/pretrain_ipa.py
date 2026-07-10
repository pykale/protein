from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaleprotein import AutoProteinData, AutoProteinGenerator, AutoProteinModel, AutoProteinPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MapDiff-style pretraining stage.")
    parser.add_argument("--config", default="examples/mapdiff/config.yaml")
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    example_dir = config_path.parent

    print("1. load data: processed CATH .pt graph files")
    data_module = AutoProteinData(config["data_id"], data_dir=example_dir / "data/processed")
    data, _ = data_module.load("train")

    print("2. preprocess: featurize residue structures")
    structure_preprocessor = AutoProteinPreprocessor("protein/structure")
    structure_data = structure_preprocessor.featurize(data)
    print({name: tuple(value.shape) for name, value in structure_data.items()})

    print("3. embed/model: AutoProteinModel('InverseFolding/MapDiff')")
    structure_encoder = AutoProteinModel(config["model_id"], pretrain=args.pretrain)
    sequence_generator = AutoProteinGenerator(config["model_id"], pretrain=args.pretrain)
    optimizer = torch.optim.Adam(
        list(structure_encoder.parameters()) + list(sequence_generator.parameters()),
        lr=float(config["train"]["lr"]),
    )

    print("4. train: local optimization loop")
    structure_encoder.train()
    sequence_generator.train()
    last_loss = 0.0
    for _ in range(int(config["train"]["epochs"])):
        optimizer.zero_grad()
        structure_embedding = structure_encoder.embed(structure_data)
        generation = sequence_generator.generate(structure_embedding)
        loss = F.cross_entropy(generation["logits"].transpose(1, 2), structure_data["labels"], ignore_index=20)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    print({"train_loss": last_loss})


if __name__ == "__main__":
    main()
