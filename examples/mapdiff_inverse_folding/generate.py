"""Generate inverse-folded sequences from a PDB or processed graph file."""

import argparse
import json
from pathlib import Path

import torch

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinModel,
)
from kaleprotein.utils import move_to_device
from examples.mapdiff_inverse_folding import register_model_card


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A PDB, processed .pt graph, or directory of graphs.")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument(
        "--checkpoint",
        type=Path,
        help="A local full MapDiff checkpoint.",
    )
    weights.add_argument("--pretrained", action="store_true", help="Resolve and strictly load the configured v1.0.1 release.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    register_model_card()
    torch.manual_seed(args.seed)

    # 1. Load, preprocess, collate, and batch input structures.
    config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")
    loader = AutoProteinDataLoader(
        "CATH/InverseFolding",
        config=config,
        source=args.input,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2. Build the model and load requested weights through AutoProteinModel.
    model = AutoProteinModel(
        "InverseFolding/MapDiff",
        pretrain=args.pretrained,
        checkpoint=args.checkpoint,
    ).to(args.device)
    model.eval()

    # 3. Encode each condition batch and run the registered generator.
    generated_batches = []
    with torch.no_grad():
        for inputs in loader:
            inputs = move_to_device(inputs, args.device)
            embeddings = model.embed(**inputs)
            generated_batches.append(
                model.generate(
                    **embeddings,
                    steps=args.steps,
                    method=args.method,
                    num_samples=args.num_samples,
                    temperature=args.temperature,
                )
            )
    if not generated_batches:
        raise ValueError("Cannot generate from an empty inverse-folding dataset.")
    generation = _merge_generations(generated_batches)

    # 4. Serialize generated sequences and their denoising trajectories.
    serializable = {
        "sequences": generation["sequences"],
        "trajectory": generation["trajectory"],
        "trajectories": generation["trajectories"],
        "sample_ids": generation["sample_ids"],
        "reference_sequences": generation["reference_sequences"],
    }
    text = json.dumps(serializable, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return generation


def _merge_generations(generations):
    sequences = []
    token_ids = []
    logits = []
    trajectories = []
    references = []
    sample_ids = []
    for generation in generations:
        sequences.extend(generation["sequences"])
        references.extend(generation.get("reference_sequences", []))
        sample_ids.extend(generation.get("sample_ids", []))
        trajectories.extend(generation.get("trajectories", [generation["trajectory"]]))
        if generation.get("token_ids") is not None:
            token_ids.append(generation["token_ids"])
        if generation.get("logits") is not None:
            logits.append(generation["logits"])
    return {
        "sequences": sequences,
        "token_ids": torch.cat(token_ids) if token_ids else None,
        "logits": torch.cat(logits) if logits else None,
        "trajectory": _merge_primary_trajectories(generations),
        "trajectories": trajectories,
        "sampling_method": generations[0].get("sampling_method"),
        "reference_sequences": references,
        "sample_ids": sample_ids,
    }


def _merge_primary_trajectories(generations):
    trajectories = [generation["trajectory"] for generation in generations]
    expected_steps = len(trajectories[0])
    if any(len(trajectory) != expected_steps for trajectory in trajectories):
        raise RuntimeError("MapDiff batches produced incompatible trajectory lengths.")
    merged = []
    for index in range(expected_steps):
        states = [trajectory[index] for trajectory in trajectories]
        timesteps = {int(state["timestep"]) for state in states}
        if len(timesteps) != 1:
            raise RuntimeError("MapDiff batches produced incompatible trajectory schedules.")
        merged.append(
            {
                "timestep": timesteps.pop(),
                "sequences": [
                    sequence
                    for state in states
                    for sequence in state.get("sequences", [])
                ],
            }
        )
    return merged


if __name__ == "__main__":
    main()
