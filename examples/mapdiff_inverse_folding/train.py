"""Train the full MapDiff IPA prior or categorical diffusion model."""

from __future__ import annotations

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
from examples.mapdiff_inverse_folding.collators import MapDiffCollator


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument(
        "--stage",
        choices=("ipa", "diffusion"),
        required=True,
        help="Pretrain the IPA prior or train the complete diffusion model.",
    )
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--scheduler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the release OneCycleLR schedule.",
    )
    parser.add_argument("--validation-every", type=int)
    parser.add_argument("--validation-steps", type=int)
    parser.add_argument(
        "--validation-method",
        choices=("ddim", "ddpm"),
    )
    parser.add_argument("--validation-num-samples", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("mapdiff.pt")
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    register_model_card()
    torch.manual_seed(args.seed)

    # 1. Load, preprocess, and batch CATH or coordinate structures.
    config = AutoProteinConfig.from_pretrained(
        "InverseFolding/MapDiff"
    )
    settings = _training_settings(config, args)
    args.batch_size = settings["batch_size"]
    train_loader = _make_loader(
        args.data,
        config,
        args,
        shuffle=True,
    )
    validation_loader = (
        _make_loader(
            args.validation_data,
            config,
            args,
            shuffle=False,
        )
        if args.validation_data
        else None
    )

    # 2. Build the complete model and restore an optional stage checkpoint.
    model = AutoProteinModel(
        "InverseFolding/MapDiff",
        checkpoint=args.checkpoint,
    ).to(args.device)
    parameters = (
        list(model.network.prior_model.parameters())
        if args.stage == "ipa"
        else list(model.parameters())
    )
    optimizer = torch.optim.Adam(
        parameters,
        lr=settings["learning_rate"],
        betas=settings["betas"],
        weight_decay=settings["weight_decay"],
    )
    total_steps = settings["epochs"] * len(train_loader)
    if total_steps < 1:
        raise ValueError("Cannot train MapDiff on an empty dataset.")
    scheduler = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=settings["learning_rate"],
            total_steps=total_steps,
        )
        if settings["scheduler"] and total_steps > 1
        else None
    )

    # 3. Embed structural conditions and optimize the selected full objective.
    best_validation_score = None
    for epoch in range(1, settings["epochs"] + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device=args.device,
            optimizer=optimizer,
            scheduler=scheduler,
            parameters=parameters,
            gradient_clip=args.gradient_clip,
        )
        report = {"epoch": epoch, "train_loss": train_loss}
        should_validate = validation_loader is not None and (
            epoch % settings["validation_every"] == 0
            or epoch == settings["epochs"]
        )
        if should_validate:
            if args.stage == "ipa":
                validation_loss = _run_epoch(
                    model,
                    validation_loader,
                    device=args.device,
                )
                report["validation_loss"] = validation_loss
                validation_score = (-validation_loss,)
            else:
                validation_metrics = _evaluate_generation(
                    model,
                    validation_loader,
                    device=args.device,
                    steps=settings["validation_steps"],
                    method=settings["validation_method"],
                    num_samples=settings["validation_num_samples"],
                )
                report["validation"] = validation_metrics
                validation_score = (
                    validation_metrics["sequence_recovery"],
                    -validation_metrics["perplexity"],
                )
            if (
                best_validation_score is None
                or validation_score > best_validation_score
            ):
                best_validation_score = validation_score
                model.save_checkpoint(args.output)
        print(json.dumps(report))

    # 4. Save a complete checkpoint loadable by AutoProteinModel.
    if validation_loader is None:
        model.save_checkpoint(args.output)
    return args.output


def _make_loader(source, config, args, *, shuffle):
    return AutoProteinDataLoader(
        "CATH/InverseFolding",
        config=config,
        source=source,
        collator=MapDiffCollator(config=config, stage=args.stage),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def _run_epoch(
    model,
    loader,
    *,
    device,
    optimizer=None,
    scheduler=None,
    parameters=None,
    gradient_clip=1.0,
):
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs in loader:
            inputs = move_to_device(inputs, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embeddings = model.embed(**inputs)
            prediction = model.predict(**embeddings)
            loss = prediction["loss"]
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters or model.parameters(), gradient_clip
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            losses.append(loss.detach())
    if not losses:
        raise ValueError("Cannot train or validate on an empty dataset.")
    return torch.stack(losses).mean().item()


def _evaluate_generation(
    model,
    loader,
    *,
    device,
    steps,
    method,
    num_samples,
):
    model.eval()
    sequences = []
    references = []
    sample_ids = []
    logits = []
    with torch.no_grad():
        for inputs in loader:
            inputs = move_to_device(inputs, device)
            embeddings = model.embed(**inputs)
            generation = model.generate(
                **embeddings,
                steps=steps,
                method=method,
                num_samples=num_samples,
            )
            sequences.extend(generation["sequences"])
            references.extend(generation["reference_sequences"])
            sample_ids.extend(generation["sample_ids"])
            logits.append(generation["logits"].detach().cpu())
    if not sequences:
        raise ValueError("Cannot validate MapDiff on an empty dataset.")
    return model.evaluate(
        sequences=sequences,
        reference_sequences=references,
        sample_ids=sample_ids,
        logits=torch.cat(logits),
    )


def _training_settings(config, args):
    training = config.get("training", {})
    stage = dict(training.get(args.stage, {}))
    validation = dict(training.get("validation", {}))
    values = {
        "epochs": args.epochs
        if args.epochs is not None
        else int(stage.get("epochs", 200 if args.stage == "ipa" else 100)),
        "batch_size": args.batch_size
        if args.batch_size is not None
        else int(stage.get("batch_size", 8)),
        "learning_rate": args.learning_rate
        if args.learning_rate is not None
        else float(stage.get("learning_rate", 5e-4)),
        "weight_decay": args.weight_decay
        if args.weight_decay is not None
        else float(stage.get("weight_decay", 0.0)),
        "scheduler": args.scheduler
        if args.scheduler is not None
        else bool(stage.get("scheduler", True)),
        "betas": (
            float(stage.get("beta1", 0.95)),
            float(stage.get("beta2", 0.999)),
        ),
        "validation_every": args.validation_every
        if args.validation_every is not None
        else int(stage.get("validation_every", 1)),
        "validation_steps": args.validation_steps
        if args.validation_steps is not None
        else int(validation.get("steps", 50)),
        "validation_method": args.validation_method
        or validation.get("method", "ddim"),
        "validation_num_samples": args.validation_num_samples
        if args.validation_num_samples is not None
        else int(validation.get("num_samples", 1)),
    }
    positive = (
        "epochs",
        "batch_size",
        "learning_rate",
        "validation_every",
        "validation_steps",
        "validation_num_samples",
    )
    invalid = [name for name in positive if values[name] <= 0]
    if invalid:
        raise ValueError(
            "MapDiff training settings must be positive: "
            f"{invalid}."
        )
    if values["weight_decay"] < 0:
        raise ValueError("MapDiff weight_decay must be non-negative.")
    return values


if __name__ == "__main__":
    main()
