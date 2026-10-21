"""Classifier training and out-of-fold prediction without touching the final test set.

CLI: python -m defects.train {splits,fit,oof} --config CONFIG
The sample-index JSON is a list of {id, image_path, labels}; paths are relative to
that index. The splits command requires --groups, a JSON id -> group mapping.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.config import get_config
from defects.dataset import (
    DefectDataset,
    Sample,
    build_dataset_manifest,
    file_sha256,
)
from defects.labels import CODES
from defects.model import build_model
from defects.provenance import _git_dirty, _git_sha, _run_directory, _verify, manifest_path
from defects.splits import make_splits, read_splits, write_splits
from eval.models import macro_f1, per_class_prf

# manifest_path and _git_dirty live in defects.provenance but are used through this module:
# tests import and patch them here, and strict mypy needs the explicit export.
__all__ = [
    "FitResult",
    "_git_dirty",
    "fit",
    "load_checkpoint",
    "main",
    "manifest_path",
    "out_of_fold",
    "predict",
    "select_device",
    "setup_determinism",
]


def setup_determinism(seed: int) -> None:
    """Seed python, numpy and torch and force deterministic kernels so a run can be repeated."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device() -> str:
    """Best available device: cuda, then mps, then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _loader(samples: Sequence[Sample], cfg: dict[str, Any], *, training: bool) -> Any:
    """DataLoader over the samples; training shuffles and drops a short last batch."""
    from torch.utils.data import DataLoader

    return DataLoader(
        DefectDataset(samples, cfg, training=training),
        batch_size=cfg["batch_size"],
        shuffle=training,
        drop_last=training and len(samples) > cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )


def predict(
    model: Any, samples: Sequence[Sample], cfg: dict[str, Any], *, device: str | None = None
) -> list[list[float]]:
    """Sigmoid probabilities per sample, one column per class code."""
    import torch

    target = device or select_device()
    model.to(target).eval()
    result: list[list[float]] = []
    with torch.no_grad():
        for images, _ in _loader(samples, cfg, training=False):
            result.extend(model(images.to(target)).sigmoid().cpu().tolist())
    return result


@dataclass(frozen=True)
class FitResult:
    """Where the best checkpoint was written, the per-epoch metrics and the best epoch."""

    checkpoint: Path
    history: list[dict[str, float]]
    best_epoch: int


def fit(
    samples: Sequence[Sample],
    split_file: Path,
    fold: int,
    cfg: dict[str, Any],
    *,
    run_id: str,
    device: str | None = None,
) -> FitResult:
    """Verify a standalone run before training its persisted fold."""
    identity = _verify(samples, split_file, cfg)
    return _fit(samples, split_file, fold, cfg, run_id=run_id, device=device, identity=identity)


def _fit(
    samples: Sequence[Sample],
    split_file: Path,
    fold: int,
    cfg: dict[str, Any],
    *,
    run_id: str,
    device: str | None,
    identity: str,
) -> FitResult:
    """Train with the dataset identity already verified by this run's caller."""
    import torch

    splits = read_splits(split_file)
    splits.validate([s.sample_id for s in samples])
    if not 0 <= fold < len(splits.folds):
        raise ValueError("Invalid fold index")
    if cfg["epochs"] < 1 or cfg["patience"] < 1 or cfg["min_delta"] < 0:
        raise ValueError("epochs/patience must be positive; min_delta must be nonnegative")
    seed = int(get_config()["seed"])
    setup_determinism(seed)
    target = device or select_device()
    by_id = {s.sample_id: s for s in samples}
    train_ids, val_ids, _ = splits.folds[fold]
    training = _loader([by_id[i] for i in train_ids], cfg, training=True)
    validation = _loader([by_id[i] for i in val_ids], cfg, training=False)
    model = build_model(cfg).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    directory = _run_directory(cfg, run_id, fold)
    checkpoint = directory / "best.pt"
    split_hash = file_sha256(split_file)
    git_sha = _git_sha()
    git_dirty = _git_dirty()
    best = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        seen = 0
        for images, labels in training:
            optimizer.zero_grad()
            loss = criterion(model(images.to(target)), labels.to(target))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(images)
            seen += len(images)
        model.eval()
        val_total = 0.0
        probabilities: list[list[float]] = []
        truth: list[list[float]] = []
        with torch.no_grad():
            for images, labels in validation:
                logits = model(images.to(target))
                val_total += float(criterion(logits, labels.to(target)).cpu()) * len(images)
                probabilities.extend(logits.sigmoid().cpu().tolist())
                truth.extend(labels.tolist())
        metrics = {
            "train_loss": total / seen,
            "val_loss": val_total / len(val_ids),
            "macro_f1": macro_f1(
                per_class_prf(
                    np.asarray(truth, dtype=np.float64),
                    np.asarray(probabilities, dtype=np.float64),
                    cfg["threshold"],
                )
            ),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Non-finite training metrics")
        history.append(metrics)
        if metrics["val_loss"] < best - cfg["min_delta"]:
            best = metrics["val_loss"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
            sidecar = {
                "config": cfg,
                "seed": seed,
                "epoch": epoch,
                "metrics": metrics,
                "git_sha": git_sha,
                "git_dirty": git_dirty,
                "dataset_identity": identity,
                "split_file_sha256": split_hash,
                "fold_index": fold,
                "class_codes": list(CODES),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
            checkpoint.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    return FitResult(checkpoint, history, best_epoch)


def load_checkpoint(checkpoint: Path, cfg: dict[str, Any], *, device: str = "cpu") -> Any:
    """Check byte integrity and class order before loading weights."""
    import torch

    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    if sidecar["class_codes"] != list(CODES) or sidecar["checkpoint_sha256"] != file_sha256(
        checkpoint
    ):
        raise ValueError("Checkpoint provenance mismatch")
    model = build_model({**cfg, "pretrained": False}).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model


def out_of_fold(
    samples: Sequence[Sample],
    split_file: Path,
    cfg: dict[str, Any],
    *,
    run_id: str,
    device: str | None = None,
) -> dict[str, list[float]]:
    """Score each non-test item once using only its outer fold's best checkpoint."""
    splits = read_splits(split_file)
    splits.validate([s.sample_id for s in samples])
    identity = _verify(samples, split_file, cfg)
    by_id = {s.sample_id: s for s in samples}
    result: dict[str, list[float]] = {}
    for k, (_, _, heldout) in enumerate(splits.folds):
        fitted = _fit(samples, split_file, k, cfg, run_id=run_id, device=device, identity=identity)
        model = load_checkpoint(fitted.checkpoint, cfg, device=device or select_device())
        probabilities = predict(model, [by_id[i] for i in heldout], cfg, device=device)
        result.update(zip(heldout, probabilities, strict=True))
    if set(result) != set(by_id) - set(splits.test):
        raise ValueError("Invalid out-of-fold coverage")
    return result


def _required_path(cfg: dict[str, Any], key: str, base: Path) -> Path:
    """Resolve a path from the config relative to the config file; error if the key is unset."""
    if not cfg.get(key):
        raise ValueError(f"Set {key} in the training config")
    return (base / Path(str(cfg[key])).expanduser()).resolve()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry for the splits, fit and oof modes; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["splits", "fit", "oof"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--groups", type=Path, help="Required id -> group JSON for splits mode")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args(argv)
    with args.config.open() as stream:
        cfg = yaml.safe_load(stream)
    base = args.config.resolve().parent
    index = _required_path(cfg, "sample_index", base)
    samples = [
        Sample(row["id"], (index.parent / row["image_path"]).resolve(), tuple(row["labels"]))
        for row in json.loads(index.read_text())
    ]
    split_file = _required_path(cfg, "splits_file", base)
    if args.mode == "splits":
        if args.groups is None:
            parser.error("splits requires --groups; no grouping default is safe")
        groups = json.loads(args.groups.read_text())
        splits = make_splits(
            [s.sample_id for s in samples],
            groups.__getitem__,
            test_fraction=cfg["test_fraction"],
            n_folds=cfg["n_folds"],
            val_fraction=cfg["val_fraction"],
            seed=int(get_config()["seed"]),
        )
        manifest = build_dataset_manifest(samples)
        if split_file.exists() or manifest_path(split_file).exists():
            raise ValueError("Refusing to overwrite persisted splits or manifest")
        try:
            write_splits(splits, split_file)
            manifest_path(split_file).write_text(json.dumps(manifest, indent=2) + "\n")
        except BaseException:
            # A partial pair must not block a retry through the overwrite guard.
            split_file.unlink(missing_ok=True)
            manifest_path(split_file).unlink(missing_ok=True)
            raise
    else:
        cfg["checkpoint_dir"] = str(_required_path(cfg, "checkpoint_dir", base))
        if not isinstance(cfg.get("run_id"), str):
            raise ValueError("Set run_id in the training config")
        if args.mode == "fit":
            fitted = fit(samples, split_file, args.fold, cfg, run_id=cfg["run_id"])
            print(fitted.checkpoint)
        else:
            output = _required_path(cfg, "oof_file", base)
            if output.exists():
                raise ValueError("Refusing to overwrite out-of-fold predictions")
            result = out_of_fold(samples, split_file, cfg, run_id=cfg["run_id"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
