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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.config import get_config
from common.progress import make_bar
from common.tracking import open_tracker
from defects.dataset import (
    DefectDataset,
    Sample,
    build_dataset_manifest,
    file_sha256,
)
from defects.labels import CODES
from defects.model import build_model
from defects.provenance import _git_dirty, _git_sha, _run_directory, _verify, manifest_path
from defects.runtime import amp_enabled, forward_logits, select_device
from defects.splits import make_splits, read_splits, write_splits
from defects.tile_dataset import TileDataset, check_cached, image_scores, tile_pools
from defects.tile_index import TileParams, TileRecord, read_tile_index
from defects.tile_predict import predict_tiled
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
    "predict_tiled",
    "select_device",
    "setup_determinism",
]


def setup_determinism(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch; by default force deterministic kernels (repeatable runs).

    deterministic=False keeps the seeds but lets cuDNN pick its fastest kernels (about 10% faster
    in a measured run) at the price of bitwise repeatability.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


# Moved to defects.runtime so tile scoring can share them; the old names stay importable.
_amp_enabled = amp_enabled
_forward = forward_logits


def _data_loader(dataset: Any, cfg: dict[str, Any], *, training: bool) -> Any:
    """DataLoader over a dataset; training shuffles and drops a short last batch."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=training,
        drop_last=training and len(dataset) > cfg["batch_size"],
        num_workers=cfg["num_workers"],
        # Windows respawns workers every epoch otherwise (about 20 s each); needs workers > 0.
        persistent_workers=bool(cfg.get("persistent_workers", False)) and cfg["num_workers"] > 0,
    )


def _loader(samples: Sequence[Sample], cfg: dict[str, Any], *, training: bool) -> Any:
    """DataLoader over whole images."""
    return _data_loader(DefectDataset(samples, cfg, training=training), cfg, training=training)


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


@dataclass(frozen=True)
class _FoldData:
    """The loaders of one fold, plus what only the tile path knows."""

    training: Any
    validation: Any
    val_tiles: list[TileRecord] | None  # None: whole images, no image-level aggregation
    sidecar: dict[str, Any]  # extra provenance written next to the checkpoint


def _fold_data(
    by_id: dict[str, Sample],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    cfg: dict[str, Any],
    seed: int,
) -> _FoldData:
    """Whole-image loaders, or cached-tile loaders when the config names a tile index."""
    if not cfg.get("tile_index"):
        return _FoldData(
            _loader([by_id[i] for i in train_ids], cfg, training=True),
            _loader([by_id[i] for i in val_ids], cfg, training=False),
            None,
            {"input": "images"},
        )
    index_path = Path(cfg["tile_index"])
    index = read_tile_index(index_path)
    params = TileParams.from_config(cfg)
    if index.params != params:
        raise ValueError(
            f"{index_path} was built with other tile parameters than this config: "
            f"{index.params} vs {params}"
        )
    if not cfg.get("tile_cache_dir"):
        raise ValueError("Set tile_cache_dir in the training config when tile_index is set")
    cache_dir = Path(cfg["tile_cache_dir"])
    train_tiles, val_tiles = tile_pools(index, train_ids, val_ids, cfg, seed=seed)
    if not train_tiles or not val_tiles:
        raise ValueError(
            f"the tile pool is empty for this fold (train {len(train_tiles)}, "
            f"validation {len(val_tiles)}); check the tile index and the clean-tile settings"
        )
    check_cached([*train_tiles, *val_tiles], cache_dir)
    return _FoldData(
        _data_loader(TileDataset(train_tiles, cache_dir, cfg, training=True), cfg, training=True),
        _data_loader(TileDataset(val_tiles, cache_dir, cfg, training=False), cfg, training=False),
        val_tiles,
        {
            "input": "tiles",
            "tile_params": asdict(params),
            "tile_index_sha256": file_sha256(index_path),
        },
    )


def _train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    target: str,
    cfg: dict[str, Any],
    desc: str,
) -> float:
    """One pass over the training data; returns the mean loss per item."""
    model.train()
    total = 0.0
    seen = 0
    with make_bar(loader, desc=desc, total=len(loader), enabled=cfg.get("progress")) as bar:
        for images, labels in bar:
            optimizer.zero_grad()
            loss = criterion(_forward(model, images, target, cfg), labels.to(target))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(images)
            seen += len(images)
            bar.set_postfix(loss=f"{total / seen:.4f}")
    return total / seen


def _validate(
    model: Any, loader: Any, criterion: Any, target: str, cfg: dict[str, Any], desc: str
) -> tuple[float, list[list[float]], list[list[float]]]:
    """Mean loss per item, sigmoid probabilities and labels over the validation data."""
    import torch

    model.eval()
    total = 0.0
    seen = 0
    probabilities: list[list[float]] = []
    truth: list[list[float]] = []
    with (
        torch.no_grad(),
        make_bar(loader, desc=desc, total=len(loader), enabled=cfg.get("progress")) as bar,
    ):
        for images, labels in bar:
            logits = _forward(model, images, target, cfg)
            total += float(criterion(logits, labels.to(target)).cpu()) * len(images)
            seen += len(images)
            probabilities.extend(logits.sigmoid().cpu().tolist())
            truth.extend(labels.tolist())
    return total / seen, probabilities, truth


def _macro_f1(truth: Any, probabilities: Any, threshold: float) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    return macro_f1(per_class_prf(truth_array, probability_array, threshold))


def _epoch_metrics(
    train_loss: float,
    val_loss: float,
    probabilities: list[list[float]],
    truth: list[list[float]],
    data: _FoldData,
    by_id: dict[str, Sample],
    val_ids: Sequence[str],
    threshold: float,
) -> dict[str, float]:
    """Loss and macro-F1 per item (a tile or an image).

    With tiles, pool_image_macro_f1 scores each validation image by its highest tile probability.
    It uses the capped validation pool (few clean tiles), so it flatters the model compared with
    scoring every tile of an image; val_images_unscored counts the images with no tile in the pool.
    """
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "macro_f1": _macro_f1(truth, probabilities, threshold),
    }
    if data.val_tiles is not None:
        scores = image_scores(data.val_tiles, probabilities)
        ids = sorted(i for i in val_ids if i in scores)
        metrics["pool_image_macro_f1"] = _macro_f1(
            [by_id[i].labels for i in ids], [scores[i] for i in ids], threshold
        )
        metrics["val_images_unscored"] = float(len(val_ids) - len(ids))
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite training metrics")
    return metrics


# Config values copied into the tracker as run parameters.
_TRACKED_KEYS = (
    "model_name",
    "pretrained",
    "input_size",
    "lr",
    "weight_decay",
    "batch_size",
    "epochs",
    "patience",
    "head_dropout",
    "drop_path",
    "threshold",
    "amp",
    "channels_last",
    "fused_optimizer",
    "deterministic",
    "tile_size",
    "tile_overlap",
    "positive_overlap",
    "negative_overlap",
    "clean_per_positive",
    "min_clean_per_image",
)


def _tracking_uri(cfg: dict[str, Any]) -> str | None:
    """A sqlite MLflow store at cfg["tracking_db"], or None (tracking off)."""
    database = cfg.get("tracking_db")
    if not database:
        return None
    return f"sqlite:///{Path(str(database)).expanduser().resolve().as_posix()}"


def _run_params(cfg: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """What the tracker records about a run: fold, seed, input kind, code version, settings."""
    return {
        "fold": provenance["fold_index"],
        "seed": provenance["seed"],
        "input": provenance["input"],
        "git_sha": provenance["git_sha"],
        **{k: cfg[k] for k in _TRACKED_KEYS if k in cfg},
    }


def _save_checkpoint(
    model: Any, checkpoint: Path, provenance: dict[str, Any], epoch: int, metrics: dict[str, float]
) -> None:
    """Write the weights, then a sidecar JSON that pins them (hash) to their provenance."""
    import torch

    torch.save(model.state_dict(), checkpoint)
    sidecar = {
        **provenance,
        "epoch": epoch,
        "metrics": metrics,
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")


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
    _amp_enabled(cfg)  # fail on an unsupported amp value before any work
    seed = int(get_config()["seed"])
    setup_determinism(seed, bool(cfg.get("deterministic", True)))
    target = device or select_device()
    by_id = {s.sample_id: s for s in samples}
    train_ids, val_ids, _ = splits.folds[fold]
    data = _fold_data(by_id, train_ids, val_ids, cfg, seed)
    model = build_model(cfg).to(target)
    if cfg.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        fused=bool(cfg.get("fused_optimizer", False)) and target == "cuda",
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    directory = _run_directory(cfg, run_id, fold)
    checkpoint = directory / "best.pt"
    provenance = {
        "config": cfg,
        "seed": seed,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "dataset_identity": identity,
        "split_file_sha256": file_sha256(split_file),
        "fold_index": fold,
        "class_codes": list(CODES),
        **data.sidecar,
    }
    best = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    with open_tracker(
        _tracking_uri(cfg),
        experiment=str(cfg.get("tracking_experiment", "defects")),
        run_name=f"{run_id}-fold{fold}",
    ) as tracker:
        tracker.log_params(_run_params(cfg, provenance))
        for epoch in range(cfg["epochs"]):
            label = f"fold {fold} epoch {epoch + 1}/{cfg['epochs']}"
            train_loss = _train_epoch(
                model, data.training, optimizer, criterion, target, cfg, f"{label} train"
            )
            val_loss, probabilities, truth = _validate(
                model, data.validation, criterion, target, cfg, f"{label} val"
            )
            metrics = _epoch_metrics(
                train_loss, val_loss, probabilities, truth, data, by_id, val_ids, cfg["threshold"]
            )
            history.append(metrics)
            tracker.log_metrics(metrics, step=epoch)
            if metrics["val_loss"] < best - cfg["min_delta"]:
                best = metrics["val_loss"]
                best_epoch = epoch
                stale = 0
                _save_checkpoint(model, checkpoint, provenance, epoch, metrics)
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    break
        if best_epoch >= 0:
            tracker.log_file(checkpoint.with_suffix(".json"))
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
        held = [by_id[i] for i in heldout]
        if cfg.get("tile_index"):
            probabilities = predict_tiled(
                model,
                held,
                cfg,
                tile_size=int(cfg["tile_size"]),
                overlap=float(cfg["tile_overlap"]),
                device=device,
            )
        else:
            probabilities = predict(model, held, cfg, device=device)
        result.update(zip(heldout, probabilities, strict=True))
    if set(result) != set(by_id) - set(splits.test):
        raise ValueError("Invalid out-of-fold coverage")
    return result


def _required_path(cfg: dict[str, Any], key: str, base: Path) -> Path:
    """Resolve a path from the config relative to the config file; error if the key is unset."""
    if not cfg.get(key):
        raise ValueError(f"Set {key} in the training config")
    return (base / Path(str(cfg[key])).expanduser()).resolve()


def _resolve_optional_paths(cfg: dict[str, Any], base: Path) -> dict[str, Any]:
    """Copy of cfg with the optional tile and tracking paths resolved against the config folder."""
    resolved = dict(cfg)
    for key in ("tile_index", "tile_cache_dir", "tracking_db"):
        if cfg.get(key):
            resolved[key] = str(_required_path(cfg, key, base))
    return resolved


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
        cfg = _resolve_optional_paths(cfg, base)
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
