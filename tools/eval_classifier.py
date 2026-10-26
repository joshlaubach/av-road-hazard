"""Compare tile-classifier checkpoints: validation tiles, clean probe tiles, alerts per frame.

Usage: python -m tools.eval_classifier --train-config CFG.yaml --checkpoint baseline=best.pt
           --checkpoint NAME=best.pt [...] [--fold 0] [--probe-negatives FILE.json]
           [--nuscenes-root DIR --review-key KEY.json --review-calls CSV] [--export-dir DIR
           --export-key-dir DIR] [--out report.json]
The first checkpoint is the baseline. The training config supplies the splits, tile index, tile
cache and sample index. Per checkpoint the report has, at the per-class thresholds of
configs/default.yaml:
  validation  precision, recall, AUROC and flagged count per class on the fold's validation tiles
              (the capped training-style pool, not the natural mix), and recall at the false-alarm
              rate the baseline has at its threshold;
  probe       how many annotator-confirmed clean tiles (a JSON list of {"image_id","row","col"},
              kept out of training) are still flagged, with a 90% Wilson range;
  frames      alerts per frame on every nuScenes CAM_FRONT frame through the shipped pipeline (road
              mask and detected objects, identical for each checkpoint), by day/night, defect type,
              road cover and review bucket (Crack, No crack, Ambiguous, Unreviewed); and a
              PASS/FAIL recall guard against the baseline.
Never-reviewed alerts can be exported as an annotation batch. Needs USE_STUBS=0 for the frame part.
The images never leave this machine; the report holds numbers and hashes only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.config import get_config
from defects.infer import Classifier, defect_settings
from defects.labels import CODES
from eval.classifier import (
    Box,
    check_calls,
    class_report,
    flagged_share,
    recall_at_baseline_fpr,
    recall_guard,
    summarize_alerts,
    wilson_interval,
)
from eval.provenance import provenance

Crop = Any  # a PIL image
Reviewed = dict[tuple[str, Box], str]


@dataclass(frozen=True)
class Context:
    """The resolved training config and the folder it lives in."""

    cfg: dict[str, Any]
    base: Path


def load_context(train_config: Path) -> Context:
    from defects.train import _resolve_optional_paths

    base = train_config.resolve().parent
    cfg = _resolve_optional_paths(yaml.safe_load(train_config.read_text()), base)
    return Context({**cfg, "num_workers": 4, "persistent_workers": False}, base)


def shipped_thresholds() -> tuple[float, ...]:
    return defect_settings(get_config()).thresholds


def load_model(checkpoint: Path, ctx: Context) -> tuple[Any, str]:
    from defects.model import load_checkpoint
    from defects.runtime import select_device

    device = select_device()
    return load_checkpoint(checkpoint, ctx.cfg, device=device).eval(), device


def load_classifier(checkpoint: Path) -> Classifier:
    """A checkpoint as find_defects wants it: the model with the config it was trained with."""
    from defects.model import load_checkpoint

    config = json.loads(checkpoint.with_suffix(".json").read_text())["config"]
    return Classifier(load_checkpoint(checkpoint, config).eval(), config)


def score_crops(model: Any, device: str, crops: list[Crop], ctx: Context) -> np.ndarray:
    """Class probabilities, one row per crop, through the evaluation transform."""
    import torch

    from defects.dataset import image_transform
    from defects.runtime import forward_logits

    transform = image_transform(ctx.cfg, training=False)
    batch = int(ctx.cfg["batch_size"])
    rows = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            x = torch.stack([transform(c) for c in crops[start : start + batch]])
            rows.append(forward_logits(model, x, device, ctx.cfg).sigmoid().cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, len(CODES)))


def score_validation(model: Any, device: str, ctx: Context, fold: int) -> tuple[Any, Any]:
    """(truth, probabilities) for the fold's validation tiles."""
    import torch

    from defects.runtime import forward_logits
    from defects.splits import read_splits
    from defects.tile_dataset import TileDataset, tile_pools
    from defects.tile_index import read_tile_index
    from defects.train import _data_loader

    cfg = ctx.cfg
    train_ids, val_ids, _ = read_splits((ctx.base / cfg["splits_file"]).resolve()).folds[fold]
    index = read_tile_index(Path(cfg["tile_index"]))
    _, tiles = tile_pools(index, train_ids, val_ids, cfg, seed=int(get_config()["seed"]))
    loader = _data_loader(
        TileDataset(tiles, Path(cfg["tile_cache_dir"]), cfg, training=False), cfg, training=False
    )
    probs: list[list[float]] = []
    truth: list[list[float]] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = forward_logits(model, images, device, cfg)  # same path as scoring frames
            probs.extend(logits.sigmoid().cpu().tolist())
            truth.extend(labels.tolist())
    return np.asarray(truth), np.asarray(probs)


def probe_crops(ctx: Context, probe_file: Path) -> list[Crop]:
    """Crops of the listed RDD2022 tiles, in file order."""
    from PIL import Image

    from defects.tile_index import image_tiles, read_tile_index

    index = read_tile_index(Path(ctx.cfg["tile_index"]))
    sample_index = (ctx.base / ctx.cfg["sample_index"]).resolve()
    rows = json.loads(sample_index.read_text())
    paths = {r["id"]: sample_index.parent / r["image_path"] for r in rows}
    crops = []
    for row in json.loads(probe_file.read_text()):
        grid = {
            (t.row, t.col): t.box
            for t in image_tiles(index.params, row["image_id"], index.images[row["image_id"]])
        }
        with Image.open(paths[row["image_id"]]) as image:
            crops.append(image.convert("RGB").crop(grid[(row["row"], row["col"])]))
    return crops


def load_reviewed(key_file: Path, calls_csv: Path) -> Reviewed:
    """The human call for each reviewed nuScenes tile, keyed by (frame, tile box)."""
    key = json.loads(key_file.read_text())
    with calls_csv.open(newline="") as handle:
        human = {r["tile_id"]: r["call"] for r in csv.DictReader(handle)}
    calls = check_calls(list(key), human)
    return {(v["frame"], tuple(v["box"])): calls[t] for t, v in key.items()}


def with_interval(count: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(count, total)
    return {"count": count, "total": total, "low90": low, "high90": high}


def validation_part(
    scores: tuple[Any, Any], base_probs: Any, thresholds: tuple[float, ...]
) -> dict[str, Any]:
    truth, probs = scores
    report = class_report(truth, probs, thresholds, CODES)
    matched = recall_at_baseline_fpr(truth, base_probs, probs, thresholds, CODES)
    return {c: {**report[c], "recall_at_baseline_fpr": matched[c]["recall"]} for c in CODES}


def frames_part(
    checkpoint: Path,
    contexts: list[Any],
    root: Path,
    reviewed: Reviewed,
) -> tuple[dict[str, Any], list[Any]]:
    from tools.eval_frames import collect_alerts

    alerts = collect_alerts(contexts, root, classifier=load_classifier(checkpoint))
    night = sum(c.night for c in contexts)
    return summarize_alerts(
        alerts, frames=len(contexts), night_frames=night, reviewed=reviewed
    ), alerts


def prepare_contexts(root: Path, version: str) -> list[Any]:
    """Road mask and detected objects for every frame, computed once for all checkpoints."""
    from defects.road_crop import MODEL_SUBDIR, RoadSegmenter
    from perception.infer import detect
    from tools.eval_frames import load_frames, prepare_frames

    if get_config()["pipeline"]["use_stubs"]:
        raise SystemExit("set USE_STUBS=0: the frame part needs the real road and object models")
    weights = Path(get_config()["paths"]["checkpoint_dir"]) / MODEL_SUBDIR
    return prepare_frames(load_frames(root, version), root, RoadSegmenter(weights), detect)


def model_files() -> dict[str, Path]:
    """The road and detector weight files the frame part depends on, for the provenance block."""
    from defects.road_crop import MODEL_FILES, MODEL_SUBDIR
    from perception.detector import DETECTOR_SUBDIR
    from perception.infer import detector_settings

    checkpoints = Path(get_config()["paths"]["checkpoint_dir"])
    files = {f"road/{name}": checkpoints / MODEL_SUBDIR / name for name in MODEL_FILES}
    name, _ = detector_settings(get_config())
    files["detector"] = checkpoints / DETECTOR_SUBDIR / f"{name}.safetensors"
    return files


def evaluate(
    checkpoints: dict[str, Path],
    ctx: Context,
    *,
    fold: int = 0,
    probe_file: Path | None = None,
    nuscenes: tuple[Path, str, Path, Path] | None = None,
    export: tuple[Path, Path] | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    """The report; the first checkpoint is the baseline. nuscenes is (root, version, key, calls)."""
    thresholds = shipped_thresholds()
    names = list(checkpoints)
    scored: dict[str, tuple[Any, Any]] = {}
    probe: dict[str, Any] = {}
    crops = probe_crops(ctx, probe_file) if probe_file else []
    for name, path in checkpoints.items():
        model, device = load_model(path, ctx)
        scored[name] = score_validation(model, device, ctx, fold)
        if crops:
            probe[name] = flagged_share(score_crops(model, device, crops, ctx), thresholds)
    base_probs = scored[names[0]][1]
    result: dict[str, Any] = {name: {} for name in names}
    for name in names:
        result[name]["validation"] = validation_part(scored[name], base_probs, thresholds)
        if name in probe:
            result[name]["probe"] = {
                **probe[name],
                **with_interval(probe[name]["flagged"], probe[name]["tiles"]),
            }
    files = {f"checkpoint/{n}": p for n, p in checkpoints.items()}
    cracks_total = 0
    if nuscenes:
        root, version, key_file, calls_csv = nuscenes
        reviewed = load_reviewed(key_file, calls_csv)
        contexts = prepare_contexts(root, version)
        cracks_total = sum(call == "Crack" for call in reviewed.values())
        every_alert = []
        for name, path in checkpoints.items():
            summary, alerts = frames_part(path, contexts, root, reviewed)
            result[name]["frames"] = summary
            every_alert += alerts
        if export:
            from tools.eval_frames import export_unreviewed

            # Unreviewed alerts of every checkpoint, so the new model's extra alerts get labelled.
            export_unreviewed(every_alert, root, export[0], export[1], reviewed, prefix="n2")
        files |= {"review_key": key_file, "review_calls": calls_csv, **model_files()}
    base_recall = {c: result[names[0]]["validation"][c]["recall"] for c in CODES}
    for name in names[1:]:
        reviewed_run = "frames" in result[name]
        result[name]["guard"] = recall_guard(
            base_recall,
            {c: result[name]["validation"][c]["recall"] for c in CODES},
            cracks_kept=result[name]["frames"]["buckets"]["Crack"] if reviewed_run else None,
            cracks_total=cracks_total if reviewed_run else None,
            baseline_cracks_kept=result[names[0]]["frames"]["buckets"]["Crack"]
            if reviewed_run
            else None,
        )
    if probe_file:
        files["probe"] = probe_file
    return {
        "baseline": names[0],
        "fold": fold,
        "provenance": provenance(
            files,
            thresholds=dict(zip(CODES, thresholds, strict=True)),
            repo=repo or Path(__file__).resolve().parents[1],
        ),
        "checkpoints": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--probe-negatives", type=Path, default=None)
    parser.add_argument("--nuscenes-root", type=Path, default=None)
    parser.add_argument("--nuscenes-version", default="v1.0-mini")
    parser.add_argument("--review-key", type=Path, default=None)
    parser.add_argument("--review-calls", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--export-key-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    frame_args = (args.nuscenes_root, args.review_key, args.review_calls)
    if any(frame_args) and not all(frame_args):
        parser.error("--nuscenes-root, --review-key and --review-calls go together")
    if bool(args.export_dir) != bool(args.export_key_dir):
        parser.error("--export-dir and --export-key-dir go together")
    if args.export_dir and not all(frame_args):
        parser.error("exporting alerts needs the nuScenes arguments")
    checkpoints = {n: Path(p) for n, _, p in (spec.partition("=") for spec in args.checkpoint)}
    nuscenes = (
        (args.nuscenes_root, args.nuscenes_version, args.review_key, args.review_calls)
        if all(frame_args)
        else None
    )
    export = (args.export_dir, args.export_key_dir) if args.export_dir else None
    report = evaluate(
        checkpoints,
        load_context(args.train_config),
        fold=args.fold,
        probe_file=args.probe_negatives,
        nuscenes=nuscenes,
        export=export,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("USE_STUBS", "0")
    raise SystemExit(main())
