"""Measure the pinned COCO detectors on nuScenes CAM_FRONT: accuracy, slices and CPU speed.

Usage: python -m tools.eval_detector --nuscenes-root data/nuscenes/raw [--version v1.0-mini]
           [--models NAME ...] [--timing-frames 100] [--threads N] [--out report.json]
--timing-frames 0 skips the speed test.
Needs the nuscenes-devkit and the weights from tools/fetch_detector.py. Ground truth is each
annotation's 3D box projected into the image and clamped to it, with labels mapped by
perception/labels.py. Two ground-truth sets are scored: all projected boxes, and the evaluable ones
(at least 40% visible and at least 32 x 32 px; these two cut-offs are this tool's own choice, not
the source repo's). Scenes are split 80/20 by sorted token, as docs/EVALUATION.md specifies; the
score threshold is chosen on the training scenes only (maximum F1 on evaluable ground truth) and
the headline numbers are those at that threshold on the held-out scenes. Accuracy runs on the
default device; speed runs on the CPU (--threads limits its threads, to imitate a smaller host),
one frame at a time after warm-up, and reports the median and 95th percentile. The nuScenes
images never leave this machine and nothing is written but the report. Matching, F1 and AP
are eval/models.py (same label, IoU >= 0.5, greedy by score).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.config import get_config
from eval.models import (
    PRF,
    Detection,
    Frame,
    GroundTruth,
    average_precision,
    detection_prf,
    recall_by_size,
)
from perception.detector import DETECTOR_SUBDIR, DETECTORS, detect_objects, load_detector_model
from perception.labels import FUSION_LABELS, nuscenes_label

COLLECT_THRESHOLD = 0.05  # keep nearly everything once; thresholds are applied afterwards
THRESHOLD_GRID = [round(0.1 + 0.05 * i, 2) for i in range(17)]  # 0.10 .. 0.90
MIN_VISIBILITY_LEVEL = 2  # nuScenes visibility 2 = 40-60% of the object visible
MIN_AREA_PX = 32 * 32
SIZE_EDGES = (1024.0, 9216.0)  # 32 x 32 and 96 x 96 px, the COCO small / medium / large cut-offs
_NIGHT = re.compile(r"\bnight\b", re.IGNORECASE)

# (scene token, is night, image path, ground truth)
LoadedFrame = tuple[str, bool, Path, list["GtBox"]]


@dataclass(frozen=True)
class GtBox:
    """A ground-truth box with the nuScenes visibility level (1 to 4) of its annotation."""

    label: str
    box: tuple[float, float, float, float]
    visibility: int


@dataclass(frozen=True)
class FrameData:
    scene: str
    night: bool
    detections: list[Detection]
    ground_truth: list[GtBox]


def split_scenes(tokens: Sequence[str], val_fraction: float = 0.2) -> tuple[list[str], list[str]]:
    """(train, validation): sort the tokens and hold out the last val_fraction, at least one."""
    ordered = sorted(tokens)
    n_val = max(1, round(len(ordered) * val_fraction))
    return ordered[:-n_val], ordered[-n_val:]


def is_night(description: str) -> bool:
    """Whether a nuScenes scene description contains the word "night"."""
    return bool(_NIGHT.search(description))


def evaluable(box: GtBox) -> bool:
    """Visible enough and large enough that a detector could be fairly expected to find it."""
    x1, y1, x2, y2 = box.box
    return box.visibility >= MIN_VISIBILITY_LEVEL and (x2 - x1) * (y2 - y1) >= MIN_AREA_PX


def best_threshold(frames: Sequence[Frame], grid: Sequence[float]) -> tuple[float, PRF]:
    """The grid threshold with the highest micro F1; a tie goes to the higher threshold."""
    if not grid:
        raise ValueError("need at least one threshold")
    best: tuple[float, PRF] | None = None
    for threshold in sorted(grid):
        score = detection_prf(frames, score_thr=threshold)
        if best is None or score.f1 >= best[1].f1:
            best = (threshold, score)
    assert best is not None
    return best


def latency_summary(seconds: Sequence[float]) -> dict[str, float]:
    """Sample count, median and 95th percentile (linear interpolation) of per-frame seconds."""
    if not seconds:
        raise ValueError("need at least one latency sample")
    return {
        "n": len(seconds),
        "p50_s": float(np.percentile(seconds, 50)),
        "p95_s": float(np.percentile(seconds, 95)),
    }


# ------------------------------------------------------------------- scoring


def _frames(
    data: Sequence[FrameData], *, keep_gt: Callable[[GtBox], bool], threshold: float = 0.0
) -> list[Frame]:
    return [
        (
            [d for d in f.detections if d.score >= threshold],
            [GroundTruth(g.label, g.box) for g in f.ground_truth if keep_gt(g)],
        )
        for f in data
    ]


def threshold_sweep(
    data: Sequence[FrameData],
    thresholds: Sequence[float],
    *,
    only_evaluable: bool,
    label: str | None = None,
) -> list[dict[str, float]]:
    """Precision, recall and F1 at each score threshold, optionally for one class only."""
    keep = evaluable if only_evaluable else _all
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        frames: list[Frame] = [
            (
                [d for d in preds if label is None or d.label == label],
                [g for g in gts if label is None or g.label == label],
            )
            for preds, gts in _frames(data, keep_gt=keep, threshold=threshold)
        ]
        score = detection_prf(frames)
        rows.append(
            {
                "threshold": threshold,
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "n_predictions": sum(len(preds) for preds, _ in frames),
                "n_ground_truth": sum(len(gts) for _, gts in frames),
            }
        )
    return rows


def _all(_: GtBox) -> bool:
    return True


def _score(data: Sequence[FrameData], threshold: float, *, only_evaluable: bool) -> dict[str, Any]:
    keep = evaluable if only_evaluable else _all
    score = detection_prf(_frames(data, keep_gt=keep), score_thr=threshold)
    return {
        "frames": len(data),
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "ap50": average_precision(_frames(data, keep_gt=keep, threshold=COLLECT_THRESHOLD)),
    }


def score_model(data: Sequence[FrameData], train_scenes: set[str]) -> dict[str, Any]:
    """Pick the threshold on training scenes, then score held-out scenes, slices and sizes."""
    train = [f for f in data if f.scene in train_scenes]
    val = [f for f in data if f.scene not in train_scenes]
    threshold, _ = best_threshold(_frames(train, keep_gt=evaluable), THRESHOLD_GRID)
    at_threshold = _frames(data, keep_gt=_all, threshold=threshold)
    return {
        "threshold": threshold,
        "train_scenes_evaluable": _score(train, threshold, only_evaluable=True),
        "held_out_evaluable": _score(val, threshold, only_evaluable=True),
        "held_out_all": _score(val, threshold, only_evaluable=False),
        "all_frames_evaluable": _score(data, threshold, only_evaluable=True),
        "day_evaluable": _score([f for f in data if not f.night], threshold, only_evaluable=True),
        "night_evaluable": _score([f for f in data if f.night], threshold, only_evaluable=True),
        "sweep_evaluable": threshold_sweep(data, THRESHOLD_GRID, only_evaluable=True),
        "sweep_all_gt": threshold_sweep(data, THRESHOLD_GRID, only_evaluable=False),
        "sweep_by_class_evaluable": {
            label: threshold_sweep(data, THRESHOLD_GRID, only_evaluable=True, label=label)
            for label in FUSION_LABELS
        },
        "recall_by_size_all_gt": [
            {"low": lo, "high": None if math.isinf(hi) else hi, "n": n, "recall": r}
            for lo, hi, n, r in recall_by_size(at_threshold, SIZE_EDGES)
        ],
    }


# ------------------------------------------------------------------- nuScenes


def _project(nusc: Any, sample: dict[str, Any]) -> tuple[Path, list[GtBox]]:
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    from PIL import Image

    token = sample["data"]["CAM_FRONT"]
    path, boxes, intrinsic = nusc.get_sample_data(token, box_vis_level=BoxVisibility.ANY)
    with Image.open(path) as image:
        width, height = image.size
    out: list[GtBox] = []
    for box in boxes:
        label = nuscenes_label(box.name)
        if label is None:
            continue
        pts = view_points(box.corners(), np.array(intrinsic), normalize=True)[:2]
        x1, y1 = max(0.0, float(pts[0].min())), max(0.0, float(pts[1].min()))
        x2, y2 = min(float(width), float(pts[0].max())), min(float(height), float(pts[1].max()))
        if x2 > x1 and y2 > y1:
            visibility = int(nusc.get("sample_annotation", box.token)["visibility_token"])
            out.append(GtBox(label, (x1, y1, x2, y2), visibility))
    return Path(path), out


def _load_frames(nusc: Any) -> list[LoadedFrame]:
    scenes = {s["token"]: s for s in nusc.scene}
    frames: list[LoadedFrame] = []
    for sample in nusc.sample:
        scene = scenes[sample["scene_token"]]
        path, boxes = _project(nusc, sample)
        frames.append((scene["token"], is_night(scene["description"]), path, boxes))
    return frames


def _rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def run_model(
    name: str,
    frames: list[LoadedFrame],
    checkpoint_dir: Path,
    timing_frames: int,
    threads: int | None = None,
) -> dict[str, Any]:
    spec = DETECTORS[name]
    weights = checkpoint_dir / DETECTOR_SUBDIR / spec.filename
    model = load_detector_model(weights, spec)
    data = [
        FrameData(
            scene,
            night,
            [
                Detection(o.label, o.score, o.box)
                for o in detect_objects(model, _rgb(path), score_threshold=COLLECT_THRESHOLD)
            ],
            boxes,
        )
        for scene, night, path, boxes in frames
    ]
    train, _ = split_scenes(sorted({f[0] for f in frames}))
    result = score_model(data, set(train))
    if timing_frames > 0:
        result["cpu_latency"] = _time_on_cpu(name, weights, frames, timing_frames, threads)
    return result


def _time_on_cpu(
    name: str, weights: Path, frames: list[LoadedFrame], count: int, threads: int | None
) -> dict[str, Any]:
    import torch

    if threads is not None:
        torch.set_num_threads(threads)

    model = load_detector_model(weights, DETECTORS[name], device="cpu")
    step = max(1, len(frames) // count)
    paths = [f[2] for f in frames[::step]][:count]
    seconds: list[float] = []
    for i, path in enumerate(paths):
        frame = _rgb(path)
        start = time.perf_counter()
        detect_objects(model, frame, score_threshold=0.5, device="cpu")
        if i >= 3:  # the first few calls are warm-up
            seconds.append(time.perf_counter() - start)
    return {
        **latency_summary(seconds),
        "threads": torch.get_num_threads(),
        "cpu": platform.processor(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--models", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    parser.add_argument("--timing-frames", type=int, default=100)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from nuscenes.nuscenes import NuScenes

    checkpoint_dir = args.checkpoint_dir or Path(get_config()["paths"]["checkpoint_dir"])
    nusc = NuScenes(version=args.version, dataroot=str(args.nuscenes_root), verbose=False)
    frames = _load_frames(nusc)
    train, held_out = split_scenes(sorted({f[0] for f in frames}))
    report: dict[str, Any] = {
        "version": args.version,
        "frames": len(frames),
        "train_scenes": train,
        "held_out_scenes": held_out,
        "models": {
            name: run_model(name, frames, checkpoint_dir, args.timing_frames, args.threads)
            for name in args.models
        },
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
