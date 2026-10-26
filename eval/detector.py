"""Pure detector-evaluation logic: scene split, ground-truth filtering, thresholds and scoring.

Used by tools/eval_detector.py; depends only on the stdlib, numpy and eval.models.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from eval.models import (
    PRF,
    Detection,
    Frame,
    GroundTruth,
    average_precision,
    detection_prf,
    match_detections,
    recall_by_size,
)

COLLECT_THRESHOLD = 0.05  # keep nearly everything once; thresholds are applied afterwards
THRESHOLD_GRID = [round(0.1 + 0.05 * i, 2) for i in range(17)]  # 0.10 .. 0.90
MIN_VISIBILITY_LEVEL = 2  # nuScenes visibility 2 = 40-60% of the object visible
MIN_AREA_PX = 32 * 32
SIZE_EDGES = (1024.0, 9216.0)  # 32 x 32 and 96 x 96 px, the COCO small / medium / large cut-offs
_NIGHT = re.compile(r"\bnight\b", re.IGNORECASE)


@dataclass(frozen=True)
class GtBox:
    """A ground-truth box with the nuScenes visibility level (1 to 4) of its annotation."""

    label: str
    box: tuple[float, float, float, float]
    visibility: int
    attribute: str | None = None  # nuScenes attribute name, e.g. "vehicle.parked"; None if absent


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


CPU_P95_LIMIT_S = 2.0  # ADR 0014: p95 per image on the intended host


def speed_verdict(
    rungs: Sequence[dict[str, Any]], limit_s: float = CPU_P95_LIMIT_S
) -> list[dict[str, Any]]:
    """Each timing (a thread count with its latency summary) with passed = p95 within limit_s."""
    return [{**rung, "passed": rung["p95_s"] <= limit_s} for rung in rungs]


# ------------------------------------------------------- vocabulary and attributes


def restrict_to_vocabulary(
    boxes: Sequence[GtBox], to_label: Callable[[str], str | None]
) -> tuple[list[GtBox], dict[str, int]]:
    """Ground truth in the labels a detector can emit, plus counts of what it cannot see.

    to_label gives the label the detector would use for a ground-truth label, or None if it has
    none (perception.labels.to_vocabulary does this for a vocabulary). Boxes with None are not
    scored (they would be guaranteed misses and would change the headline numbers); their
    evaluable ones are counted per label in the second result so the gap is reported, not hidden.
    """
    kept: list[GtBox] = []
    outside: dict[str, int] = {}
    for box in boxes:
        label = to_label(box.label)
        if label is None:
            if evaluable(box):
                outside[box.label] = outside.get(box.label, 0) + 1
            continue
        kept.append(replace(box, label=label))
    return kept, outside


def recall_by_attribute(
    data: Sequence[FrameData], threshold: float, iou_thr: float = 0.5
) -> dict[str, dict[str, float]]:
    """Recall of evaluable ground truth per nuScenes attribute (parked, moving, standing, ...).

    Boxes without an attribute are left out. Returns {attribute: {"n": count, "recall": r}}.
    """
    found: dict[str, int] = {}
    total: dict[str, int] = {}
    for frame in data:
        gts = [g for g in frame.ground_truth if evaluable(g)]
        preds = [d for d in frame.detections if d.score >= threshold]
        missed = set(
            match_detections(preds, [GroundTruth(g.label, g.box) for g in gts], iou_thr).fn
        )
        for index, gt in enumerate(gts):
            if gt.attribute is None:
                continue
            total[gt.attribute] = total.get(gt.attribute, 0) + 1
            found[gt.attribute] = found.get(gt.attribute, 0) + (index not in missed)
    return {
        name: {"n": total[name], "recall": found.get(name, 0) / total[name]}
        for name in sorted(total)
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


def score_model(
    data: Sequence[FrameData], train_scenes: set[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Pick the threshold on training scenes, then score held-out scenes, slices and sizes.

    labels are the classes to report separately in the per-class sweep.
    """
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
            for label in labels
        },
        "recall_by_attribute_evaluable": recall_by_attribute(data, threshold),
        "recall_by_size_all_gt": [
            {"low": lo, "high": None if math.isinf(hi) else hi, "n": n, "recall": r}
            for lo, hi, n, r in recall_by_size(at_threshold, SIZE_EDGES)
        ],
    }
