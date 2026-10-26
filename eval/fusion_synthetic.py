"""Synthetic fusion evaluation: paste defect patches onto driving frames, then score what is found.

Pure helpers for tools/eval_fusion_synthetic.py. Depends on numpy and Pillow only; the pipeline,
the frames and the patches come from the tool.

The idea: a patch pasted at a known place is a defect whose position, size and surroundings we
know, so the answer key is exact. The road ground truth is a hand-drawn polygon per frame, so it
does not come from the road model being tested. A placement counts as "on road" only when the
whole box lies inside the polygon shrunk by a margin, and "off road" only when it lies outside the
polygon grown by the same margin, so a polygon that is a few pixels off cannot change the answer.

The pipeline reports tile-sized defect boxes, not tight ones, so a patch counts as found when a
reported defect box contains the patch center. The box overlap (IoU) is recorded but is low by
design for a small patch inside a large tile. in_road is judged on the box the pipeline reports,
against the polygon, using the same rule the fusion code uses (share of the box on the road).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from common.geometry import box_iou
from eval.classifier import wilson_interval

Box = tuple[float, float, float, float]
Mask = npt.NDArray[np.bool_]
NearFn = Callable[[Box], set[str]]
SAMPLING_TRIES = 300


# ------------------------------------------------------------------------------- the road


def polygon_mask(polygon: Sequence[tuple[float, float]], size: tuple[int, int]) -> Mask:
    """Boolean (height, width) mask of a polygon; size is (width, height)."""
    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in polygon], fill=1)
    return np.asarray(canvas, dtype=np.uint8) > 0


def box_share(box: Box, mask: Mask) -> float:
    """Share of the box (clipped to the image) that lies on the mask; 0.0 with no area."""
    height, width = mask.shape
    x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
    x2, y2 = min(width, round(box[2])), min(height, round(box[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].mean())


def _window_count(mask: Mask, radius: int) -> npt.NDArray[np.int64]:
    """For each pixel, how many mask pixels lie in the (2r+1) square around it.

    The image edge repeats outward, so a road that runs off the bottom of the frame is not
    eroded there. Uses an integral image, so it is linear in the pixel count.
    """
    height, width = mask.shape
    side = 2 * radius + 1
    padded = np.pad(mask.astype(np.int64), radius, mode="edge")
    total = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (
        total[side : side + height, side : side + width]
        - total[:height, side : side + width]
        - total[side : side + height, :width]
        + total[:height, :width]
    )


def grow(mask: Mask, pixels: int) -> Mask:
    """The mask dilated by a square of the given radius."""
    near: Mask = _window_count(mask, pixels) > 0
    return near


def shrink(mask: Mask, pixels: int) -> Mask:
    """The mask eroded by a square of the given radius."""
    full: Mask = _window_count(mask, pixels) == (2 * pixels + 1) ** 2
    return full


# ------------------------------------------------------------------------------ placement


def sample_box(
    road: Mask,
    patch_size: tuple[int, int],
    *,
    on_road: bool,
    blocked: Sequence[Box],
    margin: int,
    min_y: float,
    rng: random.Random,
) -> Box | None:
    """A random box for a patch of (width, height), or None if no place was found.

    on_road: wholly inside the road shrunk by margin. Otherwise wholly outside the road grown by
    margin. Never overlaps a blocked box (a ground-truth object), and never starts above min_y
    (to keep patches out of the sky and the far background).
    """
    height, width = road.shape
    w, h = patch_size
    if w > width or h > height or min_y > height - h:
        return None
    zone = shrink(road, margin) if on_road else ~grow(road, margin)
    for _ in range(SAMPLING_TRIES):
        x1 = rng.randint(0, width - w)
        y1 = rng.randint(max(0, round(min_y)), height - h)
        if not zone[y1 : y1 + h, x1 : x1 + w].all():
            continue
        box = (float(x1), float(y1), float(x1 + w), float(y1 + h))
        if any(_overlaps(box, other) for other in blocked):
            continue
        return box
    return None


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def paste_patch(
    frame: npt.NDArray[np.uint8], patch: npt.NDArray[np.uint8], box: Box, feather: float = 0.15
) -> npt.NDArray[np.uint8]:
    """A copy of the frame with the patch resized into the box and blended in at its edges.

    feather is the width of the soft edge as a share of the box's shorter side (0 = hard edge).
    """
    x1, y1, x2, y2 = (round(v) for v in box)
    out = frame.copy()
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return out
    resized = np.asarray(Image.fromarray(patch).resize((w, h), Image.Resampling.LANCZOS))
    alpha = np.ones((h, w))
    ramp = feather * min(w, h)
    if ramp > 0:
        across = np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5) / ramp
        down = np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5) / ramp
        alpha = np.clip(np.minimum.outer(down, across), 0.0, 1.0)
    region = out[y1:y2, x1:x2].astype(float)
    blended = region * (1 - alpha[..., None]) + resized.astype(float) * alpha[..., None]
    out[y1:y2, x1:x2] = np.rint(blended).astype(np.uint8)
    return out


# ------------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Reported:
    """One defect the pipeline reported: its box, confidence, in_road and near-object labels."""

    box: Box
    confidence: float
    in_road: bool | None
    near: tuple[str, ...]


@dataclass(frozen=True)
class Trial:
    """The outcome of one pasted patch."""

    truth_box: Box
    on_road: bool
    found: bool
    iou: float
    in_road_correct: bool | None  # None: not found, or the pipeline said unknown
    reported_in_road: bool | None
    near_expected: tuple[str, ...]
    near_reported: tuple[str, ...]


def _contains_center(box: Box, truth: Box) -> bool:
    cx, cy = (truth[0] + truth[2]) / 2, (truth[1] + truth[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def score_trial(
    truth: Box,
    on_road: bool,
    reported: Sequence[Reported],
    *,
    road_truth: Mask,
    road_threshold: float,
    expected_near: NearFn,
) -> Trial:
    """Score one patch against the defects the pipeline reported for its frame.

    The covering defect with the highest confidence is the one scored. expected_near gives the
    labels of ground-truth objects that fusion's own rule would call near a box; road_truth is
    the hand-drawn road and road_threshold the share of a box that must lie on it.
    """
    covering = [d for d in reported if _contains_center(d.box, truth)]
    if not covering:
        return Trial(truth, on_road, False, 0.0, None, None, (), ())
    best = max(covering, key=lambda d: d.confidence)
    correct: bool | None = None
    if best.in_road is not None:
        expected = box_share(best.box, road_truth) >= road_threshold
        correct = best.in_road == expected
    return Trial(
        truth,
        on_road,
        True,
        box_iou(truth, best.box),
        correct,
        best.in_road,
        tuple(sorted(expected_near(best.box))),
        tuple(sorted(set(best.near))),
    )


# ------------------------------------------------------------------------------- summary


def _rate(found: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(found, total)
    return {
        "trials": total,
        "found": found,
        "rate": found / total if total else None,
        "range90": (low, high),
    }


def _bands(edges: Sequence[float], unit: str, names: Sequence[str]) -> list[str]:
    labels = [f"{names[0]} (<{edges[0]:g} {unit})"]
    for low, high, name in zip(edges, edges[1:], names[1:], strict=False):
        labels.append(f"{name} ({low:g}-{high:g} {unit})")
    labels.append(f"{names[-1]} (>={edges[-1]:g} {unit})")
    return labels


def _band_of(value: float, edges: Sequence[float]) -> int:
    return sum(value >= edge for edge in edges)


def _by_band(trials: Sequence[Trial], labels: Sequence[str], key: Callable[[Trial], int]) -> Any:
    out = {}
    for index, label in enumerate(labels):
        group = [t for t in trials if key(t) == index]
        out[label] = _rate(sum(t.found for t in group), len(group))
    return out


def summarize(
    trials: Sequence[Trial],
    *,
    size_edges_px: Sequence[float],
    y_edges_frac: Sequence[float],
    image_height: float,
) -> dict[str, Any]:
    """Detection rate by size and height, off-road leakage, in_road accuracy, near objects."""
    on = [t for t in trials if t.on_road]
    off = [t for t in trials if not t.on_road]
    found = [t for t in trials if t.found]

    size_labels = _bands(size_edges_px, "px", ("small", "medium", "large"))
    y_labels = [
        f"far (above {y_edges_frac[0]:.0%} of height)",
        f"mid ({y_edges_frac[0]:.0%}-{y_edges_frac[1]:.0%})",
        f"near (below {y_edges_frac[1]:.0%})",
    ]
    on_report = {
        "trials": len(on),
        "found": sum(t.found for t in on),
        "found_range90": wilson_interval(sum(t.found for t in on), len(on)),
        "by_size": _by_band(
            on, size_labels, lambda t: _band_of(t.truth_box[2] - t.truth_box[0], size_edges_px)
        ),
        "by_height": _by_band(
            on,
            y_labels,
            lambda t: _band_of((t.truth_box[1] + t.truth_box[3]) / 2 / image_height, y_edges_frac),
        ),
    }  # fmt: skip

    leaked = sum(t.found and t.reported_in_road is not False for t in off)
    off_report = {
        "trials": len(off),
        "not_reported": sum(not t.found for t in off),
        "reported_off_road": sum(t.found and t.reported_in_road is False for t in off),
        "leaked": leaked,
        "leak_rate": leaked / len(off) if off else None,
        "leak_range90": wilson_interval(leaked, len(off)),
    }

    scored = [t for t in found if t.in_road_correct is not None]
    right = sum(bool(t.in_road_correct) for t in scored)
    low, high = wilson_interval(right, len(scored))
    in_road_report = {
        "scored": len(scored),
        "correct": right,
        "unknown": len(found) - len(scored),
        "rate": right / len(scored) if scored else None,
        "range90": (low, high),
    }

    expected = sum(len(t.near_expected) for t in found)
    reported = sum(len(t.near_reported) for t in found)
    matched = sum(len(set(t.near_expected) & set(t.near_reported)) for t in found)
    near_report = {
        "defects": len(found),
        "expected": expected,
        "reported": reported,
        "matched": matched,
        "recall": matched / expected if expected else None,
        "precision": matched / reported if reported else None,
        "exact": sum(set(t.near_expected) == set(t.near_reported) for t in found),
    }
    return {
        "on_road": on_report,
        "off_road": off_report,
        "in_road": in_road_report,
        "near_objects": near_report,
        "median_iou": float(np.median([t.iou for t in found])) if found else None,
    }


def verdict(report: Mapping[str, Any], criteria: Mapping[str, float]) -> dict[str, Any]:
    """PASS or FAIL against the configured limits, naming every limit that is missed.

    A criterion with no data fails: an empty evaluation must not pass.
    """
    failures: list[str] = []

    def check(name: str, value: float | None, limit: float, at_least: bool) -> None:
        if value is None:
            failures.append(f"{name}: no data")
        elif at_least and value < limit:
            failures.append(f"{name}: {value:.2f} is below the minimum {limit:.2f}")
        elif not at_least and value > limit:
            failures.append(f"{name}: {value:.2f} is above the maximum {limit:.2f}")

    large = next(v for k, v in report["on_road"]["by_size"].items() if k.startswith("large"))
    off = report["off_road"]
    leak = off["leaked"] / off["trials"] if off["trials"] else None
    check(
        "detection rate, large patches", large["rate"], criteria["min_detection_rate_large"], True
    )
    check("in_road accuracy", report["in_road"]["rate"], criteria["min_in_road_accuracy"], True)
    check("off-road leak rate", leak, criteria["max_off_road_leak_rate"], False)
    check(
        "near_objects recall", report["near_objects"]["recall"], criteria["min_near_recall"], True
    )
    return {"passed": not failures, "failures": failures}
