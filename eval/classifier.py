"""Pure classifier-evaluation helpers: flagging at per-class thresholds and the report parts.

Used by tools/eval_classifier.py and the eval notebooks; depends only on numpy and eval.models.
A tile is flagged when any class probability reaches that class's threshold, the rule
defects/tile_defects.py applies on a driving frame.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from eval.models import auroc, per_class_prf, recall_at_fpr


def flags(probs: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Boolean (tiles, classes) array: True where a probability reaches its class threshold."""
    if len(thresholds) != probs.shape[1]:
        raise ValueError("need one threshold per class")
    return np.asarray(probs >= np.asarray(thresholds))


def class_report(
    truth: np.ndarray, probs: np.ndarray, thresholds: Sequence[float], codes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Per class: precision and recall at its threshold, AUROC, tiles flagged, true positives."""
    scores = per_class_prf(truth, probs, np.asarray(thresholds))
    flagged = flags(probs, thresholds)
    return {
        code: {
            "precision": scores[k].precision,
            "recall": scores[k].recall,
            "auroc": auroc(truth[:, k], probs[:, k]),
            "flagged": int(flagged[:, k].sum()),
            "positives": int((truth[:, k] > 0.5).sum()),
        }
        for k, code in enumerate(codes)
    }


def flagged_share(probs: np.ndarray, thresholds: Sequence[float]) -> dict[str, int]:
    """Tiles flagged for any class; on tiles known to be clean, these are false alarms."""
    return {
        "tiles": int(probs.shape[0]),
        "flagged": int(flags(probs, thresholds).any(axis=1).sum()),
    }


Z_90 = 1.645  # two-sided 90% normal quantile


def wilson_interval(successes: int, total: int, z: float = Z_90) -> tuple[float, float]:
    """Wilson score interval for a rate of successes in total; 90% by default.

    Unlike the plain "rate plus or minus a margin" it is not degenerate at 0 or total and stays in
    [0, 1], so it is safe for the small counts these reviews rest on. With no samples it says
    nothing: (0, 1).
    """
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be between 0 and n, got {successes} of {total}")
    if total == 0:
        return 0.0, 1.0
    rate = successes / total
    denom = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denom
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == total else min(1.0, centre + half)
    return low, high


def recall_at_baseline_fpr(
    truth: np.ndarray,
    baseline_probs: np.ndarray,
    new_probs: np.ndarray,
    thresholds: Sequence[float],
    codes: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per class: the new model's recall at the baseline's false-positive rate at its threshold.

    It answers "is the new model better, or only quieter?": both are compared at the same share of
    clean tiles flagged. recall is NaN for a class with no negatives or no positives.
    """
    out: dict[str, dict[str, float]] = {}
    for k, code in enumerate(codes):
        negative = truth[:, k] <= 0.5
        if not negative.any() or negative.all():
            out[code] = {"baseline_fpr": math.nan, "recall": math.nan}
            continue
        fpr = float(np.mean(baseline_probs[negative, k] >= thresholds[k]))
        recall, _ = recall_at_fpr(new_probs[~negative, k], new_probs[negative, k], fpr)
        out[code] = {"baseline_fpr": fpr, "recall": recall}
    return out


UNREVIEWED = "Unreviewed"


def alert_buckets(
    alerts: Sequence[tuple[str, tuple[int, int, int, int]]],
    reviewed: dict[tuple[str, tuple[int, int, int, int]], str],
) -> dict[str, int]:
    """Count distinct alerted tiles by the human call on the same (frame, tile box).

    A tile alerting for two classes counts once; tiles nobody reviewed are Unreviewed. The counts
    sum to the number of distinct alerted tiles.
    """
    counts: dict[str, int] = defaultdict(int)
    for tile in set(alerts):
        counts[reviewed.get(tile, UNREVIEWED)] += 1
    for call in ("Crack", "No crack", "Ambiguous", UNREVIEWED):
        counts.setdefault(call, 0)
    return dict(counts)


def recall_guard(
    baseline_recall: dict[str, float],
    new_recall: dict[str, float],
    *,
    cracks_kept: int | None,
    cracks_total: int | None,
    baseline_cracks_kept: int | None = None,
    max_recall_drop: float = 0.03,
    min_cracks_kept: int = 5,
) -> dict[str, Any]:
    """PASS/FAIL that a quieter model has not lost real cracks; both limits are provisional.

    Fails when a class loses more than max_recall_drop of validation recall against the baseline
    (a drop of exactly the limit passes), when a class's recall is unknown (NaN), or when fewer of
    the human-reviewed cracks stay flagged than min_cracks_kept, capped at the number of reviewed
    cracks and at the number the baseline itself kept. cracks_kept None skips the crack check, for
    a run without a review.
    """
    reasons = []
    for code, base in baseline_recall.items():
        new = new_recall[code]
        if math.isnan(base) or math.isnan(new):
            reasons.append(f"{code}: validation recall not checked (no positives)")
        elif round(base - new, 9) > max_recall_drop:
            reasons.append(
                f"{code}: validation recall fell {base - new:.3f} (limit {max_recall_drop})"
            )
    if cracks_kept is not None and cracks_total is not None:
        need = min(min_cracks_kept, cracks_total)
        if baseline_cracks_kept is not None:
            need = min(need, baseline_cracks_kept)
        if cracks_kept < need:
            reasons.append(f"only {cracks_kept} of {cracks_total} reviewed cracks still flagged")
    return {"passed": not reasons, "reasons": reasons}


ROAD_COVER_BANDS = ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0001))  # the last band includes 1.0
UNDER_BAND = "under 0.4"  # below the road cut-off, e.g. after the cut-off was changed
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Alert:
    """One defect alert on a driving frame: where, what class, and the context it came with."""

    frame: str
    box: Box
    name: str
    confidence: float
    night: bool
    road_cover: float


def _band_label(low: float, high: float) -> str:
    return f"{low:.1f}-{min(high, 1.0):.1f}"


def summarize_alerts(
    alerts: Sequence[Alert],
    *,
    frames: int,
    night_frames: int,
    reviewed: Mapping[tuple[str, Box], str],
) -> dict[str, Any]:
    """Alerts per frame overall and by day/night, defect type, review bucket and road cover.

    frames and night_frames are the number of frames scored (alert-free frames count), because
    alerts per frame is the headline: a quieter model is one with fewer of them. Alerts count a
    tile once per class it is flagged for; buckets count distinct tiles, so they can sum to fewer.
    """
    if frames < 1:
        raise ValueError("need at least one frame")
    if not 0 <= night_frames <= frames:
        raise ValueError("night_frames must be between 0 and frames")
    night = sum(a.night for a in alerts)
    by_class: dict[str, int] = defaultdict(int)
    for alert in alerts:
        by_class[alert.name] += 1
    bands = {UNDER_BAND: sum(a.road_cover < ROAD_COVER_BANDS[0][0] for a in alerts)}
    bands |= {
        _band_label(lo, hi): sum(lo <= a.road_cover < hi for a in alerts)
        for lo, hi in ROAD_COVER_BANDS
    }
    return {
        "alerts": len(alerts),
        "per_frame": len(alerts) / frames,
        "day": _group(len(alerts) - night, frames - night_frames),
        "night": _group(night, night_frames),
        "by_class": dict(sorted(by_class.items())),
        "buckets": alert_buckets([(a.frame, a.box) for a in alerts], dict(reviewed)),
        "by_road_cover": bands,
    }


def _group(alerts: int, frames: int) -> dict[str, Any]:
    return {"frames": frames, "alerts": alerts, "per_frame": alerts / frames if frames else 0.0}


REVIEW_CALLS = ("Crack", "No crack", "Ambiguous", "Not road")


def check_calls(tile_ids: Sequence[str], calls: Mapping[str, str]) -> dict[str, str]:
    """The call for each tile id; a missing tile, a blank sheet row or an unknown word is an error.

    A half-filled answer sheet must fail loudly: a blank call would otherwise become its own
    bucket in the report.
    """
    out: dict[str, str] = {}
    for tile in tile_ids:
        if tile not in calls:
            raise ValueError(f"no call for tile {tile}")
        if calls[tile] not in REVIEW_CALLS:
            raise ValueError(f"unknown call {calls[tile]!r} for tile {tile}; use {REVIEW_CALLS}")
        out[tile] = calls[tile]
    return out
