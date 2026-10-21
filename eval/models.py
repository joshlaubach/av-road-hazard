"""Per-model metrics: detection matching, precision/recall/F1, AP, and classifier scores.

Pure functions on plain arrays, no model or dataset imports. Definitions are recorded in
docs/EVALUATION.md. A prediction matches a ground-truth object when the labels are equal and the
box IoU is at least iou_thr; predictions are matched greedily in descending score order.

AP is a reimplementation (COCO-style 101-point interpolation), not torchmetrics: compare it with the
source baseline (AP@0.50 0.312) only after checking both on the same predictions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2), pixels
Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Detection:
    """One predicted box: class label, confidence score, and pixel box."""

    label: str
    score: float
    box: Box


@dataclass(frozen=True)
class GroundTruth:
    """One labeled real object: class label and pixel box."""

    label: str
    box: Box


# One image: (predictions, ground truth).
Frame = tuple[Sequence[Detection], Sequence[GroundTruth]]


@dataclass(frozen=True)
class Matches:
    """Outcome of matching one image: true positives as (prediction, ground truth) index pairs,
    false positives (predictions with no match), false negatives (missed ground truths)."""

    tp: list[tuple[int, int]]  # (prediction index, ground-truth index)
    fp: list[int]  # prediction indices
    fn: list[int]  # ground-truth indices


@dataclass(frozen=True)
class PRF:
    """Precision, recall and their harmonic mean (F1)."""

    precision: float
    recall: float
    f1: float


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes: overlap area divided by combined area (0 to 1).

    1.0 is identical boxes, 0.0 is no overlap. The max(..., 0) clamps handle disjoint boxes, whose
    raw intersection width or height would be negative.
    """
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    preds: Sequence[Detection], gts: Sequence[GroundTruth], iou_thr: float = 0.5
) -> Matches:
    """Greedy matching: highest score first, each takes the best unmatched same-label object."""
    # Indices of ground truths already claimed, so one real object matches at most one prediction.
    taken: set[int] = set()
    tp: list[tuple[int, int]] = []
    fp: list[int] = []
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i].score):
        # best_iou starts at the threshold, so only a box overlapping at least that much can win.
        best, best_iou = -1, iou_thr
        for gi, gt in enumerate(gts):
            if gi in taken or gt.label != preds[pi].label:
                continue
            overlap = iou(preds[pi].box, gt.box)
            if overlap >= best_iou:
                best, best_iou = gi, overlap
        if best >= 0:  # found an unclaimed same-label object that overlaps enough
            taken.add(best)
            tp.append((pi, best))
        else:  # nothing to claim: a duplicate, a wrong class, or a box on empty road
            fp.append(pi)
    return Matches(tp, fp, [gi for gi in range(len(gts)) if gi not in taken])


def prf(tp: int, fp: int, fn: int) -> PRF:
    """Precision, recall, F1; a score with an empty denominator is 0.0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return PRF(p, r, 2 * p * r / (p + r) if p + r else 0.0)


def detection_prf(frames: Sequence[Frame], iou_thr: float = 0.5, score_thr: float = 0.0) -> PRF:
    """Micro-averaged over all frames, counting only predictions with score >= score_thr."""
    # Micro-average: add up counts over all frames first, then compute the ratios once.
    tp = fp = fn = 0
    for preds, gts in frames:
        kept = [p for p in preds if p.score >= score_thr]
        m = match_detections(kept, gts, iou_thr)
        tp, fp, fn = tp + len(m.tp), fp + len(m.fp), fn + len(m.fn)
    return prf(tp, fp, fn)


def average_precision(frames: Sequence[Frame], iou_thr: float = 0.5) -> float:
    """Mean over classes (with at least one ground truth) of 101-point interpolated AP."""
    labels = {gt.label for _, gts in frames for gt in gts}
    aps = [_class_ap(frames, label, iou_thr) for label in sorted(labels)]
    return float(np.mean(aps)) if aps else math.nan


def _class_ap(frames: Sequence[Frame], label: str, iou_thr: float) -> float:
    """AP for one class: how precision holds up as you accept predictions from most to least sure.

    Steps: (1) match per image and record, for every prediction, its score and whether it was a
    true positive; (2) sort all predictions by score; (3) walk down the list building the
    precision-recall curve; (4) average the precision at 101 evenly spaced recall levels.
    """
    scored: list[tuple[float, bool]] = []
    n_gt = 0
    for preds, gts in frames:
        p = [d for d in preds if d.label == label]
        g = [t for t in gts if t.label == label]
        n_gt += len(g)
        m = match_detections(p, g, iou_thr)
        matched = {pi for pi, _ in m.tp}
        scored += [(d.score, i in matched) for i, d in enumerate(p)]
    scored.sort(key=lambda s: -s[0])  # most confident first
    hits = np.array([s[1] for s in scored], dtype=np.float64)
    # After accepting the top k predictions: tp_cum true positives and fp_cum false positives.
    tp_cum = np.cumsum(hits)
    fp_cum = np.cumsum(1.0 - hits)
    recall = tp_cum / n_gt if len(hits) else np.array([], dtype=np.float64)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # Precision envelope: best precision at this recall or higher.
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(hits) else precision
    # For each recall level t, take the precision at the first point that reaches t. Levels the
    # curve never reaches count as 0, which is how missed objects pull AP down.
    total = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        idx = int(np.searchsorted(recall, t, side="left"))
        total += float(precision[idx]) if idx < len(precision) else 0.0
    return total / 101


def recall_by_size(
    frames: Sequence[Frame], area_edges: Sequence[float], iou_thr: float = 0.5
) -> list[tuple[float, float, int, float]]:
    """Recall per ground-truth box-area bin: (low, high, n_objects, recall).

    area_edges are ascending pixel-area cutoffs, e.g. (1024, 9216) gives three bins. There are
    deliberately no default cutoffs: they depend on the data.
    """
    edges = [0.0, *area_edges, math.inf]  # n cutoffs make n + 1 bins, the last open-ended
    found = [0] * (len(edges) - 1)
    total = [0] * (len(edges) - 1)
    for preds, gts in frames:
        missed = set(match_detections(preds, gts, iou_thr).fn)  # ground truths nobody matched
        for gi, gt in enumerate(gts):
            area = (gt.box[2] - gt.box[0]) * (gt.box[3] - gt.box[1])
            # searchsorted finds where the area falls among the edges; minus 1 gives the bin index.
            b = int(np.searchsorted(edges, area, side="right")) - 1
            total[b] += 1
            found[b] += gi not in missed
    return [
        (edges[i], edges[i + 1], total[i], found[i] / total[i] if total[i] else math.nan)
        for i in range(len(total))
    ]


def per_class_prf(y_true: Array, y_prob: Array, threshold: float | Array) -> list[PRF]:
    """Multi-label scores per class (columns). y_true is 0/1, y_prob is a probability."""
    pred = y_prob >= threshold  # flagged where the probability reaches the threshold
    truth = y_true > 0.5
    out: list[PRF] = []
    for k in range(truth.shape[1]):  # one class per column; classes are scored independently
        tp = int(np.sum(pred[:, k] & truth[:, k]))
        fp = int(np.sum(pred[:, k] & ~truth[:, k]))
        fn = int(np.sum(~pred[:, k] & truth[:, k]))
        out.append(prf(tp, fp, fn))
    return out


def macro_f1(scores: Sequence[PRF]) -> float:
    """Unweighted mean of per-class F1, so rare classes count as much as common ones."""
    return float(np.mean([s.f1 for s in scores]))


def auroc(y_true: Array, scores: Array) -> float:
    """Area under the ROC curve from ranks; ties get average ranks. NaN if a class is absent."""
    # AUROC equals the chance a random positive outranks a random negative. Computed from ranks
    # (the Mann-Whitney U statistic), so no thresholds and no extra library are needed.
    pos = y_true > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:  # AUROC is undefined without both kinds of example
        return math.nan
    # Tied scores share the average of the ranks they span, which counts a tie as half a win.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ((ends - counts + 1 + ends) / 2.0)[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auroc(y_true: Array, y_prob: Array) -> float:
    """Mean AUROC over classes that have both positives and negatives."""
    values = [auroc(y_true[:, k], y_prob[:, k]) for k in range(y_true.shape[1])]
    valid = [v for v in values if not math.isnan(v)]
    return float(np.mean(valid)) if valid else math.nan


def recall_at_fpr(pos_scores: Array, neg_scores: Array, max_fpr: float) -> tuple[float, float]:
    """Highest recall whose false-positive rate on negatives is at most max_fpr.

    Returns (recall, threshold). A score at or above the threshold is flagged. Use it with
    non-road tiles as the negatives to pair the recall target with the false-positive guardrail.
    """
    if len(neg_scores) == 0:
        raise ValueError("need at least one negative score")
    # How many negatives may be flagged at this false-positive rate, e.g. 5% of 400 is 20.
    allowed = math.floor(max_fpr * len(neg_scores) + 1e-9)  # tolerance: 0.29 * 100 is 28.999...
    ordered = np.sort(neg_scores)[::-1]  # highest-scoring negatives first
    # Set the threshold just above the first negative that must NOT be flagged (nextafter is the
    # next representable float), so exactly the top `allowed` negatives stay at or above it.
    threshold = (
        float(np.nextafter(ordered[allowed], np.inf)) if allowed < len(ordered) else -math.inf
    )
    recall = float(np.mean(pos_scores >= threshold)) if len(pos_scores) else math.nan
    return recall, threshold
