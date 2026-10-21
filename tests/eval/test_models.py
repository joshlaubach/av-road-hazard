from __future__ import annotations

import math

import numpy as np
import pytest

from eval.models import (
    Detection,
    Frame,
    GroundTruth,
    auroc,
    average_precision,
    detection_prf,
    iou,
    macro_auroc,
    macro_f1,
    match_detections,
    per_class_prf,
    prf,
    recall_at_fpr,
    recall_by_size,
)

BOX = (0.0, 0.0, 10.0, 10.0)


def test_iou_partial_overlap_and_disjoint() -> None:
    assert iou(BOX, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1 / 3)
    assert iou(BOX, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert iou(BOX, BOX) == 1.0


def test_match_takes_best_score_first_and_marks_duplicates_false_positive() -> None:
    gts = [GroundTruth("car", BOX)]
    preds = [Detection("car", 0.6, BOX), Detection("car", 0.9, BOX)]
    m = match_detections(preds, gts)
    assert m.tp == [(1, 0)]  # the 0.9 prediction wins
    assert m.fp == [0]
    assert m.fn == []


def test_match_requires_same_label_and_min_iou() -> None:
    gts = [GroundTruth("car", BOX)]
    wrong_label = match_detections([Detection("truck", 0.9, BOX)], gts)
    assert (wrong_label.tp, wrong_label.fp, wrong_label.fn) == ([], [0], [0])
    low_iou = match_detections([Detection("car", 0.9, (6.0, 0.0, 16.0, 10.0))], gts)
    assert low_iou.tp == []  # IoU 0.25 < 0.5


def test_prf_values_and_empty_denominators() -> None:
    s = prf(tp=1, fp=1, fn=0)
    assert (s.precision, s.recall) == (0.5, 1.0)
    assert s.f1 == pytest.approx(2 / 3)
    assert prf(0, 0, 0).f1 == 0.0


def test_detection_prf_applies_score_threshold() -> None:
    frames: list[Frame] = [
        (
            [Detection("car", 0.9, BOX), Detection("car", 0.2, (50.0, 50.0, 60.0, 60.0))],
            [GroundTruth("car", BOX)],
        )
    ]
    assert detection_prf(frames).precision == 0.5
    assert detection_prf(frames, score_thr=0.5).precision == 1.0


def test_average_precision_perfect_and_one_miss() -> None:
    perfect: list[Frame] = [([Detection("car", 0.9, BOX)], [GroundTruth("car", BOX)])]
    assert average_precision(perfect) == pytest.approx(1.0)
    other = (30.0, 30.0, 40.0, 40.0)
    mixed: list[Frame] = [
        (
            [Detection("car", 0.9, BOX), Detection("car", 0.8, (60.0, 60.0, 70.0, 70.0))],
            [GroundTruth("car", BOX), GroundTruth("car", other)],
        )
    ]
    # Recall reaches 0.5 at precision 1.0, never higher: 51 of 101 recall points score 1.0.
    assert average_precision(mixed) == pytest.approx(51 / 101)


def test_average_precision_without_ground_truth_is_nan() -> None:
    assert math.isnan(average_precision([([Detection("car", 0.9, BOX)], [])]))


def test_recall_by_size_bins_by_ground_truth_area() -> None:
    small, large = (0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 100.0, 100.0)
    frames: list[Frame] = [
        ([Detection("car", 0.9, large)], [GroundTruth("car", small), GroundTruth("car", large)])
    ]
    # The 'small' ground truth sits in the same place but is far smaller than the large prediction.
    bins = recall_by_size(frames, area_edges=(1000.0,))
    assert bins[0][:3] == (0.0, 1000.0, 1)
    assert bins[0][3] == 0.0
    assert bins[1][:3] == (1000.0, math.inf, 1)
    assert bins[1][3] == 1.0


def test_per_class_prf_and_macro_f1() -> None:
    y_true = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)
    y_prob = np.array([[0.9, 0.2], [0.8, 0.6], [0.7, 0.1], [0.1, 0.1]])
    scores = per_class_prf(y_true, y_prob, threshold=0.5)
    assert (scores[0].precision, scores[0].recall) == (2 / 3, 1.0)
    assert (scores[1].precision, scores[1].recall) == (1.0, 0.5)
    assert macro_f1(scores) == pytest.approx((0.8 + 2 / 3) / 2)


def test_auroc_known_value_ties_and_missing_class() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    assert auroc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert math.isnan(auroc(np.ones(3), np.array([0.1, 0.2, 0.3])))


def test_macro_auroc_skips_classes_without_both_labels() -> None:
    y = np.array([[0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.float64)
    p = np.array([[0.1, 0.5], [0.4, 0.5], [0.35, 0.5], [0.8, 0.5]])
    assert macro_auroc(y, p) == pytest.approx(0.75)  # column 1 has no negatives


def test_recall_at_fpr_cap() -> None:
    neg = np.array([0.1, 0.2, 0.3, 0.4])
    pos = np.array([0.35, 0.5, 0.6])
    recall, threshold = recall_at_fpr(pos, neg, max_fpr=0.25)
    assert recall == 1.0
    assert np.mean(neg >= threshold) <= 0.25
    recall0, threshold0 = recall_at_fpr(pos, neg, max_fpr=0.0)
    assert recall0 == pytest.approx(2 / 3)
    assert np.mean(neg >= threshold0) == 0.0


def test_recall_at_fpr_needs_negatives() -> None:
    with pytest.raises(ValueError, match="negative"):
        recall_at_fpr(np.array([0.5]), np.array([]), max_fpr=0.1)


def test_recall_at_fpr_is_not_fooled_by_float_error() -> None:
    neg = np.arange(100, dtype=np.float64)  # 100 distinct negatives
    pos = np.array([1000.0])
    _, threshold = recall_at_fpr(pos, neg, max_fpr=0.29)  # 0.29 * 100 == 28.999999999999996
    assert np.sum(neg >= threshold) == 29
