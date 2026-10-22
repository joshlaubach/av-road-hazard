"""Country-aware evaluation on small hand-checked arrays; no model or image is involved."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from defects.evaluation import (
    Predictions,
    auroc,
    average_precision,
    build_report,
    class_metrics,
    country_of_id,
    country_prior_probs,
    evaluate_by_country,
    macro_over_countries,
    read_predictions,
    write_predictions,
)


def arr(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def predictions(
    ids: Sequence[str], labels: list[list[int]], probs: list[list[float]], **meta: object
) -> Predictions:
    return Predictions(ids=ids, labels=np.asarray(labels), probs=np.asarray(probs), meta=dict(meta))


# --- single-class metrics -------------------------------------------------------------------------


def test_auroc_and_average_precision_match_the_documented_example() -> None:
    y, p = arr([0, 0, 1, 1]), arr([0.1, 0.4, 0.35, 0.8])
    assert auroc(y, p) == pytest.approx(0.75)
    assert average_precision(y, p) == pytest.approx(5 / 6)


def test_perfect_reversed_and_tied_scores() -> None:
    y = arr([0, 0, 1, 1])
    assert auroc(y, arr([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, arr([0.9, 0.8, 0.2, 0.1])) == 0.0
    tied = arr([0.5, 0.5, 0.5, 0.5])
    assert auroc(y, tied) == pytest.approx(0.5)
    assert average_precision(y, tied) == pytest.approx(0.5)  # equals the prevalence


def test_auroc_needs_both_classes_and_average_precision_needs_a_positive() -> None:
    assert auroc(arr([0, 0, 0]), arr([0.1, 0.2, 0.3])) is None
    assert auroc(arr([1, 1]), arr([0.1, 0.2])) is None
    assert average_precision(arr([0, 0]), arr([0.1, 0.2])) is None
    assert average_precision(arr([1, 1]), arr([0.1, 0.2])) == 1.0


def test_precision_recall_f1_at_a_threshold() -> None:
    m = class_metrics(arr([1, 1, 0, 0]), arr([0.9, 0.4, 0.6, 0.1]), threshold=0.5)
    assert (m.n, m.positives, m.prevalence) == (4, 2, 0.5)
    assert (m.precision, m.recall, m.f1) == (0.5, 0.5, 0.5)
    assert m.notes == ()


def test_a_threshold_that_predicts_nothing_leaves_precision_undefined_not_zero() -> None:
    m = class_metrics(arr([1, 0]), arr([0.2, 0.1]), threshold=0.5)
    assert m.precision is None and "no predicted positives" in m.notes
    assert (m.recall, m.f1) == (0.0, 0.0)


def test_a_class_with_no_positives_reports_what_is_undefined_and_why() -> None:
    m = class_metrics(arr([0, 0, 0]), arr([0.9, 0.1, 0.2]), threshold=0.5)
    assert (m.positives, m.prevalence) == (0, 0.0)
    assert (m.auroc, m.average_precision, m.recall, m.f1) == (None, None, None, None)
    assert m.precision == 0.0  # one predicted positive, and it is wrong
    assert "no positive examples" in m.notes


def test_a_class_with_no_negatives_has_no_auroc() -> None:
    m = class_metrics(arr([1, 1]), arr([0.9, 0.8]), threshold=0.5)
    assert m.auroc is None and "no negative examples" in m.notes
    assert (m.average_precision, m.recall, m.f1) == (1.0, 1.0, 1.0)


# --- predictions container ---------------------------------------------------------------------


def test_country_comes_from_the_first_part_of_an_image_or_tile_id() -> None:
    assert country_of_id("Czech/train/annotations/xmls/Czech_000000") == "Czech"
    assert country_of_id("Norway/train/annotations/xmls/Norway_000001#r2c3") == "Norway"
    assert country_of_id("plain") == "ALL"


@pytest.mark.parametrize(
    ("ids", "labels", "probs", "message"),
    [
        (["a/1", "a/1"], [[1, 0, 0, 0]] * 2, [[0.5] * 4] * 2, "duplicate"),
        (["a/1"], [[1, 0, 0, 0]] * 2, [[0.5] * 4], "shape"),
        (["a/1"], [[1, 0, 0, 0]], [[0.5] * 3], "shape"),
        (["a/1"], [[2, 0, 0, 0]], [[0.5] * 4], "binary"),
        (["a/1"], [[1, 0, 0, 0]], [[1.5, 0.5, 0.5, 0.5]], "probabilit"),
        (["a/1"], [[1, 0, 0, 0]], [[float("nan"), 0.5, 0.5, 0.5]], "probabilit"),
        ([], np.zeros((0, 4)).tolist(), np.zeros((0, 4)).tolist(), "empty"),
    ],
)
def test_invalid_predictions_are_rejected(
    ids: list[str], labels: list[list[int]], probs: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        predictions(ids, labels, probs)


def test_predictions_round_trip_through_the_saved_artifact(tmp_path: Path) -> None:
    original = predictions(
        ["Czech/a#r0c0", "India/b#r1c1"],
        [[1, 0, 0, 1], [0, 0, 0, 0]],
        [[0.9, 0.1, 0.2, 0.7], [0.1, 0.2, 0.3, 0.4]],
        model="smoke",
        checkpoint_sha256="abc",
    )
    path = tmp_path / "predictions.npz"
    write_predictions(original, path)
    loaded = read_predictions(path)
    assert loaded.ids == original.ids
    assert loaded.countries == ("Czech", "India")
    assert loaded.meta == original.meta
    np.testing.assert_array_equal(loaded.labels, original.labels)
    np.testing.assert_allclose(loaded.probs, original.probs, atol=1e-6)


# --- by country -----------------------------------------------------------------------------------


def two_country_predictions() -> Predictions:
    ids = ["Czech/1", "Czech/2", "Japan/1", "Japan/2", "Japan/3"]
    labels = [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    probs = [[0.9, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1], [0.8, 0.1, 0.1, 0.1]]
    probs += [[0.7, 0.1, 0.1, 0.1], [0.75, 0.1, 0.1, 0.1]]
    return predictions(ids, labels, probs)


def test_metrics_are_reported_per_country_and_per_class_with_their_counts() -> None:
    result = evaluate_by_country(two_country_predictions(), threshold=0.5)
    assert set(result) == {"Czech", "Japan", "ALL"}
    assert result["Czech"]["D00"].n == 2 and result["Czech"]["D00"].positives == 1
    assert result["Japan"]["D00"].n == 3 and result["Japan"]["D00"].positives == 2
    assert result["ALL"]["D00"].n == 5 and result["ALL"]["D00"].positives == 3
    assert result["Czech"]["D00"].auroc == 1.0
    # Japan's one negative (0.75) outranks one of its two positives (0.7) and not the other (0.8).
    assert result["Japan"]["D00"].auroc == pytest.approx(0.5)
    assert result["ALL"]["D00"].auroc == pytest.approx(5 / 6)
    assert result["Czech"]["D10"].positives == 0 and result["Czech"]["D10"].auroc is None


def test_macro_over_countries_averages_only_defined_values_and_says_how_many() -> None:
    result = evaluate_by_country(two_country_predictions(), threshold=0.5)
    mean, used = macro_over_countries(result, "D00", "auroc")
    assert used == 2 and mean == pytest.approx(0.75)  # (1.0 + 0.5) / 2
    assert macro_over_countries(result, "D10", "auroc") == (None, 0)


# --- country prior baseline -----------------------------------------------------------------------


def test_country_prior_predicts_training_prevalence_and_pooled_for_unseen_countries() -> None:
    train_ids = ["Czech/1", "Czech/2", "Czech/3", "Czech/4", "Japan/1", "Japan/2", "Japan/3"]
    train_ids += ["Japan/4"]
    train_labels = np.zeros((8, 4), dtype=np.int64)
    train_labels[:2, 0] = 1  # Czech D00 = 2/4
    train_labels[4:, 0] = 1  # Japan D00 = 4/4
    train_labels[0, 3] = 1  # Czech D40 = 1/4
    probs = country_prior_probs(train_ids, train_labels, ["Czech/x", "Japan/y", "Norway/z"])
    np.testing.assert_allclose(probs[0], [0.5, 0, 0, 0.25])
    np.testing.assert_allclose(probs[1], [1.0, 0, 0, 0])
    np.testing.assert_allclose(probs[2], [6 / 8, 0, 0, 1 / 8])  # pooled, not a guess of zero


def test_country_prior_refuses_to_run_without_training_data() -> None:
    with pytest.raises(ValueError, match="training"):
        country_prior_probs([], np.zeros((0, 4)), ["Czech/x"])


# --- report ---------------------------------------------------------------------------------


def test_report_compares_model_and_baseline_and_is_valid_json() -> None:
    model = two_country_predictions()
    baseline = predictions(
        model.ids, model.labels.tolist(), [[0.5, 0.1, 0.1, 0.1]] * len(model.ids), model="prior"
    )
    report = build_report(model, baseline, threshold=0.5)
    json.dumps(report)  # None and floats only; no numpy scalars
    assert report["threshold"] == 0.5
    assert report["countries"]["Czech"]["D00"]["model"]["auroc"] == 1.0
    assert report["countries"]["Czech"]["D00"]["baseline"]["auroc"] == pytest.approx(0.5)
    assert report["macro"]["D00"]["model"]["auroc"]["countries_used"] == 2


def test_report_without_a_baseline_omits_it() -> None:
    report = build_report(two_country_predictions(), None, threshold=0.5)
    assert "baseline" not in report["countries"]["Czech"]["D00"]
