"""Classifier evaluation helpers: flagging at per-class thresholds and the three report parts."""

import numpy as np
import pytest

from eval.classifier import (
    Alert,
    alert_buckets,
    check_calls,
    class_report,
    flagged_share,
    flags,
    recall_at_baseline_fpr,
    recall_guard,
    summarize_alerts,
    wilson_interval,
)

CODES = ("D00", "D10")
THRESHOLDS = (0.9, 0.5)


def test_flags_apply_each_class_threshold() -> None:
    probs = np.array([[0.95, 0.1], [0.8, 0.6], [0.1, 0.1]])

    assert flags(probs, THRESHOLDS).tolist() == [[True, False], [False, True], [False, False]]


def test_flags_rejects_wrong_threshold_count() -> None:
    with pytest.raises(ValueError, match="one threshold per class"):
        flags(np.zeros((2, 2)), (0.5,))


def test_class_report_counts_precision_and_recall() -> None:
    truth = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    probs = np.array([[0.95, 0.1], [0.2, 0.1], [0.1, 0.7], [0.99, 0.6]])

    report = class_report(truth, probs, THRESHOLDS, CODES)

    assert report["D00"]["flagged"] == 2
    assert report["D00"]["positives"] == 2
    assert report["D00"]["precision"] == pytest.approx(0.5)
    assert report["D00"]["recall"] == pytest.approx(0.5)
    assert report["D10"]["precision"] == pytest.approx(0.5)
    assert report["D10"]["recall"] == pytest.approx(1.0)
    assert 0.0 <= report["D10"]["auroc"] <= 1.0


def test_class_report_precision_is_zero_when_nothing_is_flagged() -> None:
    truth = np.array([[1, 0], [0, 0]])
    probs = np.array([[0.1, 0.1], [0.1, 0.1]])

    assert class_report(truth, probs, THRESHOLDS, CODES)["D00"]["precision"] == 0.0


def test_flagged_share_counts_tiles_flagged_for_any_class() -> None:
    probs = np.array([[0.95, 0.1], [0.1, 0.6], [0.1, 0.1], [0.2, 0.2]])

    assert flagged_share(probs, THRESHOLDS) == {"tiles": 4, "flagged": 2}


# --- Wilson interval, matched false-alarm recall, alert buckets and the recall guard -----------


def test_wilson_interval_matches_a_hand_computed_case() -> None:
    low, high = wilson_interval(3, 23)  # 90%

    assert low == pytest.approx(0.053, abs=0.002)
    assert high == pytest.approx(0.285, abs=0.002)


def test_wilson_interval_is_not_degenerate_at_zero() -> None:
    low, high = wilson_interval(0, 23)

    assert low == 0.0
    assert high == pytest.approx(0.105, abs=0.003)


def test_wilson_interval_stays_inside_zero_and_one() -> None:
    assert wilson_interval(23, 23)[1] == 1.0
    assert wilson_interval(5, 5)[0] > 0.0


def test_wilson_interval_with_no_samples_says_nothing() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="between 0 and n"):
        wilson_interval(4, 3)


def test_recall_at_baseline_false_alarm_rate_uses_the_baseline_threshold() -> None:
    truth = np.array([[1], [1], [0], [0], [0], [0]])
    base = np.array([[0.9], [0.4], [0.95], [0.1], [0.1], [0.1]])  # 1 of 4 negatives flagged at .5
    new = np.array([[0.9], [0.8], [0.6], [0.1], [0.1], [0.1]])

    got = recall_at_baseline_fpr(truth, base, new, (0.5,), ("D00",))

    assert got["D00"]["baseline_fpr"] == pytest.approx(0.25)
    assert got["D00"]["recall"] == pytest.approx(1.0)  # new keeps both positives at 25% FPR


def test_recall_at_baseline_false_alarm_rate_needs_negatives() -> None:
    truth = np.array([[1], [1]])
    probs = np.array([[0.9], [0.8]])

    got = recall_at_baseline_fpr(truth, probs, probs, (0.5,), ("D00",))

    assert np.isnan(got["D00"]["recall"])


def test_alert_buckets_split_alerts_by_review_status() -> None:
    alerts = [
        ("f1", (0, 0, 4, 4)),
        ("f1", (4, 0, 8, 4)),
        ("f2", (0, 0, 4, 4)),
        ("f3", (0, 0, 4, 4)),
    ]
    reviewed = {
        ("f1", (0, 0, 4, 4)): "Crack",
        ("f1", (4, 0, 8, 4)): "No crack",
        ("f2", (0, 0, 4, 4)): "Ambiguous",
    }

    assert alert_buckets(alerts, reviewed) == {
        "Crack": 1,
        "No crack": 1,
        "Ambiguous": 1,
        "Unreviewed": 1,
    }


def test_alert_buckets_count_a_tile_once_even_if_it_alerts_for_two_classes() -> None:
    alerts = [("f1", (0, 0, 4, 4)), ("f1", (0, 0, 4, 4))]

    assert alert_buckets(alerts, {("f1", (0, 0, 4, 4)): "Crack"})["Crack"] == 1


def test_recall_guard_passes_when_nothing_is_lost() -> None:
    base = {"D00": 0.8, "D10": 0.7}
    new = {"D00": 0.79, "D10": 0.72}

    verdict = recall_guard(base, new, cracks_kept=6, cracks_total=6)

    assert verdict["passed"] is True
    assert verdict["reasons"] == []


def test_recall_guard_fails_on_a_validation_recall_drop() -> None:
    verdict = recall_guard({"D00": 0.8}, {"D00": 0.7}, cracks_kept=6, cracks_total=6)

    assert verdict["passed"] is False
    assert "D00" in verdict["reasons"][0]


def test_recall_guard_fails_when_reviewed_cracks_are_lost() -> None:
    verdict = recall_guard({"D00": 0.8}, {"D00": 0.8}, cracks_kept=4, cracks_total=6)

    assert verdict["passed"] is False
    assert "4 of 6" in verdict["reasons"][0]


def test_recall_guard_limits_are_adjustable() -> None:
    verdict = recall_guard(
        {"D00": 0.8}, {"D00": 0.7}, cracks_kept=6, cracks_total=6, max_recall_drop=0.15
    )

    assert verdict["passed"] is True


# --- Alerts per frame ---------------------------------------------------------------------------


def _alert(**kw: object) -> Alert:
    base = {
        "frame": "f1", "box": (0, 0, 4, 4), "name": "D00", "confidence": 0.9,
        "night": False, "road_cover": 0.9,
    }  # fmt: skip
    return Alert(**{**base, **kw})  # type: ignore[arg-type]


def test_summary_counts_alerts_per_frame_overall_and_by_day_and_night() -> None:
    alerts = [_alert(), _alert(frame="f2", night=True), _alert(frame="f2", night=True, name="D10")]

    got = summarize_alerts(alerts, frames=4, night_frames=1, reviewed={})

    assert got["alerts"] == 3
    assert got["per_frame"] == pytest.approx(0.75)
    assert got["day"] == {"frames": 3, "alerts": 1, "per_frame": pytest.approx(1 / 3)}
    assert got["night"] == {"frames": 1, "alerts": 2, "per_frame": pytest.approx(2.0)}


def test_summary_counts_alerts_by_defect_type_and_review_bucket() -> None:
    alerts = [_alert(), _alert(name="D10", box=(4, 0, 8, 4)), _alert(name="D10", frame="f2")]
    reviewed = {("f1", (0, 0, 4, 4)): "Crack"}

    got = summarize_alerts(alerts, frames=2, night_frames=0, reviewed=reviewed)

    assert got["by_class"] == {"D00": 1, "D10": 2}
    assert got["buckets"] == {"Crack": 1, "No crack": 0, "Ambiguous": 0, "Unreviewed": 2}


def test_summary_bands_alerts_by_road_cover() -> None:
    alerts = [_alert(road_cover=0.45), _alert(road_cover=0.65), _alert(road_cover=0.95)]

    got = summarize_alerts(alerts, frames=3, night_frames=0, reviewed={})

    assert got["by_road_cover"] == {"under 0.4": 0, "0.4-0.6": 1, "0.6-0.8": 1, "0.8-1.0": 1}


def test_summary_without_frames_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        summarize_alerts([], frames=0, night_frames=0, reviewed={})


def test_summary_rejects_more_night_frames_than_frames() -> None:
    with pytest.raises(ValueError, match="night_frames"):
        summarize_alerts([], frames=2, night_frames=3, reviewed={})


# --- Review findings: guard edge cases and the under-cover band ---------------------------------


def test_recall_guard_allows_a_drop_of_exactly_the_limit() -> None:
    verdict = recall_guard({"D00": 0.80}, {"D00": 0.77}, cracks_kept=6, cracks_total=6)

    assert verdict["passed"] is True  # 0.80 - 0.77 is 0.03 up to float error


def test_recall_guard_flags_a_class_whose_recall_is_unknown() -> None:
    verdict = recall_guard({"D00": float("nan")}, {"D00": 0.5}, cracks_kept=6, cracks_total=6)

    assert verdict["passed"] is False
    assert "D00" in verdict["reasons"][0]
    assert "not checked" in verdict["reasons"][0]


def test_recall_guard_without_a_review_checks_only_validation_recall() -> None:
    verdict = recall_guard({"D00": 0.8}, {"D00": 0.8}, cracks_kept=None, cracks_total=None)

    assert verdict == {"passed": True, "reasons": []}


def test_recall_guard_does_not_ask_more_of_a_candidate_than_the_baseline_kept() -> None:
    verdict = recall_guard(
        {"D00": 0.8},
        {"D00": 0.8},
        cracks_kept=4,
        cracks_total=6,
        baseline_cracks_kept=4,
    )

    assert verdict["passed"] is True


def test_recall_guard_still_fails_a_candidate_below_the_baseline_crack_count() -> None:
    verdict = recall_guard(
        {"D00": 0.8}, {"D00": 0.8}, cracks_kept=3, cracks_total=6, baseline_cracks_kept=4
    )

    assert verdict["passed"] is False


def test_summary_puts_alerts_under_the_road_cut_off_in_their_own_band() -> None:
    alerts = [_alert(road_cover=0.3), _alert(road_cover=0.4), _alert(road_cover=1.0)]

    got = summarize_alerts(alerts, frames=3, night_frames=0, reviewed={})

    assert got["by_road_cover"]["under 0.4"] == 1
    assert got["by_road_cover"]["0.4-0.6"] == 1
    assert got["by_road_cover"]["0.8-1.0"] == 1
    assert sum(got["by_road_cover"].values()) == 3


def test_matched_fpr_recall_is_lower_for_a_worse_model() -> None:
    truth = np.array([[1], [1], [0], [0], [0], [0]])
    base = np.array([[0.9], [0.8], [0.6], [0.1], [0.1], [0.1]])  # 1 of 4 negatives flagged
    worse = np.array([[0.9], [0.05], [0.8], [0.7], [0.1], [0.1]])  # two negatives above a positive

    got = recall_at_baseline_fpr(truth, base, worse, (0.5,), ("D00",))

    assert got["D00"]["recall"] == pytest.approx(0.5)


# --- Review calls --------------------------------------------------------------------------------


def test_check_calls_returns_the_calls_for_every_tile() -> None:
    got = check_calls(["a", "b"], {"a": "Crack", "b": "No crack", "extra": "Ambiguous"})

    assert got == {"a": "Crack", "b": "No crack"}


def test_check_calls_rejects_a_tile_with_no_call() -> None:
    with pytest.raises(ValueError, match="no call for tile b"):
        check_calls(["a", "b"], {"a": "Crack"})


@pytest.mark.parametrize("call", ["", "Unsure", "crack"])
def test_check_calls_rejects_a_blank_or_unknown_call(call: str) -> None:
    with pytest.raises(ValueError, match="unknown call"):
        check_calls(["a"], {"a": call})
