"""Pure helpers of tools/eval_detector.py: scene split, GT filtering, thresholds and timing."""

from __future__ import annotations

import pytest

from eval.models import Detection, GroundTruth
from tools.eval_detector import (
    GtBox,
    best_threshold,
    evaluable,
    is_night,
    latency_summary,
    split_scenes,
)

# ---------------------------------------------------------------- scene split


def test_split_is_by_sorted_token_with_the_last_fifth_held_out() -> None:
    tokens = [f"t{i}" for i in (9, 3, 7, 1, 5, 0, 2, 4, 6, 8)]
    train, val = split_scenes(tokens)
    assert val == ["t8", "t9"] and train == [f"t{i}" for i in range(8)]


def test_split_is_deterministic_and_disjoint() -> None:
    tokens = [f"s{i:02d}" for i in range(7)]
    assert split_scenes(tokens) == split_scenes(list(reversed(tokens)))
    train, val = split_scenes(tokens)
    assert not set(train) & set(val) and len(val) == 1  # round(7 * 0.2) = 1


def test_split_always_holds_something_out() -> None:
    train, val = split_scenes(["a", "b"], val_fraction=0.01)
    assert len(val) == 1 and len(train) == 1


# -------------------------------------------------------------------- night


@pytest.mark.parametrize(
    ("description", "night"),
    [
        ("Night, big street, bus stop", True),
        ("Arrive at busy intersection, night driving", True),
        ("Parked truck, construction, intersection", False),
        ("Knight rider", False),  # a whole word only
    ],
)
def test_night_is_a_whole_word_in_the_description(description: str, night: bool) -> None:
    assert is_night(description) is night


# --------------------------------------------------------- ground truth filter


def gt(visibility: int, side: float) -> GtBox:
    return GtBox("car", (0.0, 0.0, side, side), visibility)


def test_evaluable_needs_visibility_level_2_and_a_32px_box() -> None:
    assert evaluable(gt(2, 32.0))
    assert not evaluable(gt(1, 100.0))  # under 40% visible
    assert not evaluable(gt(4, 31.0))  # under 1024 px of area


# ---------------------------------------------------------------- threshold


def frame(scores: list[float], hit: list[bool]) -> tuple[list[Detection], list[GroundTruth]]:
    """One ground-truth car; each prediction is on it (hit) or far away (miss)."""
    preds = [
        Detection("car", s, (0.0, 0.0, 10.0, 10.0) if h else (500.0, 500.0, 510.0, 510.0))
        for s, h in zip(scores, hit, strict=True)
    ]
    return preds, [GroundTruth("car", (0.0, 0.0, 10.0, 10.0))]


def test_best_threshold_maximises_f1() -> None:
    # A true detection at 0.9 and two false ones at 0.6 and 0.3: only 0.7 or higher is clean.
    frames = [frame([0.9, 0.6, 0.3], [True, False, False])]
    threshold, score = best_threshold(frames, [0.3, 0.5, 0.7])
    assert threshold == 0.7 and score.f1 == pytest.approx(1.0)


def test_ties_go_to_the_higher_threshold() -> None:
    frames = [frame([0.95], [True])]  # every threshold up to 0.95 gives the same F1
    assert best_threshold(frames, [0.3, 0.5, 0.7])[0] == 0.7


def test_an_empty_grid_is_an_error() -> None:
    with pytest.raises(ValueError, match="threshold"):
        best_threshold([frame([0.9], [True])], [])


# ------------------------------------------------------------------ latency


def test_latency_summary_reports_median_and_p95() -> None:
    summary = latency_summary([float(i) for i in range(1, 101)])
    assert summary == {"n": 100, "p50_s": pytest.approx(50.5), "p95_s": pytest.approx(95.05)}


def test_latency_needs_at_least_one_sample() -> None:
    with pytest.raises(ValueError, match="sample"):
        latency_summary([])
