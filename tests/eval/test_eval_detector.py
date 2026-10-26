"""eval/detector.py helpers: scene split, GT filtering, thresholds, sweeps and timing."""

from __future__ import annotations

import pytest

from eval.detector import (
    FrameData,
    GtBox,
    best_threshold,
    evaluable,
    is_night,
    latency_summary,
    score_model,
    split_scenes,
    threshold_sweep,
)
from eval.models import Detection, GroundTruth

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


# -------------------------------------------------------------------- sweep


def data_frame() -> FrameData:
    """Two cars and a person. Detections: car 0.9, car 0.6, person 0.3 (hits), car 0.4 (false)."""
    box = (0.0, 0.0, 100.0, 100.0)
    other = (200.0, 0.0, 300.0, 100.0)
    far = (500.0, 500.0, 600.0, 600.0)
    return FrameData(
        "s",
        False,
        [
            Detection("car", 0.9, box),
            Detection("car", 0.6, other),
            Detection("car", 0.4, far),
            Detection("pedestrian", 0.3, (400.0, 0.0, 500.0, 100.0)),
        ],
        [
            GtBox("car", box, 4),
            GtBox("car", other, 4),
            GtBox("pedestrian", (400.0, 0.0, 500.0, 100.0), 4),
        ],
    )


def test_sweep_gives_one_row_per_threshold_with_precision_and_recall() -> None:
    rows = threshold_sweep([data_frame()], [0.3, 0.5, 0.7], only_evaluable=False)
    assert [r["threshold"] for r in rows] == [0.3, 0.5, 0.7]
    # 0.3 keeps all four (3 true, 1 false); 0.5 keeps 0.9 and 0.6; 0.7 keeps only 0.9.
    assert rows[0]["precision"] == pytest.approx(0.75)
    assert rows[0]["recall"] == pytest.approx(1.0)
    assert rows[1]["precision"] == pytest.approx(1.0)
    assert rows[1]["recall"] == pytest.approx(2 / 3)
    assert rows[2]["precision"] == pytest.approx(1.0)
    assert rows[2]["recall"] == pytest.approx(1 / 3)


def test_recall_never_rises_as_the_threshold_rises() -> None:
    rows = threshold_sweep([data_frame()], [0.1, 0.3, 0.5, 0.7, 0.9], only_evaluable=False)
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)


def test_sweep_can_be_limited_to_one_class() -> None:
    rows = threshold_sweep([data_frame()], [0.2, 0.5], only_evaluable=False, label="pedestrian")
    # The only person detection scores 0.3: found at 0.2, gone at 0.5.
    assert rows[0]["recall"] == pytest.approx(1.0) and rows[1]["recall"] == 0.0


def test_sweep_can_skip_ground_truth_that_is_not_evaluable() -> None:
    frame = FrameData(
        "s",
        False,
        [],
        [GtBox("car", (0.0, 0.0, 100.0, 100.0), 1)],  # under 40% visible
    )
    only = threshold_sweep([frame], [0.5], only_evaluable=True)
    everything = threshold_sweep([frame], [0.5], only_evaluable=False)
    assert only[0]["recall"] == 0.0 and only[0]["n_ground_truth"] == 0
    assert everything[0]["n_ground_truth"] == 1


# ------------------------------------------------------------------ score_model


def test_score_model_uses_the_labels_it_is_given_and_holds_out_other_scenes() -> None:
    train = FrameData("a", False, [], [])
    held_out = data_frame()
    report = score_model([train, held_out], {"a"}, ["car", "pedestrian"])
    assert set(report["sweep_by_class_evaluable"]) == {"car", "pedestrian"}
    assert report["held_out_all"]["frames"] == 1 and report["train_scenes_evaluable"]["frames"] == 1


def test_the_evaluation_module_does_not_import_perception() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import eval.detector; "
        "assert not [m for m in sys.modules if m == 'perception' or m.startswith('perception.')]"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(root)], check=True)


# ------------------------------------------------------------- speed verdicts


def test_speed_verdict_marks_each_rung_against_the_p95_limit() -> None:
    from eval.detector import speed_verdict

    rungs = [
        {"threads": 24, "n": 97, "p50_s": 0.9, "p95_s": 1.2},
        {"threads": 4, "n": 97, "p50_s": 1.8, "p95_s": 2.65},
    ]

    got = speed_verdict(rungs, limit_s=2.0)

    assert [r["passed"] for r in got] == [True, False]
    assert got[1]["threads"] == 4  # the measurements are kept


def test_speed_verdict_does_not_change_its_input() -> None:
    from eval.detector import speed_verdict

    rungs = [{"threads": 2, "n": 1, "p50_s": 1.0, "p95_s": 1.5}]

    speed_verdict(rungs, limit_s=2.0)

    assert "passed" not in rungs[0]


def test_speed_verdict_limit_is_inclusive() -> None:
    from eval.detector import speed_verdict

    assert speed_verdict([{"threads": 1, "n": 1, "p50_s": 1.0, "p95_s": 2.0}], limit_s=2.0)[0][
        "passed"
    ]
