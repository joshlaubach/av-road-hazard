"""The synthetic fusion evaluation: placements, pasting, per-trial scoring and the summary."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from eval.fusion_synthetic import (
    Reported,
    Trial,
    box_share,
    grow,
    paste_patch,
    polygon_mask,
    sample_box,
    score_trial,
    shrink,
    summarize,
    verdict,
)

SIZE = (200, 100)  # (width, height)
ROAD = [(0, 40), (200, 40), (200, 100), (0, 100)]  # the lower 60% of the frame


def test_polygon_mask_covers_the_polygon_and_nothing_else() -> None:
    mask = polygon_mask(ROAD, SIZE)
    assert mask.shape == (100, 200)  # numpy order: (height, width)
    assert mask[70, 100] and not mask[10, 100]
    assert mask.sum() == pytest.approx(200 * 60, rel=0.05)


def test_box_share_is_the_fraction_of_the_box_on_the_mask() -> None:
    mask = polygon_mask(ROAD, SIZE)
    assert box_share((10, 50, 30, 70), mask) == 1.0
    assert box_share((10, 10, 30, 30), mask) == 0.0
    assert box_share((10, 30, 30, 50), mask) == pytest.approx(0.5, abs=0.06)


def test_box_share_of_a_box_outside_the_image_is_zero() -> None:
    assert box_share((300, 300, 320, 320), polygon_mask(ROAD, SIZE)) == 0.0


def test_shrink_and_grow_move_the_edge_by_the_margin() -> None:
    mask = polygon_mask(ROAD, SIZE)
    assert not shrink(mask, 10)[48, 100] and shrink(mask, 10)[60, 100]
    assert grow(mask, 10)[32, 100] and not grow(mask, 10)[20, 100]


def test_shrink_does_not_eat_the_road_at_the_image_border() -> None:
    assert shrink(polygon_mask(ROAD, SIZE), 10)[99, 5]


def test_sampled_road_box_sits_well_inside_the_road_and_the_off_road_box_well_outside() -> None:
    mask = polygon_mask(ROAD, SIZE)
    rng = random.Random(1)
    on = sample_box(mask, (20, 20), on_road=True, blocked=[], margin=8, min_y=0.0, rng=rng)
    off = sample_box(mask, (20, 10), on_road=False, blocked=[], margin=8, min_y=0.0, rng=rng)
    assert on is not None and off is not None
    assert box_share(on, shrink(mask, 8)) == 1.0
    assert box_share(off, grow(mask, 8)) == 0.0


def test_sampling_avoids_blocked_boxes_and_the_minimum_height() -> None:
    mask = polygon_mask(ROAD, SIZE)
    blocked = [(0.0, 40.0, 100.0, 100.0)]  # the left half of the road
    for seed in range(20):
        box = sample_box(
            mask, (20, 20), on_road=True, blocked=blocked, margin=5, min_y=60.0,
            rng=random.Random(seed),
        )  # fmt: skip
        assert box is not None and box[0] >= 100.0 and box[1] >= 60.0


def test_sampling_gives_none_when_there_is_no_room() -> None:
    mask = polygon_mask(ROAD, SIZE)
    assert sample_box(mask, (500, 500), on_road=True, blocked=[], margin=5, min_y=0.0,
                      rng=random.Random(0)) is None  # fmt: skip


def test_the_same_seed_gives_the_same_box() -> None:
    mask = polygon_mask(ROAD, SIZE)
    args = {"on_road": True, "blocked": [], "margin": 5, "min_y": 0.0}
    a = sample_box(mask, (20, 20), rng=random.Random(5), **args)  # type: ignore[arg-type]
    b = sample_box(mask, (20, 20), rng=random.Random(5), **args)  # type: ignore[arg-type]
    assert a == b


def test_paste_changes_only_the_box_and_leaves_the_input_alone() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    patch = np.full((8, 8, 3), 255, dtype=np.uint8)
    out = paste_patch(frame, patch, (50, 50, 90, 70), feather=0.0)
    assert frame.sum() == 0  # the input is not modified
    assert out[60, 70].tolist() == [255, 255, 255]
    assert out[10, 10].tolist() == [0, 0, 0]
    assert out[:50].sum() == 0 and out[71:].sum() == 0


def test_paste_feathers_the_edge() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    patch = np.full((8, 8, 3), 255, dtype=np.uint8)
    out = paste_patch(frame, patch, (50, 50, 90, 70), feather=0.25)
    assert 0 < out[60, 51, 0] < 255 and out[60, 70, 0] == 255


# --------------------------------------------------------------------- per-trial scoring

TRUTH = (100.0, 60.0, 120.0, 80.0)


def _near(box: tuple[float, float, float, float]) -> set[str]:
    return {"pedestrian"}


def _score(reported: list[Reported], on_road: bool = True) -> Trial:
    mask = polygon_mask(ROAD, SIZE)
    return score_trial(
        TRUTH, on_road, reported, road_truth=mask, road_threshold=0.5, expected_near=_near
    )


def test_a_defect_box_that_contains_the_patch_center_counts_as_found() -> None:
    tile = Reported((60, 50, 160, 100), 0.9, True, ("pedestrian",))
    trial = _score([tile])
    assert trial.found and trial.on_road
    assert trial.in_road_correct is True
    assert trial.near_expected == ("pedestrian",) and trial.near_reported == ("pedestrian",)


def test_a_box_elsewhere_is_not_found() -> None:
    assert not _score([Reported((0, 0, 40, 40), 0.9, True, ())]).found


def test_the_highest_confidence_covering_box_is_the_one_scored() -> None:
    weak = Reported((60, 50, 160, 100), 0.4, None, ())
    strong = Reported((70, 50, 170, 100), 0.9, True, ("car",))
    trial = _score([weak, strong])
    assert trial.near_reported == ("car",)


def test_in_road_is_judged_against_the_hand_outlined_road_for_the_reported_box() -> None:
    off_road_tile = Reported((90, 0, 130, 35), 0.9, True, ())  # all above the road: truth False
    far_patch = (100.0, 10.0, 110.0, 20.0)
    trial = score_trial(
        far_patch, False, [off_road_tile], road_truth=polygon_mask(ROAD, SIZE),
        road_threshold=0.5, expected_near=_near,
    )  # fmt: skip
    assert trial.found and trial.in_road_correct is False  # reported True, truth False


def test_an_unknown_in_road_is_counted_apart_not_as_right_or_wrong() -> None:
    trial = _score([Reported((60, 50, 160, 100), 0.9, None, ())])
    assert trial.in_road_correct is None


def test_a_trial_with_no_defects_has_no_in_road_or_near_scores() -> None:
    trial = _score([])
    assert not trial.found and trial.in_road_correct is None and trial.near_reported == ()


# ------------------------------------------------------------------------------ summary


def _trial(found: bool, on_road: bool = True, box=(0, 70, 30, 100), **kw) -> Trial:  # type: ignore[no-untyped-def]
    base = {
        "truth_box": box, "on_road": on_road, "found": found, "iou": 0.1 if found else 0.0,
        "in_road_correct": True if found else None,
        "reported_in_road": True if found else None,
        "near_expected": (), "near_reported": (),
    }  # fmt: skip
    base.update(kw)
    return Trial(**base)


def _summ(trials: list[Trial]) -> dict[str, Any]:
    return summarize(trials, size_edges_px=(20, 40), y_edges_frac=(0.6, 0.8), image_height=100)


def test_detection_rate_is_by_size_and_by_height_with_ranges() -> None:
    trials = [_trial(True), _trial(True), _trial(False), _trial(True, box=(0, 40, 50, 60))]
    report = _summ(trials)
    on = report["on_road"]
    assert on["trials"] == 4 and on["found"] == 3
    low, high = on["found_range90"]
    assert low < 0.75 < high
    assert on["by_size"]["small (<20 px)"]["trials"] == 0
    assert on["by_size"]["medium (20-40 px)"]["trials"] == 3
    large = on["by_size"]["large (>=40 px)"]
    assert (large["trials"], large["found"], large["rate"]) == (1, 1, 1.0)
    assert on["by_height"]["far (above 60% of height)"]["trials"] == 1


def test_off_road_leakage_counts_found_defects_not_reported_as_off_the_road() -> None:
    trials = [
        _trial(False, on_road=False),
        _trial(True, on_road=False, reported_in_road=False),
        _trial(True, on_road=False, reported_in_road=True),
        _trial(True, on_road=False, reported_in_road=None),
    ]
    off = _summ(trials)["off_road"]
    assert off["trials"] == 4 and off["not_reported"] == 1 and off["reported_off_road"] == 1
    assert off["leaked"] == 2


def test_in_road_accuracy_ignores_unknown_and_reports_how_many_were_unknown() -> None:
    trials = [
        _trial(True, in_road_correct=True), _trial(True, in_road_correct=False),
        _trial(True, in_road_correct=None), _trial(True, in_road_correct=True),
    ]  # fmt: skip
    acc = _summ(trials)["in_road"]
    assert acc == {"scored": 3, "correct": 2, "unknown": 1, "rate": pytest.approx(2 / 3),
                   "range90": acc["range90"]}  # fmt: skip


def test_near_objects_precision_and_recall_are_micro_averaged_over_labels() -> None:
    trials = [
        _trial(True, near_expected=("car", "pedestrian"), near_reported=("car",)),
        _trial(True, near_expected=(), near_reported=("truck",)),
        _trial(True, near_expected=("cyclist",), near_reported=("cyclist",)),
    ]
    near = _summ(trials)["near_objects"]
    assert near["expected"] == 3 and near["reported"] == 3 and near["matched"] == 2
    assert near["recall"] == pytest.approx(2 / 3) and near["precision"] == pytest.approx(2 / 3)
    assert near["exact"] == 1 and near["defects"] == 3


def test_an_empty_summary_does_not_divide_by_zero() -> None:
    report = _summ([])
    assert report["on_road"]["trials"] == 0 and report["in_road"]["rate"] is None
    assert report["near_objects"]["recall"] is None


# ------------------------------------------------------------------------------- verdict

CRITERIA = {
    "min_detection_rate_large": 0.7,
    "min_in_road_accuracy": 0.9,
    "max_off_road_leak_rate": 0.1,
    "min_near_recall": 0.5,
}


def _report(found_large: int, leaked: int, correct: int, matched: int) -> dict:  # type: ignore[type-arg]
    return {
        "on_road": {"by_size": {"large (>=40 px)": {"trials": 10, "found": found_large,
                                                   "rate": found_large / 10}}},
        "off_road": {"trials": 10, "leaked": leaked},
        "in_road": {"scored": 10, "correct": correct, "rate": correct / 10},
        "near_objects": {"expected": 10, "matched": matched, "recall": matched / 10},
    }  # fmt: skip


def test_verdict_passes_when_every_criterion_is_met() -> None:
    result = verdict(_report(8, 0, 10, 6), CRITERIA)
    assert result["passed"] and result["failures"] == []


def test_verdict_lists_every_failed_criterion() -> None:
    result = verdict(_report(5, 3, 8, 2), CRITERIA)
    assert not result["passed"] and len(result["failures"]) == 4


def test_verdict_refuses_to_pass_with_no_evidence() -> None:
    empty = {
        "on_road": {"by_size": {"large (>=40 px)": {"trials": 0, "found": 0, "rate": None}}},
        "off_road": {"trials": 0, "leaked": 0},
        "in_road": {"scored": 0, "correct": 0, "rate": None},
        "near_objects": {"expected": 0, "matched": 0, "recall": None},
    }
    assert not verdict(empty, CRITERIA)["passed"]


def test_the_evaluation_module_does_not_import_perception() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import eval.fusion_synthetic; "
        "assert not [m for m in sys.modules if m == 'perception' or m.startswith('perception.')]"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(root)], check=True)
