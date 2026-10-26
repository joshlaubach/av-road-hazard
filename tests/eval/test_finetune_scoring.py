"""eval/finetune.py: per-scene counts, leave-one-scene-out cutoffs and the ADR 0015 verdicts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from eval.detector import FrameData, GtBox
from eval.finetune import (
    add_counts,
    choose_cutoffs,
    class_counts,
    new_label_verdict,
    rates,
    regression_verdict,
    scene_rows,
    scene_table,
)
from eval.models import Detection

BOX = (100.0, 100.0, 160.0, 160.0)
FAR = (400.0, 400.0, 460.0, 460.0)
GRID = [0.3, 0.5, 0.7, 0.9]
LIMITS = {
    "min_precision": 0.7, "min_recall": 0.4, "min_boxes": 20,
    "min_passing_scenes": {"barrier": 2, "traffic_cone": 1},
    "max_recall_drop": 0.03, "max_precision_drop": 0.05,
}  # fmt: skip


def _frame(scene: str, detections: list[Detection], truth: list[GtBox]) -> FrameData:
    return FrameData(scene, False, detections, truth)


def _gt(label: str = "traffic_cone", box=BOX, visibility: int = 4) -> GtBox:  # type: ignore[no-untyped-def]
    return GtBox(label, box, visibility)


def test_counts_match_by_label_and_overlap() -> None:
    frame = _frame(
        "s",
        [
            Detection("traffic_cone", 0.9, BOX),  # hit
            Detection("traffic_cone", 0.8, FAR),  # nothing there: false positive
            Detection("car", 0.9, BOX),  # another class: ignored
        ],
        [_gt(), _gt(box=(10.0, 10.0, 70.0, 70.0))],  # the second is missed
    )
    assert class_counts([frame], "traffic_cone", 0.5) == {"tp": 1, "fp": 1, "fn": 1}


def test_the_cutoff_removes_weak_detections() -> None:
    frame = _frame("s", [Detection("traffic_cone", 0.6, BOX)], [_gt()])
    assert class_counts([frame], "traffic_cone", 0.5)["tp"] == 1
    assert class_counts([frame], "traffic_cone", 0.7) == {"tp": 0, "fp": 0, "fn": 1}


def test_boxes_that_are_too_hidden_or_small_do_not_count_as_misses() -> None:
    hidden = _frame("s", [], [_gt(visibility=1), _gt(box=(0.0, 0.0, 10.0, 10.0))])
    assert class_counts([hidden], "traffic_cone", 0.5) == {"tp": 0, "fp": 0, "fn": 0}


def test_rates_have_ranges_and_none_when_there_is_nothing_to_divide() -> None:
    result = rates({"tp": 8, "fp": 2, "fn": 8})
    assert result["precision"] == 0.8 and result["recall"] == 0.5 and result["boxes"] == 16
    assert result["precision_range90"][0] < 0.8 < result["precision_range90"][1]
    empty = rates({"tp": 0, "fp": 0, "fn": 0})
    assert empty["precision"] is None and empty["recall"] is None


def _table(per_scene: dict[str, dict[float, tuple[int, int, int]]]):  # type: ignore[no-untyped-def]
    return {
        scene: {c: {"tp": t, "fp": f, "fn": n} for c, (t, f, n) in cuts.items()}
        for scene, cuts in per_scene.items()
    }


def test_each_scene_gets_the_lowest_cutoff_that_is_precise_enough_on_the_others() -> None:
    table = _table(
        {
            "a": {0.3: (50, 50, 0), 0.5: (45, 15, 5), 0.7: (30, 2, 20), 0.9: (5, 0, 45)},
            "b": {0.3: (50, 50, 0), 0.5: (45, 15, 5), 0.7: (30, 2, 20), 0.9: (5, 0, 45)},
            "c": {0.3: (50, 50, 0), 0.5: (45, 15, 5), 0.7: (30, 2, 20), 0.9: (5, 0, 45)},
        }
    )
    chosen = choose_cutoffs(table, GRID, min_precision=0.7)
    assert all(v == {"cutoff": 0.5, "reached": True} for v in chosen.values())  # 45/60 = 0.75


def test_a_scene_never_influences_its_own_cutoff() -> None:
    perfect = {0.3: (10, 0, 0), 0.5: (10, 0, 0), 0.7: (10, 0, 0), 0.9: (10, 0, 0)}
    noisy = {0.3: (10, 90, 0), 0.5: (10, 90, 0), 0.7: (10, 40, 0), 0.9: (10, 5, 0)}
    chosen = choose_cutoffs(_table({"a": perfect, "b": noisy}), GRID, min_precision=0.7)
    assert chosen["b"]["cutoff"] == 0.3  # judged on "a" alone, which is perfect
    assert chosen["a"]["cutoff"] == 0.9  # judged on "b" alone: only 0.9 reaches 0.67...
    assert chosen["a"]["reached"] is False  # 10/(10+5) = 0.667 < 0.7: the highest is used


def test_scene_rows_use_each_scenes_own_cutoff() -> None:
    frames = {"a": [_frame("a", [Detection("traffic_cone", 0.6, BOX)], [_gt()])]}
    table = scene_table(frames, "traffic_cone", GRID)
    rows = scene_rows(table, {"a": {"cutoff": 0.5, "reached": True}})
    assert rows[0]["scene"] == "a" and rows[0]["tp"] == 1 and rows[0]["cutoff"] == 0.5


def _row(scene: str, tp: int, fp: int, fn: int) -> dict:  # type: ignore[type-arg]
    return {"scene": scene, **rates({"tp": tp, "fp": fp, "fn": fn})}


def test_a_new_label_passes_when_enough_scenes_reach_both_limits() -> None:
    rows = [_row("a", 80, 10, 40), _row("b", 70, 20, 50), _row("tiny", 3, 0, 4)]
    result = new_label_verdict(rows, "traffic_cone", LIMITS)
    assert result["passed"] and result["qualifying_scenes"] == ["a", "b"]  # "tiny" has 7 boxes
    assert result["passing_scenes"] == ["a", "b"]


def test_barriers_need_two_scenes_that_pass_on_their_own() -> None:
    one_good = [_row("a", 80, 10, 40), _row("b", 5, 40, 80)]
    result = new_label_verdict(one_good, "barrier", LIMITS)
    assert not result["passed"]
    assert any("2 needed" in r for r in result["reasons"])


def test_a_pooled_pass_that_hides_a_bad_scene_still_needs_the_per_scene_count() -> None:
    rows = [_row("a", 300, 10, 20), _row("b", 2, 30, 30)]
    assert new_label_verdict(rows, "traffic_cone", LIMITS)["passed"]  # cone needs only one scene
    assert not new_label_verdict(rows, "barrier", LIMITS)["passed"]  # barrier needs two


def test_too_few_boxes_anywhere_fails() -> None:
    result = new_label_verdict([_row("a", 5, 0, 5)], "traffic_cone", LIMITS)
    assert not result["passed"] and "only 0 scene(s)" in result["reasons"][0]


def test_a_pass_with_a_wide_range_is_flagged_as_a_caveat() -> None:
    # 14 of 20: precision 0.70 passes, but its 90% range reaches below 0.7.
    rows = [_row("a", 14, 6, 6)]
    result = new_label_verdict(rows, "traffic_cone", LIMITS)
    assert result["passed"] and result["caveat"]


def test_regression_allows_small_drops_only() -> None:
    before = {"tp": 80, "fp": 20, "fn": 20}  # precision 0.80, recall 0.80
    assert regression_verdict(before, {"tp": 79, "fp": 21, "fn": 21}, LIMITS)["passed"]
    worse = regression_verdict(before, {"tp": 70, "fp": 20, "fn": 30}, LIMITS)  # recall 0.70
    assert not worse["passed"] and "recall fell" in worse["reasons"][0]
    sloppy = regression_verdict(before, {"tp": 80, "fp": 40, "fn": 20}, LIMITS)  # precision 0.67
    assert not sloppy["passed"] and "precision fell" in sloppy["reasons"][0]


def test_add_counts_sums_each_field() -> None:
    assert add_counts([{"tp": 1, "fp": 2, "fn": 3}, {"tp": 4, "fp": 5, "fn": 6}]) == {
        "tp": 5, "fp": 7, "fn": 9,
    }  # fmt: skip


def test_the_module_does_not_import_perception() -> None:
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import eval.finetune; "
        "assert not [m for m in sys.modules if m == 'perception' or m.startswith('perception.')]"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(root)], check=True)


@pytest.mark.parametrize("label", ["barrier", "traffic_cone"])
def test_an_empty_scene_list_gives_no_rows(label: str) -> None:
    assert scene_rows(scene_table({}, label, GRID), {}) == []


# ------------------------------------------------------------------------------ the report

EVALUATION = {
    "cutoff_grid": GRID,
    "old_classes_cutoff": 0.5,
    "new_classes": ["traffic_cone"],
    "limits": LIMITS,
}


def _scene(name: str, good: bool) -> list[FrameData]:
    """20 frames, each with one cone: found with score 0.8 (good) or missed with a ghost (bad)."""
    frames = []
    for _ in range(20):
        truth = [_gt("traffic_cone"), _gt("car", box=FAR)]
        preds = [Detection("car", 0.9, FAR)]
        preds += (
            [Detection("traffic_cone", 0.8, BOX)]
            if good
            else [Detection("traffic_cone", 0.8, (0.0, 0.0, 50.0, 50.0))]
        )
        frames.append(_frame(name, preds, truth))
    return frames


def test_the_report_passes_when_the_new_class_is_found_and_old_classes_are_unchanged() -> None:
    from eval.finetune import build_report

    scenes = {"a": _scene("a", True), "b": _scene("b", True)}
    report = build_report(scenes, scenes, EVALUATION, ["car"])
    assert report["passed"]
    assert report["new_classes"]["traffic_cone"]["verdict"]["passing_scenes"] == ["a", "b"]
    assert report["old_classes"]["car"]["passed"]


def test_the_report_fails_when_the_new_class_only_produces_ghosts() -> None:
    from eval.finetune import build_report

    scenes = {"a": _scene("a", False), "b": _scene("b", False)}
    report = build_report(scenes, scenes, EVALUATION, ["car"])
    assert not report["passed"] and not report["new_classes"]["traffic_cone"]["verdict"]["passed"]


def test_the_report_fails_when_an_old_class_gets_worse() -> None:
    from eval.finetune import build_report

    good = {"a": _scene("a", True), "b": _scene("b", True)}
    no_cars = {
        s: [
            _frame(s, [d for d in f.detections if d.label != "car"], f.ground_truth) for f in frames
        ]
        for s, frames in good.items()
    }
    report = build_report(no_cars, good, EVALUATION, ["car"])
    assert not report["old_classes"]["car"]["passed"] and not report["passed"]
    assert len(report["new_classes"]["traffic_cone"]["sweep_all_scenes"]) == len(GRID)
