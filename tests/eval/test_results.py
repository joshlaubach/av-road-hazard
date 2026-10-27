"""eval/results.py: one results file for the core, and the README table rendered from it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.results import (
    README_END,
    README_START,
    build_results,
    effective_range,
    render_readme_table,
    update_readme,
    validate_results,
)

ROOT = Path(__file__).resolve().parents[2]


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "total": total, "rate": count / total, "low90": 0.0, "high90": 1.0}


def _classifier_report() -> dict[str, Any]:
    validation = {
        c: {"precision": 0.9, "recall": 0.5, "auroc": 0.95, "flagged": 10, "positives": 20,
            "recall_at_baseline_fpr": 0.5}
        for c in ("D00", "D10", "D20", "D40")
    }  # fmt: skip

    def frames(alerts: int) -> dict[str, Any]:
        return {
            "alerts": alerts, "per_frame": alerts / 404,
            "day": {"frames": 283, "alerts": alerts, "per_frame": alerts / 283},
            "night": {"frames": 121, "alerts": 0, "per_frame": 0.0},
            "by_class": {"D00": alerts}, "by_road_cover": {"0.4-0.6": alerts},
            "buckets": {"Crack": 7, "No crack": 36, "Not road": 1, "Ambiguous": 8, "Unreviewed": 0},
        }  # fmt: skip

    return {
        "baseline": "baseline", "fold": 0,
        "provenance": {"files": {"checkpoint/baseline": "a" * 64}, "git_commit": "abc",
                       "git_dirty": False, "date_utc": "2026-10-26T00:00:00Z",
                       "thresholds": {"D00": 0.9, "D10": 0.75, "D20": 0.65, "D40": 0.75}},
        "checkpoints": {
            "baseline": {"validation": validation, "probe": {"flagged": 23, "total": 23,
                         "low90": 0.9, "high90": 1.0}, "frames": frames(80)},
            "hn2-repeat10": {"validation": validation, "probe": {"flagged": 13, "total": 23,
                             "low90": 0.4, "high90": 0.7}, "frames": frames(53),
                             "guard": {"passed": True, "reasons": []}},
        },
    }  # fmt: skip


def _nonroad_report() -> dict[str, Any]:
    band = {"tiles": 100, "cover_from": 0.0, "cover_below": 0.05, "flagged": _rate(1, 100),
            "per_class": {"D00": _rate(1, 100)}}  # fmt: skip
    return {"frames": 404, "non_road_tiles_scored": 150, "clearly_not_road": band,
            "partly_road_skipped": {**band, "tiles": 50, "flagged": _rate(2, 50)},
            "provenance": {"git_commit": "abc", "git_dirty": False, "files": {},
                           "date_utc": "2026-10-26T00:00:00Z", "thresholds": {}}}  # fmt: skip


def _fusion_report() -> dict[str, Any]:
    def band(found: int, trials: int) -> dict[str, Any]:
        return {"trials": trials, "found": found, "rate": found / trials, "range90": [0.1, 0.9]}

    return {
        "on_road": {"trials": 170, "found": 54, "found_range90": [0.26, 0.38],
                    "by_size": {"small (<64 px)": band(3, 57), "medium (64-128 px)": band(17, 56),
                                "large (>=128 px)": band(34, 57)}, "by_height": {}},
        "off_road": {"trials": 45, "not_reported": 45, "reported_off_road": 0, "leaked": 0,
                     "leak_rate": 0.0, "leak_range90": [0.0, 0.06]},
        "in_road": {"scored": 54, "correct": 52, "unknown": 0, "rate": 0.963,
                    "range90": [0.89, 0.99]},
        "near_objects": {"defects": 54, "expected": 105, "reported": 85, "matched": 75,
                         "recall": 0.714, "precision": 0.882, "exact": 24},
        "median_iou": 0.11, "verdict": {"passed": False, "failures": ["large patches 0.60 < 0.70"]},
        "criteria": {"min_detection_rate_large": 0.7}, "clean_frames": {"frames": 30,
        "defects": 1, "per_frame": 0.033}, "skipped_placements": 55, "patches_in_bank": 200,
        "provenance": {"git_commit": "abc", "git_dirty": False, "files": {},
                       "date_utc": "2026-10-26T00:00:00Z", "thresholds": {}},
    }  # fmt: skip


def _detector_gap() -> dict[str, Any]:
    model = {
        "threshold": 0.7,
        "all_frames_evaluable": {"frames": 404, "precision": 0.57, "recall": 0.44, "f1": 0.5,
                                 "ap50": 0.3},
        "recall_by_size_all_gt": [{"low": 0.0, "high": 1024.0, "n": 390, "recall": 0.0026}],
        "outside_vocabulary": {"barrier": 172, "traffic_cone": 285},
    }  # fmt: skip
    return {"version": "v1.0-mini", "frames": 404, "models": {"m": model},
            "provenance": {"git_commit": "abc", "git_dirty": False, "files": {},
                           "date_utc": "2026-10-26T00:00:00Z", "thresholds": {}}}  # fmt: skip


def _finetune_report() -> dict[str, Any]:
    pooled = {"tp": 501, "fp": 173, "fn": 362, "boxes": 863, "predictions": 674,
              "precision": 0.743, "recall": 0.581, "precision_range90": [0.71, 0.77],
              "recall_range90": [0.55, 0.61]}  # fmt: skip

    def new(passed: bool) -> dict[str, Any]:
        return {"verdict": {"passed": passed, "pooled": pooled, "qualifying_scenes": ["a", "b"],
                            "passing_scenes": ["a"], "caveat": False, "caveat_scenes": [],
                            "reasons": []}}  # fmt: skip

    side = {"boxes": 100, "precision": 0.4, "recall": 0.4}
    old = {
        "pedestrian": {"passed": True, "zero_shot": side, "fine_tuned": {**side, "precision": 0.7}}
    }
    view = {"passed": False, "new_classes": {"traffic_cone": new(True), "barrier": new(False)},
            "old_classes": old}  # fmt: skip
    return {"all_cameras": view, "front_camera": {**view, "passed": False},
            "cpu_latency_ladder": [{"threads": 24, "p95_s": 0.18, "passed": True}],
            "provenance": {"git_commit": "abc", "git_dirty": False, "files": {},
                           "date_utc": "2026-10-26T00:00:00Z", "thresholds": {}}}  # fmt: skip


def _results() -> dict[str, Any]:
    return build_results(
        classifier=_classifier_report(), non_road=_nonroad_report(),
        detector_gap=_detector_gap(), fine_tune=_finetune_report(), fusion=_fusion_report(),
        chosen="hn2-repeat10", date_utc="2026-10-26T00:00:00Z", git_commit="abc",
    )  # fmt: skip


def test_the_results_have_every_section_the_core_needs() -> None:
    results = _results()
    validate_results(results)
    assert {"classifier", "detector", "synthetic_fusion", "effective_range", "retrieval"} <= set(
        results
    )
    assert results["retrieval"]["status"] == "not measured"
    assert results["classifier"]["non_road_false_positives"]["clearly_not_road"]["tiles"] == 100
    assert results["detector"]["fine_tune"]["passed"] is False
    assert results["synthetic_fusion"]["verdict"]["passed"] is False


def test_the_results_hold_numbers_only_no_frames_files_or_paths() -> None:
    text = json.dumps(_results())
    for forbidden in (".jpg", ".png", "samples/", "C:\\\\", "C:/", "n008-"):
        assert forbidden not in text


def test_the_effective_range_turns_patch_width_into_a_distance_for_a_stated_defect_size() -> None:
    rng = effective_range(_fusion_report(), focal_px=1266.0, defect_width_m=0.5)
    by = {row["band"]: row for row in rng["bands"]}
    assert by["large (>=128 px)"]["found"] == 34 and by["large (>=128 px)"]["trials"] == 57
    assert rng["distance_m_at_width_px"]["128"] == pytest.approx(4.95, abs=0.005)
    assert rng["distance_m_at_width_px"]["64"] == pytest.approx(9.89, abs=0.005)
    assert "pasted" in rng["caveat"]


def test_the_readme_table_states_the_headline_numbers_from_the_results() -> None:
    table = render_readme_table(_results())
    for expected in ("0.198", "0.131", "13 of 23", "1 of 100", "54 of 170", "5%", "30%", "60%",
                     "0 of 45", "52 of 54", "not measured", "zero-shot"):  # fmt: skip
        assert expected in table, expected


def test_the_table_is_swapped_into_the_readme_between_its_markers() -> None:
    text = f"intro\n{README_START}\nold\n{README_END}\noutro\n"
    out = update_readme(text, "NEW")
    assert out == f"intro\n{README_START}\nNEW\n{README_END}\noutro\n"
    with pytest.raises(ValueError, match="markers"):
        update_readme("no markers here", "NEW")


def test_a_results_file_missing_a_section_or_holding_a_path_is_refused() -> None:
    results = _results()
    del results["classifier"]
    with pytest.raises(ValueError, match="classifier"):
        validate_results(results)
    leaky = _results()
    leaky["classifier"]["note"] = "see C:/Users/someone/frame.jpg"
    with pytest.raises(ValueError, match="path"):
        validate_results(leaky)


# --- the committed files -------------------------------------------------------------------


def test_the_committed_results_file_is_valid() -> None:
    results = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    validate_results(results)


def test_the_readme_table_is_exactly_what_the_committed_results_render() -> None:
    results = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start, end = readme.index(README_START), readme.index(README_END)
    block = readme[start + len(README_START) : end].strip("\n")
    assert block == render_readme_table(results).strip("\n")
