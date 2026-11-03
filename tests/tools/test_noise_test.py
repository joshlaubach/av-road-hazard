"""Tests for tools/noise_test.py: small made-up records, the real fusion functions, no models."""

from __future__ import annotations

import random

from fusion.config import load_fusion_config
from fusion.fuse import fuse
from fusion.schema import Defect, HazardRecord, Object
from tools.noise_test import effect, false_defect, frame_score, run, summarize, top_n

CFG = load_fusion_config()
THRESHOLDS = {"D00": 0.9, "D10": 0.75, "D20": 0.65, "D40": 0.75}


def make_record(frame_id: str, defects: list[Defect], objects: list[Object]) -> HazardRecord:
    return fuse(
        frame_id=frame_id,
        source="nuscenes",
        image_size=(1600, 900),
        objects=objects,
        defects=defects,
        road_mask=None,
        config=CFG,
    )


def pothole(confidence: float = 0.8) -> Defect:
    return Defect(
        cls=4, name="D40", confidence=confidence, box=(600, 500, 900, 800), patch_row=0, patch_col=0
    )


def person() -> Object:
    return Object(label="pedestrian", score=0.9, box=(700, 450, 760, 600))


def test_recomputed_score_matches_the_stored_one() -> None:
    record = make_record("a", [pothole()], [person()])
    assert abs(frame_score(record.objects, record.defects, CFG) - record.hazard_score) < 1e-9


def test_a_frame_without_defects_scores_zero() -> None:
    assert frame_score([person()], [], CFG) == 0.0


def test_top_n_breaks_ties_by_frame_id() -> None:
    assert top_n({"b": 0.5, "a": 0.5, "c": 0.1})[:2] == ["a", "b"]


def test_effect_reports_points_and_ranking_changes() -> None:
    base = {f"f{i}": 0.0 for i in range(20)}
    points, members, order = effect(base, "f19", 0.5)
    assert points == 50.0 and members and order  # a clean frame jumps into the top ten


def test_effect_with_no_change_changes_nothing() -> None:
    base = {"a": 0.4, "b": 0.2}
    assert effect(base, "a", 0.4) == (0.0, False, False)


def test_false_defect_never_lowers_the_score_and_stays_on_the_frame() -> None:
    record = make_record("a", [pothole()], [person()])
    fake = false_defect(record, "D10", 0.75, random.Random(0), CFG)
    x1, y1, x2, y2 = fake.box
    assert 0 <= x1 < x2 <= 1600 and 0 <= y1 < y2 <= 900 and fake.in_road is True
    base = frame_score(record.objects, record.defects, CFG)
    assert max(base, frame_score(record.objects, [fake], CFG)) >= base


def test_removing_an_object_never_raises_the_score() -> None:
    record = make_record("a", [pothole()], [person()])
    with_object = frame_score(record.objects, record.defects, CFG)
    without = frame_score([], record.defects, CFG)
    assert without <= with_object


def test_run_checks_stored_scores_and_counts_every_case() -> None:
    records = [make_record("clean", [], [person()]), make_record("hole", [pothole()], [person()])]
    report = run(records, THRESHOLDS, CFG, positions=2, seed=1)
    assert report["recomputed_scores_differing_from_stored"] == 0
    assert report["missed_object"]["cases"] == 1  # only the frame with a defect and one object
    clean = report["false_defect"]["at_threshold"]["by_frame_group"]["clean_frames"]
    assert clean["cases"] == 1 * 4 * 2  # one frame, four classes, two places


def test_summarize_with_no_cases_is_empty_not_an_error() -> None:
    assert summarize([])["cases"] == 0
