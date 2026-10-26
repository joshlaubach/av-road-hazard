from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from fusion.config import FusionConfig, load_fusion_config
from fusion.fuse import (
    fuse,
    in_road_state,
    near_objects_for,
    normalized_distance,
    object_distance_px,
    road_multiplier,
)
from fusion.schema import Defect, HazardRecord, Meta, Object, Severity

W, H = 640, 480
CFG = load_fusion_config()
DEFECT_BOX = (100.0, 300.0, 140.0, 340.0)
# A pedestrian whose bottom-center (120, 330) is inside the defect box: distance 0.
PEDESTRIAN = Object(label="pedestrian", score=0.9, box=(110.0, 200.0, 130.0, 330.0))
ALL_ROAD = np.ones((H, W), dtype=bool)
NO_ROAD = np.zeros((H, W), dtype=bool)


def _defect(confidence: float = 0.9, box: tuple[float, float, float, float] = DEFECT_BOX) -> Defect:
    return Defect(cls=1, name="x", confidence=confidence, box=box, patch_row=0, patch_col=0)


def _high(_: Defect) -> Severity:
    return "high"


def _score(
    objects: list[Object],
    mask: np.ndarray | None = ALL_ROAD,
    confidence: float = 0.9,
    cfg: FusionConfig = CFG,
    severity: Severity = "high",
) -> float:
    record = fuse(
        frame_id="f",
        source="upload",
        image_size=(W, H),
        objects=objects,
        defects=[_defect(confidence)],
        road_mask=mask,
        config=cfg,
        severity_fn=lambda _: severity,
    )
    return record.hazard_score


def test_worked_examples() -> None:
    """High severity, confidence 0.9: 0.45 alone, 0.855 beside a pedestrian, 0 off-road."""
    assert _score([]) == pytest.approx(0.45)
    assert _score([PEDESTRIAN]) == pytest.approx(0.855)
    assert _score([PEDESTRIAN], mask=NO_ROAD) == 0.0
    assert _score([], mask=NO_ROAD) == 0.0


def test_off_road_defect_stays_in_the_record() -> None:
    """A confirmed off-road defect scores 0 but is kept, with in_road=False, for inspection."""
    record = fuse(
        frame_id="f",
        source="upload",
        image_size=(W, H),
        objects=[PEDESTRIAN],
        defects=[_defect()],
        road_mask=NO_ROAD,
        config=CFG,
    )
    assert record.hazard_score == 0.0
    assert len(record.defects) == 1
    assert record.defects[0].in_road is False
    assert record.defects[0].confidence == 0.9


def test_ordering_nearby_pedestrian_over_empty_road_over_off_road() -> None:
    """The project's central claim: a pedestrian nearby raises urgency; off-road is lowest."""
    near = _score([PEDESTRIAN])
    empty = _score([])
    off_road = _score([PEDESTRIAN], mask=NO_ROAD)
    assert near > empty > off_road == 0.0


def test_unknown_road_status_keeps_its_contribution() -> None:
    """No mask means in_road=None and the same score as on-road, never zero."""
    record = fuse(
        frame_id="f",
        source="upload",
        image_size=(W, H),
        objects=[PEDESTRIAN],
        defects=[_defect()],
        config=CFG,
        severity_fn=_high,
    )
    assert record.defects[0].in_road is None
    assert record.hazard_score == pytest.approx(_score([PEDESTRIAN], mask=ALL_ROAD))
    assert record.hazard_score > 0


def test_all_road_mask_with_default_config_is_on_road_and_scores() -> None:
    """Regression: a valid default config must not turn an on-road defect into a zero score."""
    record = fuse(
        frame_id="f",
        source="upload",
        image_size=(W, H),
        objects=[],
        defects=[_defect()],
        road_mask=ALL_ROAD,
        config=CFG,
    )
    assert record.defects[0].in_road is True
    assert record.hazard_score > 0


def test_severity_defaults_to_unknown_weight() -> None:
    """Until the class mapping is verified every fused defect is severity 'unknown'."""
    record = fuse(
        frame_id="f",
        source="upload",
        image_size=(W, H),
        objects=[],
        defects=[_defect(1.0)],
        road_mask=ALL_ROAD,
        config=CFG,
    )
    assert record.defects[0].severity == "unknown"
    assert record.hazard_score == pytest.approx(CFG.severity_weights["unknown"] * 0.5)


def test_score_does_not_increase_as_the_object_moves_away() -> None:
    """Sliding the pedestrian away horizontally never raises the score."""
    scores = []
    for dx in range(0, 600, 40):
        obj = Object(label="pedestrian", score=0.9, box=(110.0 + dx, 200.0, 130.0 + dx, 330.0))
        scores.append(_score([obj]))
    assert all(a >= b for a, b in itertools.pairwise(scores))
    assert scores[0] > scores[-1]


def test_score_does_not_decrease_with_confidence_or_severity_or_object_confidence() -> None:
    """More defect confidence, higher severity and a surer object never lower the score."""
    by_conf = [_score([PEDESTRIAN], confidence=c / 10) for c in range(11)]
    assert all(a <= b for a, b in itertools.pairwise(by_conf))
    by_sev = [_score([PEDESTRIAN], severity=s) for s in ("low", "medium", "high")]
    assert by_sev[0] < by_sev[1] < by_sev[2]
    by_obj = [
        _score([Object(label="pedestrian", score=s / 10, box=PEDESTRIAN.box)]) for s in range(11)
    ]
    assert all(a <= b for a, b in itertools.pairwise(by_obj))


def test_vulnerable_road_users_outrank_vehicles_at_equal_distance() -> None:
    """A pedestrian beside the defect scores above a car in the same place."""
    car = Object(label="car", score=0.9, box=PEDESTRIAN.box)
    assert _score([PEDESTRIAN]) > _score([car])


def test_crowd_or_duplicates_never_exceed_the_strongest_object() -> None:
    """Twenty copies of the same pedestrian score exactly like one."""
    assert _score([PEDESTRIAN] * 20) == pytest.approx(_score([PEDESTRIAN]))


def test_scores_are_bounded_and_records_validate_under_random_inputs() -> None:
    """Seeded fuzz: hazard_score stays in [0, 1] and the record round-trips through JSON."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        objs = []
        for _ in range(int(rng.integers(0, 6))):
            x1, y1 = rng.uniform(0, W - 10), rng.uniform(0, H - 10)
            objs.append(
                Object(
                    label=str(rng.choice(["pedestrian", "car", "cyclist", "weird"])),
                    score=float(rng.uniform(0, 1)),
                    box=(x1, y1, x1 + rng.uniform(0, 80), y1 + rng.uniform(0, 160)),
                )
            )
        x1, y1 = rng.uniform(-30, W), rng.uniform(-30, H)
        defect = _defect(float(rng.uniform(0, 1)), (x1, y1, x1 + rng.uniform(0, 90), y1 + 40.0))
        mask = [None, ALL_ROAD, NO_ROAD][int(rng.integers(0, 3))]
        record = fuse(
            frame_id="fz",
            source="nuscenes",
            image_size=(W, H),
            objects=objs,
            defects=[defect],
            road_mask=mask,
            config=CFG,
            severity_fn=lambda _: str(rng.choice(["low", "medium", "high", "unknown"])),  # type: ignore[arg-type,return-value]
        )
        assert 0.0 <= record.hazard_score <= 1.0
        assert HazardRecord.model_validate_json(record.model_dump_json()) == record


def test_no_objects_and_no_defects() -> None:
    """Empty inputs are valid: no defects scores 0.0 and the record is fully populated."""
    record = fuse(
        frame_id="e",
        source="nuscenes",
        image_size=(W, H),
        objects=[],
        defects=[],
        meta=Meta(time_of_day="night", location="x"),
        config=CFG,
    )
    assert record.hazard_score == 0.0
    assert record.defects == [] and record.objects == []
    assert record.meta.time_of_day == "night"
    assert record.image_size == (W, H)


def test_zero_height_object_uses_the_configured_minimum() -> None:
    """A flat box divides by min_object_height_px instead of zero."""
    flat = (100.0, 400.0, 110.0, 400.0)  # height 0, bottom-center (105, 400)
    d_px = object_distance_px(flat, DEFECT_BOX)
    assert d_px == pytest.approx(60.0)  # 60 px below the defect box
    assert normalized_distance(flat, DEFECT_BOX, CFG) == pytest.approx(
        60.0 / CFG.min_object_height_px
    )


def test_distance_is_zero_inside_and_measured_to_the_nearest_edge() -> None:
    """Bottom-center inside the defect box is 0; outside it is the gap to the nearest point."""
    assert object_distance_px(PEDESTRIAN.box, DEFECT_BOX) == 0.0
    to_the_right = (200.0, 0.0, 220.0, 320.0)  # bottom-center (210, 320): 70 px right of x2=140
    assert object_distance_px(to_the_right, DEFECT_BOX) == pytest.approx(70.0)


def test_in_road_state_threshold_edge_and_outside_image() -> None:
    """Half-covered boxes meet the 0.5 threshold; boxes outside the image are unknown."""
    half = np.zeros((H, W), dtype=bool)
    half[:, :120] = True  # box x 100..140 is exactly half road
    assert in_road_state(DEFECT_BOX, half, CFG) is True
    strict = dataclasses.replace(CFG, road_overlap_threshold=0.6)
    assert in_road_state(DEFECT_BOX, half, strict) is False
    assert in_road_state((700.0, 10.0, 720.0, 20.0), ALL_ROAD, CFG) is None
    assert in_road_state((-20.0, 340.0, 20.0, 400.0), ALL_ROAD, CFG) is True  # partly outside
    assert in_road_state(DEFECT_BOX, None, CFG) is None


def test_road_multiplier_values() -> None:
    """False is 0; True and unknown are 1."""
    assert (road_multiplier(False), road_multiplier(True), road_multiplier(None)) == (0.0, 1.0, 1.0)


def test_near_objects_is_a_sorted_local_report_that_can_omit_contributors() -> None:
    """Only objects within near_radius are listed, nearest first; a far object still adds score."""
    far = Object(label="pedestrian", score=0.9, box=(560.0, 200.0, 580.0, 330.0))  # ~3.3 heights
    mid = Object(label="car", score=0.8, box=(150.0, 250.0, 190.0, 330.0))
    near = near_objects_for(DEFECT_BOX, [far, mid, PEDESTRIAN], CFG)
    assert [n.label for n in near] == ["pedestrian", "car"]
    assert all(0.0 <= n.iou <= 1.0 and n.dist_px >= 0.0 for n in near)
    wide = dataclasses.replace(CFG, distance_scale=100.0)
    assert _score([far], cfg=wide) > _score([], cfg=wide)  # far is not in near_objects


def test_image_size_and_mask_shape_mismatch_raises() -> None:
    """A (width, height) vs (height, width) mix-up is caught at the boundary."""
    with pytest.raises(ValueError, match="road_mask shape"):
        fuse(
            frame_id="f",
            source="upload",
            image_size=(W, H),
            objects=[],
            defects=[_defect()],
            road_mask=np.ones((W, H), dtype=bool),
            config=CFG,
        )


def test_in_road_uses_continuous_area_for_fractional_boxes() -> None:
    """A fractional edge is weighted by overlap, not rounded outward (which would give 1/3)."""
    mask = np.zeros((1, 3), dtype=bool)
    mask[0, 1] = True  # only the middle pixel column is road
    box = (0.9, 0.0, 2.1, 1.0)  # covers 1.0 of its 1.2 width on road: 0.833
    assert in_road_state(box, mask, CFG) is True
    strict = dataclasses.replace(CFG, road_overlap_threshold=0.9)
    assert in_road_state(box, mask, strict) is False


def test_in_road_zero_area_boxes_are_unknown_even_on_an_empty_mask() -> None:
    """A zero-width or zero-height box has no area, so it is unknown, not off-road."""
    empty = np.zeros((10, 10), dtype=bool)
    assert in_road_state((0.5, 2.0, 0.5, 6.0), empty, CFG) is None
    assert in_road_state((2.0, 3.5, 6.0, 3.5), empty, CFG) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("which", ["defect", "object"])
def test_non_finite_box_is_rejected(bad: float, with_mask: bool, which: str) -> None:
    # +inf goes in x2 so the schema's ordering check still accepts the box.
    box = (0.0, 0.0, bad, 10.0) if bad == float("inf") else (bad, 0.0, 10.0, 10.0)
    objects = [Object(label="pedestrian", score=0.9, box=box)] if which == "object" else []
    defects = (
        [Defect(cls=1, name="D00", confidence=0.9, box=box, patch_row=0, patch_col=0)]
        if which == "defect"
        else []
    )
    with pytest.raises(ValueError, match="non-finite"):
        fuse(
            frame_id="f",
            source="upload",
            image_size=(W, H),
            objects=objects,
            defects=defects,
            road_mask=ALL_ROAD if with_mask else None,
        )
