"""tools/eval_fusion_synthetic.py: the patch bank, the near-object truth and the per-frame run."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.detector import GtBox
from eval.fusion_synthetic import Reported, polygon_mask
from fusion.config import load_fusion_config
from fusion.schema import DefectRecord, HazardRecord, NearObject
from tools.eval_fusion_synthetic import (
    PatchCrop,
    _pick_patch,
    build_patch_bank,
    make_expected_near,
    reported_from_record,
    run_frame,
)

XML = """<annotation><size><width>200</width><height>100</height></size>
<object><name>{code}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax>
<ymax>{y2}</ymax></bndbox></object></annotation>"""
PATCH_CFG = {
    "min_width_px": 20, "max_width_px": 40, "min_defect_side_px": 30, "context": 0.25,
    "feather": 0.0, "countries": ["Czech"], "min_center_y_frac": 0.0, "max_area_frac": 1.0,
    "aspect_range": [0.1, 10.0], "max_upscale": 1.4,
}  # fmt: skip


def _make_dataset(root: Path, name: str, country: str, box: tuple[int, int, int, int]) -> str:
    xmls = root / country / "train" / "annotations" / "xmls"
    images = root / country / "train" / "images"
    xmls.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = box
    (xmls / f"{name}.xml").write_text(XML.format(code="D40", x1=x1, y1=y1, x2=x2, y2=y2))
    Image.fromarray(np.full((100, 200, 3), 120, dtype=np.uint8)).save(images / f"{name}.jpg")
    return f"{country}/train/annotations/xmls/{name}"


def test_the_bank_crops_the_defect_with_context_and_tracks_where_it_sits(tmp_path: Path) -> None:
    item = _make_dataset(tmp_path, "Czech_000001", "Czech", (50, 30, 110, 70))
    bank = build_patch_bank(tmp_path, [item], PATCH_CFG, 5, random.Random(0))
    assert len(bank) == 1
    crop = bank[0]
    # 25% of a 60 x 40 box is 15 x 10 px of context on each side.
    assert crop.image.shape[:2] == (60, 90)
    assert crop.inner == (15, 10, 75, 50)


def test_the_bank_skips_other_countries_and_defects_that_are_too_small(tmp_path: Path) -> None:
    us = _make_dataset(tmp_path, "United_States_000001", "United_States", (50, 30, 110, 70))
    tiny = _make_dataset(tmp_path, "Czech_000002", "Czech", (50, 30, 60, 40))
    assert build_patch_bank(tmp_path, [us, tiny], PATCH_CFG, 5, random.Random(0)) == []


def test_the_bank_stops_at_the_requested_count(tmp_path: Path) -> None:
    items = [
        _make_dataset(tmp_path, f"Czech_00000{i}", "Czech", (50, 30, 110, 70)) for i in range(4)
    ]
    assert len(build_patch_bank(tmp_path, items, PATCH_CFG, 2, random.Random(0))) == 2


# ---------------------------------------------------------------------------- near objects


def test_expected_near_lists_only_nearby_visible_objects_a_coco_detector_can_name() -> None:
    cfg = load_fusion_config()
    defect = (500.0, 500.0, 560.0, 560.0)
    ground_truth = [
        GtBox("pedestrian", (510, 440, 540, 530), 4),  # standing beside the defect: near
        GtBox("car", (1400, 300, 1600, 500), 4),  # far away: not near
        GtBox("traffic_cone", (520, 500, 540, 540), 4),  # not in the COCO vocabulary
        GtBox("cyclist", (520, 440, 545, 530), 1),  # barely visible: a detector is not expected
        GtBox("child", (505, 460, 525, 530), 4),  # not a COCO label, but its family is pedestrian
    ]
    assert make_expected_near(ground_truth, cfg)(defect) == {"pedestrian"}


def test_reported_from_record_keeps_the_box_confidence_in_road_and_near_labels() -> None:
    record = HazardRecord(
        frame_id="f",
        source="nuscenes",
        image_size=(100, 50),
        defects=[
            DefectRecord(
                name="pothole",
                confidence=0.8,
                box=(1, 2, 30, 40),
                severity="unknown",
                in_road=None,
                near_objects=[NearObject(label="car", iou=0.0, dist_px=5.0)],
            )
        ],
        hazard_score=0.1,
    )
    assert reported_from_record(record) == [Reported((1, 2, 30, 40), 0.8, None, ("car",))]


# --------------------------------------------------------------------------------- frames

RUN_CFG: dict[str, Any] = {
    "patch": {"min_width_px": 20, "max_width_px": 40, "feather": 0.0, "max_upscale": 1.4},
    "placement": {
        "on_road_per_frame": 2, "off_road_per_frame": 1, "margin_px": 5, "min_y_frac": 0.0,
    },
}  # fmt: skip
FRAME = np.full((200, 400, 3), 30, dtype=np.uint8)
BANK = [PatchCrop(np.full((20, 30, 3), 255, dtype=np.uint8), (5.0, 4.0, 25.0, 16.0))]


def _everywhere(frame: np.ndarray) -> list[Reported]:
    return [Reported((0, 0, 400, 200), 0.9, True, ())]


def test_run_frame_pastes_one_patch_per_run_and_labels_on_and_off_road() -> None:
    seen: list[int] = []

    def analyze(frame: np.ndarray) -> list[Reported]:
        seen.append(int((frame != FRAME).any(axis=2).sum()))
        return _everywhere(frame)

    road = polygon_mask([(0, 100), (400, 100), (400, 200), (0, 200)], (400, 200))
    trials, skipped = run_frame(
        analyze, FRAME, road, [], BANK, lambda _box: set(), RUN_CFG, 0.5, random.Random(3)
    )
    assert skipped == 0 and [t.on_road for t in trials] == [True, True, False]
    assert len(seen) == 3 and all(count > 0 for count in seen)  # every run saw a pasted patch
    assert all(t.found for t in trials)
    # The truth box is the defect inside the crop, so it is smaller than the pasted box.
    assert all(10 <= t.truth_box[2] - t.truth_box[0] <= 40 for t in trials)


def test_run_frame_counts_a_placement_with_no_room_as_skipped() -> None:
    everything = np.ones((200, 400), dtype=bool)  # the whole frame is road: no off-road room
    trials, skipped = run_frame(
        _everywhere, FRAME, everything, [], BANK, lambda _box: set(), RUN_CFG, 0.5, random.Random(3)
    )
    assert skipped == 1 and [t.on_road for t in trials] == [True, True]


def test_run_frame_does_not_change_the_input_frame() -> None:
    before = FRAME.copy()
    road = polygon_mask([(0, 100), (400, 100), (400, 200), (0, 200)], (400, 200))
    run_frame(
        _everywhere, FRAME, road, [], BANK, lambda _box: set(), RUN_CFG, 0.5, random.Random(1)
    )
    assert np.array_equal(FRAME, before)


# ------------------------------------------------------------------------- patch selection


def test_the_bank_skips_defects_in_the_upper_half_huge_ones_and_slivers(tmp_path: Path) -> None:
    cfg = {**PATCH_CFG, "min_center_y_frac": 0.5, "max_area_frac": 0.4, "aspect_range": [0.4, 2.5]}
    high = _make_dataset(tmp_path, "Czech_000001", "Czech", (50, 5, 110, 40))  # in the sky half
    huge = _make_dataset(tmp_path, "Czech_000002", "Czech", (0, 40, 200, 100))  # most of the photo
    sliver = _make_dataset(tmp_path, "Czech_000003", "Czech", (10, 60, 190, 90))  # 6 : 1
    good = _make_dataset(tmp_path, "Czech_000004", "Czech", (50, 55, 110, 95))
    bank = build_patch_bank(tmp_path, [high, huge, sliver, good], cfg, 10, random.Random(0))
    assert len(bank) == 1


def test_a_patch_is_not_enlarged_beyond_the_limit_when_a_bigger_one_exists() -> None:
    small = PatchCrop(np.zeros((10, 20, 3), dtype=np.uint8), (0.0, 0.0, 20.0, 10.0))
    large = PatchCrop(np.zeros((50, 100, 3), dtype=np.uint8), (0.0, 0.0, 100.0, 50.0))
    for seed in range(10):
        assert _pick_patch([small, large], 60, 1.4, random.Random(seed)) is large


def test_when_no_patch_is_big_enough_the_largest_is_used() -> None:
    small = PatchCrop(np.zeros((10, 20, 3), dtype=np.uint8), (0.0, 0.0, 20.0, 10.0))
    large = PatchCrop(np.zeros((30, 40, 3), dtype=np.uint8), (0.0, 0.0, 40.0, 30.0))
    assert _pick_patch([small, large], 400, 1.4, random.Random(0)) is large
