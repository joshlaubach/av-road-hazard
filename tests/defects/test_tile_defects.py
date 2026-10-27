"""Turning per-tile class probabilities into Defect records, and the frame-level wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from defects.patches import Tile
from defects.tile_defects import (
    find_frame_defects,
    score_frame_tiles,
    tile_probabilities,
    tiles_to_defects,
)
from fusion.schema import Object

THRESHOLDS = (0.9, 0.75, 0.65, 0.75)  # D00, D10, D20, D40
STRICT = (0.9,) * 4  # above the 0.5 that the hand-built model gives its unused classes
TILES = [Tile(0, 0, (0, 0, 100, 100)), Tile(0, 1, (100, 0, 200, 100))]


def test_a_class_at_or_above_its_own_threshold_becomes_a_defect() -> None:
    probs = [[0.95, 0.10, 0.10, 0.10], [0.10, 0.10, 0.65, 0.10]]  # the second sits on its cut
    defects = tiles_to_defects(TILES, probs, THRESHOLDS)
    assert [(d.cls, d.name, d.patch_row, d.patch_col) for d in defects] == [
        (1, "D00", 0, 0),
        (3, "D20", 0, 1),
    ]
    assert [d.confidence for d in defects] == [0.95, 0.65]


def test_thresholds_are_per_class_not_one_global_value() -> None:
    # 0.80 clears D10 (0.75) but not D00 (0.90).
    defects = tiles_to_defects(TILES[:1], [[0.80, 0.80, 0.0, 0.0]], THRESHOLDS)
    assert [d.name for d in defects] == ["D10"]


def test_defect_box_is_the_tile_rectangle_in_frame_pixels() -> None:
    (defect,) = tiles_to_defects(TILES[1:], [[0.95, 0.0, 0.0, 0.0]], THRESHOLDS)
    assert defect.box == (100.0, 0.0, 200.0, 100.0)


def test_one_tile_can_carry_several_classes() -> None:
    defects = tiles_to_defects(TILES[:1], [[0.99, 0.0, 0.0, 0.80]], THRESHOLDS)
    assert [(d.cls, d.name) for d in defects] == [(1, "D00"), (4, "D40")]


def test_a_nan_probability_is_never_flagged() -> None:
    defects = tiles_to_defects(TILES[:1], [[float("nan"), 0.9, 0.0, 0.0]], THRESHOLDS)
    assert [d.name for d in defects] == ["D10"]


def test_no_tile_above_threshold_gives_no_defects() -> None:
    assert tiles_to_defects(TILES, [[0.5] * 4, [0.5] * 4], THRESHOLDS) == []


def test_one_probability_row_per_tile_is_required() -> None:
    with pytest.raises(ValueError, match="one probability row per tile"):
        tiles_to_defects(TILES, [[0.0] * 4], THRESHOLDS)


def test_a_row_needs_one_probability_per_class() -> None:
    with pytest.raises(ValueError, match="4 class probabilities"):
        tiles_to_defects(TILES[:1], [[0.9, 0.9]], THRESHOLDS)


def test_one_threshold_per_class_is_required() -> None:
    with pytest.raises(ValueError, match="4 thresholds"):
        tiles_to_defects(TILES[:1], [[0.9] * 4], (0.5, 0.5))


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_thresholds_must_be_probabilities(bad: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        tiles_to_defects(TILES[:1], [[0.9] * 4], (0.5, 0.5, 0.5, bad))


# --- frame wrapper, on a hand-built model: class 0 fires on a bright red channel ---------------

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

SIDE = 16
FRAME_W, FRAME_H = 200, 100


def red_detector(fires: int = 0) -> Any:
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
    )
    with torch.no_grad():
        model[2].weight.zero_()
        model[2].weight[fires, 0] = 10.0
        model[2].bias.zero_()
    return model


@pytest.fixture
def cfg() -> dict[str, Any]:
    return {
        "input_size": SIDE,
        "batch_size": 4,
        "num_workers": 0,
        "amp": "none",
        "channels_last": False,
        "tile_size": 100,
        "tile_overlap": 0.0,
    }


def frame_with_white(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)  # (height, width, channels)
    for x1, y1, x2, y2 in boxes:
        frame[y1:y2, x1:x2] = 255
    return frame


@pytest.mark.ml
def test_find_frame_defects_flags_only_the_bright_tile(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((100, 0, 200, 100))  # right tile white, left tile black
    defects = find_frame_defects(red_detector(), frame, [], cfg, thresholds=STRICT)
    assert [(d.name, d.patch_col, d.box) for d in defects] == [
        ("D00", 1, (100.0, 0.0, 200.0, 100.0))
    ]


@pytest.mark.ml
def test_find_frame_defects_skips_tiles_mostly_under_an_object(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))  # both tiles white
    car = Object(label="car", score=0.9, box=(100.0, 0.0, 200.0, 100.0))  # covers the right tile
    defects = find_frame_defects(red_detector(), frame, [car], cfg, thresholds=STRICT)
    assert [d.patch_col for d in defects] == [0]


@pytest.mark.ml
def test_find_frame_defects_with_every_tile_covered_returns_nothing(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    wall = Object(label="truck", score=0.9, box=(0.0, 0.0, 200.0, 100.0))
    assert find_frame_defects(red_detector(), frame, [wall], cfg, thresholds=STRICT) == []


@pytest.mark.ml
def test_find_frame_defects_scores_only_the_region(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    defects = find_frame_defects(
        red_detector(), frame, [], cfg, thresholds=STRICT, region=(100, 0, 200, 100)
    )
    assert [d.patch_col for d in defects] == [0]  # grid columns count from the region's left edge
    assert defects[0].box == (100.0, 0.0, 200.0, 100.0)


@pytest.mark.ml
def test_find_frame_defects_rejects_a_frame_that_is_not_rgb(cfg: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="RGB"):
        find_frame_defects(
            red_detector(), np.zeros((100, 200), dtype=np.uint8), [], cfg, thresholds=STRICT
        )


@pytest.mark.ml
@pytest.mark.parametrize(("fires", "name", "cls"), [(1, "D10", 2), (2, "D20", 3), (3, "D40", 4)])
def test_find_frame_defects_maps_each_model_output_to_its_class(
    cfg: dict[str, Any], fires: int, name: str, cls: int
) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    defects = find_frame_defects(red_detector(fires), frame, [], cfg, thresholds=STRICT)
    assert {(d.cls, d.name) for d in defects} == {(cls, name)}


@pytest.mark.ml
def test_find_frame_defects_keeps_tile_order_across_batches(cfg: dict[str, Any]) -> None:
    cfg = {**cfg, "tile_size": 50, "batch_size": 1}  # a 4 x 2 grid, one tile per forward pass
    frame = frame_with_white((50, 0, 100, 50), (150, 50, 200, 100))
    defects = find_frame_defects(red_detector(), frame, [], cfg, thresholds=STRICT)
    assert [(d.patch_row, d.patch_col) for d in defects] == [(0, 1), (1, 3)]


@pytest.mark.ml
def test_find_frame_defects_scores_an_overlapping_grid(cfg: dict[str, Any]) -> None:
    cfg = {**cfg, "tile_size": 100, "tile_overlap": 0.5}  # starts at x = 0, 50, 100
    frame = frame_with_white((0, 0, 200, 100))
    defects = find_frame_defects(red_detector(), frame, [], cfg, thresholds=STRICT)
    assert [d.box[0] for d in defects] == [0.0, 50.0, 100.0]


@pytest.mark.ml
def test_find_frame_defects_forwards_the_object_cover_limit(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    # A 20%-wide object, padded 5% a side, covers 22% of the right tile.
    car = Object(label="car", score=0.9, box=(180.0, 0.0, 200.0, 100.0))
    kept = find_frame_defects(red_detector(), frame, [car], cfg, thresholds=STRICT)
    strict = find_frame_defects(
        red_detector(), frame, [car], cfg, thresholds=STRICT, max_object_cover=0.2
    )
    assert [d.patch_col for d in kept] == [0, 1]
    assert [d.patch_col for d in strict] == [0]


@pytest.mark.ml
def test_find_frame_defects_pads_object_boxes_before_checking_cover(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    # 29 px wide -> 29% bare; padding grows its open left side by 1.45 px (rounded out to 2),
    # giving 31%, so the default 30% limit drops the tile.
    car = Object(label="car", score=0.9, box=(171.0, 0.0, 200.0, 100.0))
    defects = find_frame_defects(red_detector(), frame, [car], cfg, thresholds=STRICT)
    assert [d.patch_col for d in defects] == [0]


@pytest.mark.ml
def test_find_frame_defects_accepts_an_explicit_device(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((100, 0, 200, 100))
    defects = find_frame_defects(red_detector(), frame, [], cfg, thresholds=STRICT, device="cpu")
    assert [d.patch_col for d in defects] == [1]


@pytest.mark.ml
def test_find_frame_defects_rejects_a_non_uint8_frame(cfg: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="uint8"):
        find_frame_defects(
            red_detector(), np.zeros((100, 200, 3), dtype=np.float32), [], cfg, thresholds=STRICT
        )


# --- road mask: tiles that are not on the road are not scored ----------------------------------


def road_mask(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.bool_)  # (height, width), like the frame
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


@pytest.mark.ml
def test_find_frame_defects_without_a_road_mask_treats_the_road_as_unknown(
    cfg: dict[str, Any],
) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    defects = find_frame_defects(red_detector(), frame, [], cfg, thresholds=STRICT, road_mask=None)
    assert [d.patch_col for d in defects] == [0, 1]


@pytest.mark.ml
def test_find_frame_defects_drops_an_off_road_tile_the_model_scores_high(
    cfg: dict[str, Any],
) -> None:
    frame = frame_with_white((0, 0, 200, 100))  # both tiles white, so both score high
    defects = find_frame_defects(
        red_detector(), frame, [], cfg, thresholds=STRICT, road_mask=road_mask((0, 0, 100, 100))
    )
    assert [d.patch_col for d in defects] == [0]


@pytest.mark.ml
def test_find_frame_defects_applies_the_road_mask_and_object_boxes_together(
    cfg: dict[str, Any],
) -> None:
    cfg = {**cfg, "tile_size": 50}  # a 4 x 2 grid
    frame = frame_with_white((0, 0, 200, 100))
    car = Object(label="car", score=0.9, box=(0.0, 0.0, 50.0, 50.0))  # covers tile (0, 0)
    mask = road_mask((0, 0, 100, 100))  # tiles in columns 2 and 3 are off-road
    defects = find_frame_defects(
        red_detector(), frame, [car], cfg, thresholds=STRICT, road_mask=mask
    )
    assert [(d.patch_row, d.patch_col) for d in defects] == [(0, 1), (1, 0), (1, 1)]


@pytest.mark.ml
def test_find_frame_defects_rejects_a_transposed_road_mask(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    transposed = np.ones((FRAME_W, FRAME_H), dtype=np.bool_)
    with pytest.raises(ValueError, match=r"\(200, 100\).*\(100, 200\)"):
        find_frame_defects(red_detector(), frame, [], cfg, thresholds=STRICT, road_mask=transposed)


@pytest.mark.ml
def test_find_frame_defects_with_every_tile_off_road_returns_nothing(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    assert (
        find_frame_defects(red_detector(), frame, [], cfg, thresholds=STRICT, road_mask=road_mask())
        == []
    )


@pytest.mark.ml
def test_find_frame_defects_forwards_the_minimum_road_cover(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    mask = road_mask((0, 0, 100, 100), (100, 0, 140, 100))  # the right tile is 40% on road
    lenient = find_frame_defects(
        red_detector(), frame, [], cfg, thresholds=STRICT, road_mask=mask, min_road_cover=0.3
    )
    strict = find_frame_defects(
        red_detector(), frame, [], cfg, thresholds=STRICT, road_mask=mask, min_road_cover=0.5
    )
    assert [d.patch_col for d in lenient] == [0, 1]
    assert [d.patch_col for d in strict] == [0]


@pytest.mark.ml
def test_tile_probabilities_gives_one_row_per_tile_and_one_column_per_class(
    cfg: dict[str, Any],
) -> None:
    frame = frame_with_white((100, 0, 200, 100))  # right tile white, left tile black
    probs = tile_probabilities(red_detector(), frame, TILES, cfg)
    assert probs.shape == (2, 4)
    assert probs[1, 0] > 0.99 and probs[0, 0] < 0.01  # the white tile fires, the black one not
    assert np.allclose(probs[:, 1:], 0.5)


@pytest.mark.ml
def test_tile_probabilities_of_no_tiles_is_an_empty_table(cfg: dict[str, Any]) -> None:
    assert tile_probabilities(red_detector(), frame_with_white(), [], cfg).shape == (0, 4)


@pytest.mark.ml
def test_score_frame_tiles_says_why_a_tile_was_skipped(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    car = Object(label="car", score=0.9, box=(100.0, 0.0, 200.0, 100.0))
    left, right = score_frame_tiles(red_detector(), frame, [car], cfg)
    assert (left.status, right.status) == ("scored", "skipped_object")
    assert left.probabilities is not None and left.probabilities[0] > 0.99
    assert right.probabilities is None and right.object_cover > 0.9
    assert left.road_cover is None  # no road mask: the road is unknown, nothing is off-road


@pytest.mark.ml
def test_score_frame_tiles_marks_off_road_tiles_with_their_road_cover(cfg: dict[str, Any]) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    mask[:, 100:] = True  # road under the right tile only
    left, right = score_frame_tiles(red_detector(), frame, [], cfg, road_mask=mask)
    assert (left.status, right.status) == ("skipped_off_road", "scored")
    assert left.road_cover == 0.0 and right.road_cover == 1.0


@pytest.mark.ml
def test_the_scored_tiles_give_exactly_the_defects_find_frame_defects_gives(
    cfg: dict[str, Any],
) -> None:
    frame = frame_with_white((0, 0, 200, 100))
    car = Object(label="car", score=0.9, box=(100.0, 0.0, 200.0, 100.0))
    scored = [s for s in score_frame_tiles(red_detector(), frame, [car], cfg) if s.probabilities]
    from_scores = tiles_to_defects(
        [s.tile for s in scored], [list(s.probabilities or ()) for s in scored], STRICT
    )
    assert from_scores == find_frame_defects(red_detector(), frame, [car], cfg, thresholds=STRICT)
