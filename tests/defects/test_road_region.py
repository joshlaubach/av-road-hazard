"""Road coverage of tiles: how much of a tile lies on the road mask, and dropping off-road tiles."""

from __future__ import annotations

import numpy as np
import pytest

from defects.patches import Tile
from defects.road_region import DEFAULT_MIN_ROAD, drop_offroad_tiles, road_coverage

TILE = Tile(0, 0, (100, 100, 200, 200))  # 100 x 100 px
MASK_H, MASK_W = 300, 400


def mask_with_road(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((MASK_H, MASK_W), dtype=np.bool_)  # (height, width)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def test_a_tile_fully_on_road_has_coverage_one() -> None:
    assert road_coverage(TILE, np.ones((MASK_H, MASK_W), dtype=np.bool_)) == 1.0


def test_a_tile_with_no_road_has_coverage_zero() -> None:
    assert road_coverage(TILE, np.zeros((MASK_H, MASK_W), dtype=np.bool_)) == 0.0


def test_half_the_tile_on_road_gives_one_half() -> None:
    mask = mask_with_road((0, 0, 150, MASK_H))  # left half of the tile
    assert road_coverage(TILE, mask) == pytest.approx(0.5)


def test_road_outside_the_tile_does_not_count() -> None:
    assert road_coverage(TILE, mask_with_road((200, 0, 400, 300), (0, 0, 100, 300))) == 0.0


def test_the_mask_is_not_modified() -> None:
    mask = mask_with_road((0, 0, 150, MASK_H))
    before = mask.copy()
    road_coverage(TILE, mask)
    assert np.array_equal(mask, before)


@pytest.mark.parametrize(
    "box",
    [(-1, 0, 50, 50), (0, -5, 50, 50), (350, 0, 401, 50), (0, 250, 50, 301)],
)
def test_a_tile_not_inside_the_mask_raises(box: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="inside the mask"):
        road_coverage(Tile(0, 0, box), np.ones((MASK_H, MASK_W), dtype=np.bool_))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.int64])
def test_a_non_bool_mask_raises(dtype: type) -> None:
    with pytest.raises(ValueError, match="bool"):
        road_coverage(TILE, np.ones((MASK_H, MASK_W), dtype=dtype))


@pytest.mark.parametrize("shape", [(MASK_H,), (MASK_H, MASK_W, 1), ()])
def test_a_mask_that_is_not_2d_raises(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="2-D"):
        road_coverage(TILE, np.ones(shape, dtype=np.bool_))


def test_a_tile_exactly_on_the_boundary_is_kept() -> None:
    mask = mask_with_road((0, 0, 150, MASK_H))
    assert drop_offroad_tiles([TILE], mask, min_road=0.5) == [TILE]


def test_a_tile_below_min_road_is_dropped() -> None:
    mask = mask_with_road((0, 0, 149, MASK_H))  # 49% on road
    assert drop_offroad_tiles([TILE], mask, min_road=0.5) == []


def test_the_default_min_road_is_one_half() -> None:
    assert DEFAULT_MIN_ROAD == 0.5


def test_order_is_preserved() -> None:
    tiles = [
        Tile(0, 0, (0, 0, 100, 100)),
        Tile(0, 1, (100, 0, 200, 100)),
        Tile(0, 2, (200, 0, 300, 100)),
        Tile(0, 3, (300, 0, 400, 100)),
    ]
    mask = mask_with_road((300, 0, 400, 100), (0, 0, 100, 100), (200, 0, 300, 100))
    assert drop_offroad_tiles(tiles, mask) == [tiles[0], tiles[2], tiles[3]]


def test_no_tiles_gives_an_empty_list() -> None:
    assert drop_offroad_tiles([], np.ones((MASK_H, MASK_W), dtype=np.bool_)) == []


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_min_road_must_be_in_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="min_road"):
        drop_offroad_tiles([TILE], np.ones((MASK_H, MASK_W), dtype=np.bool_), min_road=bad)


@pytest.mark.parametrize("edge", [0.0, 1.0])
def test_min_road_accepts_both_ends(edge: float) -> None:
    mask = np.ones((MASK_H, MASK_W), dtype=np.bool_)
    assert drop_offroad_tiles([TILE], mask, min_road=edge) == [TILE]


def test_bad_mask_raises_even_with_no_tiles() -> None:
    with pytest.raises(ValueError, match="bool"):
        drop_offroad_tiles([], np.ones((MASK_H, MASK_W), dtype=np.uint8))
