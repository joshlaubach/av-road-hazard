"""Object coverage of tiles: how much of a tile lies under detected objects, and tile dropping."""

from __future__ import annotations

import pytest

from defects.object_mask import drop_covered_tiles, object_coverage
from defects.patches import Tile

TILE = Tile(0, 0, (100, 100, 200, 200))  # 100 x 100 px


def test_no_objects_cover_nothing() -> None:
    assert object_coverage(TILE, []) == 0.0


def test_an_object_over_the_whole_tile_covers_all_of_it() -> None:
    assert object_coverage(TILE, [(0.0, 0.0, 1000.0, 1000.0)], pad=0.0) == 1.0


def test_half_the_tile_gives_one_half() -> None:
    assert object_coverage(TILE, [(0.0, 0.0, 150.0, 1000.0)], pad=0.0) == pytest.approx(0.5)


def test_overlapping_objects_are_counted_once() -> None:
    boxes = [
        (100.0, 100.0, 150.0, 200.0),
        (100.0, 100.0, 150.0, 200.0),
        (120.0, 100.0, 150.0, 200.0),
    ]
    assert object_coverage(TILE, boxes, pad=0.0) == pytest.approx(0.5)


def test_objects_outside_the_tile_cover_nothing() -> None:
    assert (
        object_coverage(TILE, [(300.0, 300.0, 400.0, 400.0), (0.0, 0.0, 99.0, 99.0)], pad=0.0)
        == 0.0
    )


def test_an_object_partly_outside_the_tile_is_clipped_to_it() -> None:
    # 50 px of the box's width is inside the tile; the rest lies beyond its right edge.
    assert object_coverage(TILE, [(150.0, 100.0, 400.0, 200.0)], pad=0.0) == pytest.approx(0.5)


def test_padding_grows_each_box_on_every_side_by_a_fraction_of_its_size() -> None:
    box = (140.0, 140.0, 160.0, 160.0)  # 20 x 20; pad 0.5 adds 10 px per side -> 40 x 40
    assert object_coverage(TILE, [box], pad=0.5) == pytest.approx(0.16)


def test_fractional_edges_round_outward_so_coverage_is_never_understated() -> None:
    # 10.2 px -> 11 px columns/rows touched; the 1 px strip is counted whole.
    assert object_coverage(TILE, [(100.0, 100.0, 110.2, 200.0)], pad=0.0) == pytest.approx(0.11)


def test_padding_scales_width_and_height_separately() -> None:
    box = (140.0, 130.0, 160.0, 170.0)  # 20 wide x 40 high; pad 0.5 -> 40 x 80
    assert object_coverage(TILE, [box], pad=0.5) == pytest.approx(0.32)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_box_coordinate_is_rejected_with_a_clear_message(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        object_coverage(TILE, [(0.0, 0.0, bad, 10.0)])


def test_a_flipped_box_is_rejected_instead_of_silently_shrinking() -> None:
    with pytest.raises(ValueError, match="x1<=x2 and y1<=y2"):
        object_coverage(TILE, [(150.0, 100.0, 120.0, 200.0)])


def test_drop_covered_tiles_rejects_a_bad_box_even_when_no_tile_overlaps_it() -> None:
    far_tile = Tile(0, 0, (0, 0, 10, 10))
    with pytest.raises(ValueError, match="non-finite"):
        drop_covered_tiles([far_tile], [(500.0, 500.0, float("nan"), 600.0)])


def test_negative_padding_is_rejected() -> None:
    with pytest.raises(ValueError, match="pad"):
        object_coverage(TILE, [], pad=-0.1)


def test_drop_covered_tiles_keeps_tiles_up_to_the_limit() -> None:
    clear = Tile(0, 0, (0, 0, 100, 100))
    edge = Tile(0, 1, (100, 0, 200, 100))  # exactly 30% covered
    heavy = Tile(0, 2, (200, 0, 300, 100))  # 100% covered
    boxes = [(100.0, 0.0, 130.0, 100.0), (200.0, 0.0, 300.0, 100.0)]
    kept = drop_covered_tiles([clear, edge, heavy], boxes, max_cover=0.3, pad=0.0)
    assert kept == [clear, edge]


def test_drop_covered_tiles_keeps_everything_when_there_are_no_objects() -> None:
    tiles = [Tile(0, 0, (0, 0, 10, 10)), Tile(0, 1, (10, 0, 20, 10))]
    assert drop_covered_tiles(tiles, [], max_cover=0.3) == tiles


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_max_cover_must_be_a_fraction(bad: float) -> None:
    with pytest.raises(ValueError, match="max_cover"):
        drop_covered_tiles([TILE], [], max_cover=bad)
