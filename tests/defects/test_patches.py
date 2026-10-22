"""Tile geometry on generated images and plain numbers; no dataset image is read or committed."""

from __future__ import annotations

import pytest
from PIL import Image

from defects.patches import Tile, crop_tile, frame_to_tile_box, tile_grid, tile_to_frame_box

INPUT = 40  # stand-in for the classifier's 380 px input


def starts(tiles: list[Tile], axis: int) -> list[int]:
    return sorted({t.box[axis] for t in tiles})


def test_exact_fit_without_overlap() -> None:
    tiles = tile_grid((1000, 500), 500, 0.0)
    assert [(t.row, t.col, t.box) for t in tiles] == [
        (0, 0, (0, 0, 500, 500)),
        (0, 1, (500, 0, 1000, 500)),
    ]


def test_overlap_sets_stride_and_last_tile_lands_on_the_edge() -> None:
    tiles = tile_grid((1000, 500), 500, 0.5)
    assert starts(tiles, 0) == [0, 250, 500]
    assert max(t.box[2] for t in tiles) == 1000


def test_edge_tile_is_shifted_back_instead_of_running_past_the_image() -> None:
    tiles = tile_grid((600, 600), 512, 0.2)
    assert [t.box for t in tiles if t.row == 0] == [(0, 0, 512, 512), (88, 0, 600, 512)]
    assert all(t.box[0] >= 0 and t.box[2] <= 600 and t.box[3] <= 600 for t in tiles)


def test_width_and_height_are_not_swapped() -> None:
    """A 1600x900 frame has more columns than rows; a swapped (height, width) would flip that."""
    tiles = tile_grid((1600, 900), 400, 0.0)
    assert max(t.col for t in tiles) + 1 == 4
    assert max(t.row for t in tiles) + 1 == 3
    assert max(t.box[2] for t in tiles) == 1600
    assert max(t.box[3] for t in tiles) == 900


def test_rows_and_columns_are_consecutive_from_zero_in_reading_order() -> None:
    tiles = tile_grid((1600, 900), 400, 0.25)
    assert [(t.row, t.col) for t in tiles] == sorted((t.row, t.col) for t in tiles)
    assert {t.row for t in tiles} == set(range(max(t.row for t in tiles) + 1))
    assert {t.col for t in tiles} == set(range(max(t.col for t in tiles) + 1))
    assert len({(t.row, t.col) for t in tiles}) == len(tiles)


def test_every_pixel_is_covered_and_tiles_are_square_and_equal() -> None:
    width, height = 130, 70
    tiles = tile_grid((width, height), 32, 0.3)
    assert len({t.box[2] - t.box[0] for t in tiles} | {t.box[3] - t.box[1] for t in tiles}) == 1
    covered = {
        (x, y) for t in tiles for x in range(t.box[0], t.box[2]) for y in range(t.box[1], t.box[3])
    }
    assert covered == {(x, y) for x in range(width) for y in range(height)}


def test_tile_larger_than_the_image_shrinks_to_the_shorter_side() -> None:
    assert [t.box for t in tile_grid((300, 200), 512, 0.5)] == [
        (0, 0, 200, 200),
        (100, 0, 300, 200),
    ]


def test_region_restricts_tiles_and_restarts_the_grid_index() -> None:
    """The bottom third of a 1600x900 frame: tiles stay inside it and row 0 is its first row."""
    region = (0, 600, 1600, 900)
    tiles = tile_grid((1600, 900), 380, 0.5, region=region)
    assert {t.row for t in tiles} == {0}
    assert all(t.box[1] >= 600 and t.box[3] <= 900 for t in tiles)
    assert all(t.box[3] - t.box[1] == 300 for t in tiles)  # side clamps to the region height
    assert tiles[0].box == (0, 600, 300, 900)
    assert max(t.box[2] for t in tiles) == 1600


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image_size": (0, 100), "tile_size": 10, "overlap": 0.0}, "image_size"),
        ({"image_size": (100, 100), "tile_size": 0, "overlap": 0.0}, "tile_size"),
        ({"image_size": (100, 100), "tile_size": 10, "overlap": 1.0}, "overlap"),
        ({"image_size": (100, 100), "tile_size": 10, "overlap": -0.1}, "overlap"),
        (
            {"image_size": (100, 100), "tile_size": 10, "overlap": 0.0, "region": (0, 0, 101, 50)},
            "region",
        ),
        (
            {"image_size": (100, 100), "tile_size": 10, "overlap": 0.0, "region": (50, 0, 50, 50)},
            "region",
        ),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tile_grid(**kwargs)  # type: ignore[arg-type]


def test_tile_corners_map_to_the_tile_rectangle_in_the_frame() -> None:
    tile = Tile(row=1, col=2, box=(200, 300, 300, 400))
    assert tile_to_frame_box((0, 0, INPUT, INPUT), tile, INPUT) == (200, 300, 300, 400)
    assert tile_to_frame_box((INPUT / 2, 0, INPUT, INPUT / 2), tile, INPUT) == (250, 300, 300, 350)


def test_frame_and_tile_coordinates_round_trip() -> None:
    tile = Tile(row=0, col=1, box=(88, 10, 600, 522))
    frame_box = (100.0, 40.0, 340.5, 200.25)
    mapped = frame_to_tile_box(frame_box, tile, INPUT)
    back = tile_to_frame_box(mapped, tile, INPUT)
    assert back == pytest.approx(frame_box)


def test_crop_has_the_classifier_input_size() -> None:
    image = Image.new("RGB", (130, 70))
    assert crop_tile(image, Tile(0, 1, (30, 5, 94, 69)), INPUT).size == (INPUT, INPUT)


def test_marker_lands_where_the_mapping_says_it_does() -> None:
    """End to end: a white block in the frame shows up where frame_to_tile_box says in the crop."""
    frame = Image.new("L", (160, 90), 0)
    block = (70, 30, 90, 50)  # frame coordinates, 20x20
    frame.paste(255, block)
    tile = Tile(row=0, col=1, box=(50, 10, 110, 70))  # 60x60, contains the block
    crop = crop_tile(frame, tile, INPUT)
    mapped = frame_to_tile_box((70.0, 30.0, 90.0, 50.0), tile, INPUT)
    x1, y1, x2, y2 = (round(v) for v in mapped)
    assert (x1, y1, x2, y2) == (13, 13, 27, 27)
    assert crop.getpixel(((x1 + x2) // 2, (y1 + y2) // 2)) == 255
    assert crop.getpixel((x1 - 4, (y1 + y2) // 2)) == 0
    assert crop.getpixel(((x1 + x2) // 2, y2 + 4)) == 0
