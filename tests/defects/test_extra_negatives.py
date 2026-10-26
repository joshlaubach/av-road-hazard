"""Confirmed hard negatives: read from a file, restricted to a fold's training images, repeated."""

import json
from pathlib import Path

import pytest

from defects.tile_dataset import extra_negative_tiles
from defects.tile_index import TileIndex, TileParams, entry_from_tiles, label_image_tiles
from defects.tile_labels import NEGATIVE

PARAMS = TileParams(tile_size=16, overlap=0.0, positive_overlap=0.25, negative_overlap=0.05)


def _index(*image_ids: str) -> TileIndex:
    entries = {
        image_id: entry_from_tiles((32, 32), label_image_tiles(image_id, (32, 32), [], PARAMS))
        for image_id in image_ids
    }
    return TileIndex(PARAMS, entries)


def _file(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "negatives.json"
    path.write_text(json.dumps(rows))
    return path


def test_tiles_are_clean_negatives_with_the_grid_box(tmp_path: Path) -> None:
    path = _file(tmp_path, [{"image_id": "a", "row": 1, "col": 0}])

    tiles = extra_negative_tiles(_index("a"), path, {"a"}, repeat=1)

    assert len(tiles) == 1
    assert (tiles[0].image_id, tiles[0].row, tiles[0].col) == ("a", 1, 0)
    assert tiles[0].box == (0, 16, 16, 32)
    assert tiles[0].status == NEGATIVE
    assert tiles[0].labels == (0, 0, 0, 0)


def test_repeat_multiplies_the_tiles(tmp_path: Path) -> None:
    path = _file(tmp_path, [{"image_id": "a", "row": 0, "col": 0}])

    assert len(extra_negative_tiles(_index("a"), path, {"a"}, repeat=5)) == 5


def test_images_outside_the_allowed_set_are_skipped(tmp_path: Path) -> None:
    path = _file(
        tmp_path,
        [{"image_id": "a", "row": 0, "col": 0}, {"image_id": "val", "row": 0, "col": 0}],
    )

    tiles = extra_negative_tiles(_index("a", "val"), path, {"a"}, repeat=1)

    assert [t.image_id for t in tiles] == ["a"]


def test_a_tile_that_is_not_in_the_grid_is_an_error(tmp_path: Path) -> None:
    path = _file(tmp_path, [{"image_id": "a", "row": 9, "col": 9}])

    with pytest.raises(ValueError, match="not in the tile grid"):
        extra_negative_tiles(_index("a"), path, {"a"}, repeat=1)


@pytest.mark.parametrize("repeat", [0, -1])
def test_repeat_must_be_positive(tmp_path: Path, repeat: int) -> None:
    path = _file(tmp_path, [{"image_id": "a", "row": 0, "col": 0}])

    with pytest.raises(ValueError, match="repeat"):
        extra_negative_tiles(_index("a"), path, {"a"}, repeat=repeat)


# --- Review findings --------------------------------------------------------------------------


def _index_with_box(image_id: str) -> TileIndex:
    """One 32x32 image: tile (1,0) is a D00 positive, (0,1) a D10 positive, (0,0) is ignored."""
    boxes = [("D00", 0, 16, 16, 32), ("D10", 15, 0, 31, 16)]
    tiles = label_image_tiles(image_id, (32, 32), boxes, PARAMS)
    return TileIndex(PARAMS, {image_id: entry_from_tiles((32, 32), tiles)})


def test_a_listed_positive_tile_is_refused(tmp_path: Path) -> None:
    path = _file(tmp_path, [{"image_id": "a", "row": 1, "col": 0}])

    with pytest.raises(ValueError, match="not a clean negative"):
        extra_negative_tiles(_index_with_box("a"), path, {"a"}, repeat=1)


def test_a_listed_ignored_tile_is_refused(tmp_path: Path) -> None:
    index = _index_with_box("a")
    ignored = next(iter(index.images["a"].ignored))
    path = _file(tmp_path, [{"image_id": "a", "row": ignored[0], "col": ignored[1]}])

    with pytest.raises(ValueError, match="not a clean negative"):
        extra_negative_tiles(index, path, {"a"}, repeat=1)


@pytest.mark.parametrize(
    "row",
    [
        {"image_id": "a", "row": 0},
        {"image_id": "a", "row": "0", "col": 0},
        {"image_id": "a", "row": True, "col": 0},
        {"image_id": "a", "row": 0.5, "col": 0},
        "a",
    ],
)
def test_a_malformed_row_names_the_entry(tmp_path: Path, row: object) -> None:
    path = _file(tmp_path, [{"image_id": "a", "row": 1, "col": 0}, row])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="entry 1"):
        extra_negative_tiles(_index("a"), path, {"a"}, repeat=1)


def test_a_file_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "negatives.json"
    path.write_text('{"image_id": "a"}')

    with pytest.raises(ValueError, match="list"):
        extra_negative_tiles(_index("a"), path, {"a"}, repeat=1)


def test_duplicate_rows_count_once(tmp_path: Path) -> None:
    row = {"image_id": "a", "row": 1, "col": 0}
    path = _file(tmp_path, [row, row])

    assert len(extra_negative_tiles(_index("a"), path, {"a"}, repeat=3)) == 3


def test_an_unknown_image_is_a_clear_error_even_when_not_allowed(tmp_path: Path) -> None:
    path = _file(tmp_path, [{"image_id": "ghost", "row": 0, "col": 0}])

    with pytest.raises(ValueError, match="ghost is not in the tile index"):
        extra_negative_tiles(_index("a"), path, {"a"}, repeat=1)
