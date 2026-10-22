"""Tile index and selection policy on synthetic boxes and generated images only."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import yaml
from PIL import Image

from defects.tile_index import (
    TileIndex,
    TileParams,
    TileRecord,
    entry_from_tiles,
    evaluation_pool,
    image_tiles,
    label_image_tiles,
    read_tile_index,
    select_training_tiles,
    training_pool,
    write_tile_index,
)
from defects.tile_labels import IGNORE, NEGATIVE, POSITIVE
from defects.voc import Box
from tools.make_tile_index import main

PARAMS = TileParams(tile_size=50, overlap=0.0, positive_overlap=0.25, negative_overlap=0.05)
SIZE = (100, 100)  # four 50 px tiles: (0,0) (0,1) (1,0) (1,1) as (row, col)
D40_TOP_LEFT: Box = ("D40", 10, 10, 40, 40)


def tiles_of(boxes: list[Box], image_id: str = "Czech/a") -> list[TileRecord]:
    return label_image_tiles(image_id, SIZE, boxes, PARAMS)


def status_by_cell(tiles: list[TileRecord]) -> dict[tuple[int, int], str]:
    return {(t.row, t.col): t.status for t in tiles}


def synthetic_tiles(positives: int, clean: int, ignored: int = 0) -> list[TileRecord]:
    """Tiles of one image with chosen status counts, for selection tests."""
    out = []
    for i in range(positives + clean + ignored):
        status = POSITIVE if i < positives else NEGATIVE if i < positives + clean else IGNORE
        labels = (1, 0, 0, 0) if status == POSITIVE else (0, 0, 0, 0)
        out.append(TileRecord("Czech/a", 0, i, (i, 0, i + 1, 1), status, labels))
    return out


def test_tiles_are_labeled_from_their_own_boxes_and_carry_their_position() -> None:
    tiles = tiles_of([D40_TOP_LEFT])
    assert status_by_cell(tiles) == {
        (0, 0): POSITIVE,
        (0, 1): NEGATIVE,
        (1, 0): NEGATIVE,
        (1, 1): NEGATIVE,
    }
    top_left = tiles[0]
    assert (top_left.box, top_left.labels) == ((0, 0, 50, 50), (0, 0, 0, 1))
    assert top_left.tile_id == "Czech/a#r0c0"


def test_out_of_scope_box_ignores_the_tile_it_touches() -> None:
    tiles = tiles_of([("D44", 60, 10, 90, 40)])
    assert status_by_cell(tiles)[(0, 1)] == IGNORE
    assert status_by_cell(tiles)[(0, 0)] == NEGATIVE


def test_entry_regenerates_exactly_the_tiles_it_came_from() -> None:
    tiles = tiles_of([D40_TOP_LEFT, ("D44", 60, 10, 90, 40), ("D00", 0, 60, 100, 90)])
    entry = entry_from_tiles(SIZE, tiles)
    assert image_tiles(PARAMS, "Czech/a", entry) == tiles
    assert set(entry.ignored) == {(0, 1)}
    assert entry.positives[(0, 0)] == (0, 0, 0, 1)


def test_all_clean_image_stores_nothing_but_its_size() -> None:
    entry = entry_from_tiles(SIZE, tiles_of([]))
    assert entry.positives == {} and not entry.ignored


@pytest.mark.parametrize(
    ("positives", "clean", "ratio", "kept_clean"),
    [
        (3, 20, 1.0, 3),  # one clean per positive
        (3, 20, 0.5, 2),  # ceil(1.5)
        (3, 2, 1.0, 2),  # never more than exist
        (0, 5, 1.0, 1),  # an image with no defect still contributes the minimum
        (0, 0, 1.0, 0),  # nothing to take
        (4, 20, 0.0, 1),  # ratio 0 leaves only the per-image minimum
    ],
)
def test_clean_tiles_are_capped_by_the_positive_count_with_a_minimum(
    positives: int, clean: int, ratio: float, kept_clean: int
) -> None:
    chosen = select_training_tiles(
        synthetic_tiles(positives, clean), clean_per_positive=ratio, min_clean=1, seed=42
    )
    assert sum(t.status == POSITIVE for t in chosen) == positives
    assert sum(t.status == NEGATIVE for t in chosen) == kept_clean


def test_ignored_tiles_never_enter_training_or_evaluation() -> None:
    tiles = synthetic_tiles(2, 6, ignored=4)
    chosen = select_training_tiles(tiles, clean_per_positive=1.0, min_clean=1, seed=42)
    assert all(t.status != IGNORE for t in chosen)


def test_selection_is_deterministic_and_independent_of_input_order() -> None:
    tiles = synthetic_tiles(3, 40)
    first = select_training_tiles(tiles, clean_per_positive=1.0, min_clean=1, seed=42)
    shuffled = tiles[:]
    random.Random(1).shuffle(shuffled)
    again = select_training_tiles(shuffled, clean_per_positive=1.0, min_clean=1, seed=42)
    assert first == again
    other = select_training_tiles(tiles, clean_per_positive=1.0, min_clean=1, seed=7)
    assert {t.tile_id for t in other} != {t.tile_id for t in first}


def test_evaluation_keeps_every_clean_tile_at_the_natural_mix() -> None:
    """One positive, one ignored (out-of-scope box) and two clean tiles: evaluation drops only the
    ignored one and does not cap the clean ones the way training does."""
    tiles = tiles_of([D40_TOP_LEFT, ("D44", 60, 10, 90, 40)])
    index = TileIndex(PARAMS, {"Czech/a": entry_from_tiles(SIZE, tiles)})
    held = evaluation_pool(index, ["Czech/a"])
    assert status_by_cell(held) == {(0, 0): POSITIVE, (1, 0): NEGATIVE, (1, 1): NEGATIVE}
    train = training_pool(index, ["Czech/a"], clean_per_positive=0.5, min_clean=1, seed=1)
    assert sum(t.status == NEGATIVE for t in train) == 1  # capped, unlike evaluation
    assert evaluation_pool(TileIndex(PARAMS, {}), []) == []


def test_pools_contain_only_tiles_of_the_requested_images() -> None:
    """Splits are per image, so a pool built from one split's ids can never see another's tiles."""
    index = TileIndex(
        PARAMS,
        {
            f"Czech/{name}": entry_from_tiles(SIZE, tiles_of([D40_TOP_LEFT], f"Czech/{name}"))
            for name in "abc"
        },
    )
    train = training_pool(
        index, ["Czech/a", "Czech/b"], clean_per_positive=1.0, min_clean=1, seed=1
    )
    assert {t.image_id for t in train} == {"Czech/a", "Czech/b"}
    held = evaluation_pool(index, ["Czech/c"])
    assert {t.image_id for t in held} == {"Czech/c"}
    assert len(held) == 4  # one positive + three clean, none ignored


def test_unknown_image_id_is_an_error_not_a_silent_skip() -> None:
    index = TileIndex(PARAMS, {})
    with pytest.raises(KeyError, match="Czech/missing"):
        evaluation_pool(index, ["Czech/missing"])


def test_selection_for_one_image_does_not_depend_on_other_images() -> None:
    def pool(names: str) -> set[str]:
        index = TileIndex(
            PARAMS,
            {
                f"Czech/{n}": entry_from_tiles(SIZE, tiles_of([D40_TOP_LEFT], f"Czech/{n}"))
                for n in names
            },
        )
        chosen = training_pool(
            index, [f"Czech/{n}" for n in names], clean_per_positive=1.0, min_clean=1, seed=3
        )
        return {t.tile_id for t in chosen if t.image_id == "Czech/a"}

    assert pool("a") == pool("abc")


def test_index_file_round_trips_and_is_byte_stable(tmp_path: Path) -> None:
    index = TileIndex(
        PARAMS,
        {"Czech/a": entry_from_tiles(SIZE, tiles_of([D40_TOP_LEFT, ("D44", 60, 10, 90, 40)]))},
    )
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    write_tile_index(index, first)
    write_tile_index(read_tile_index(first), second)
    assert first.read_bytes() == second.read_bytes()
    assert read_tile_index(first) == index


def test_index_with_an_unknown_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    write_tile_index(TileIndex(PARAMS, {}), path)
    data = json.loads(path.read_text())
    data["version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="version"):
        read_tile_index(path)


@pytest.mark.parametrize(
    ("ratio", "minimum"),
    [(-0.1, 1), (1.0, -1)],
)
def test_invalid_selection_arguments_are_rejected(ratio: float, minimum: int) -> None:
    with pytest.raises(ValueError, match="clean"):
        select_training_tiles(
            synthetic_tiles(1, 1), clean_per_positive=ratio, min_clean=minimum, seed=1
        )


def test_training_config_has_valid_tile_keys() -> None:
    cfg = yaml.safe_load(Path("configs/defects_train.yaml").read_text())
    params = TileParams.from_config(cfg)
    assert (params.tile_size, params.overlap) == (384, 0.2)
    assert cfg["clean_per_positive"] == 1.0 and cfg["min_clean_per_image"] == 1


def test_cli_builds_an_index_from_annotations_and_images(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "Czech/train/annotations/xmls").mkdir(parents=True)
    (root / "Czech/train/images").mkdir(parents=True)
    xml = (
        "<annotation><size><width>100</width><height>100</height></size>"
        "<object><name>D40</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>40</xmax>"
        "<ymax>40</ymax></bndbox></object></annotation>"
    )
    (root / "Czech/train/annotations/xmls/a.xml").write_text(xml)
    Image.new("RGB", (100, 100)).save(root / "Czech/train/images/a.png")
    image_id = "Czech/train/annotations/xmls/a"
    sample_index = tmp_path / "sample_index.json"
    sample_index.write_text(
        json.dumps(
            [{"id": image_id, "image_path": "raw/Czech/train/images/a.png", "labels": [0, 0, 0, 1]}]
        )
    )
    config = tmp_path / "cfg.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "tile_size": 50,
                "tile_overlap": 0.0,
                "positive_overlap": 0.25,
                "negative_overlap": 0.05,
            }
        )
    )
    out = tmp_path / "tile_index.json"
    args = ["--root", str(root), "--sample-index", str(sample_index), "--config", str(config)]
    assert main([*args, "--out", str(out)]) == 0
    index = read_tile_index(out)
    assert index.params == PARAMS
    tiles = image_tiles(index.params, image_id, index.images[image_id])
    assert status_by_cell(tiles)[(0, 0)] == POSITIVE
    assert len(tiles) == 4
