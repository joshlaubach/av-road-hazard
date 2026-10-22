"""Tile crop cache on generated images only; no dataset image is read or committed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageStat

from defects.splits import make_splits, write_splits
from defects.tile_cache import build_tile_cache, missing_tiles, tile_cache_path
from defects.tile_index import (
    TileIndex,
    TileParams,
    TileRecord,
    entry_from_tiles,
    label_image_tiles,
    write_tile_index,
)
from tools.build_tile_cache import main

PARAMS = TileParams(tile_size=50, overlap=0.0, positive_overlap=0.25, negative_overlap=0.05)
QUALITY = 95
# Quadrant (row, col) -> colour, so a wrong crop position or swapped axis shows as a wrong colour.
COLOURS = {
    (0, 0): (220, 20, 20),
    (0, 1): (20, 200, 20),
    (1, 0): (20, 20, 220),
    (1, 1): (240, 240, 240),
}
IMAGE_ID = "Czech/train/annotations/xmls/a"


def quadrant_image() -> Image.Image:
    image = Image.new("RGB", (100, 100))
    for (row, col), colour in COLOURS.items():
        image.paste(colour, (col * 50, row * 50, col * 50 + 50, row * 50 + 50))
    return image


def make_tiles(image_id: str = IMAGE_ID) -> list[TileRecord]:
    return label_image_tiles(image_id, (100, 100), [], PARAMS)


def save_source(tmp_path: Path, name: str = "a", image: Image.Image | None = None) -> Path:
    path = tmp_path / "src" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    (image or quadrant_image()).save(path)
    return path


def build(tmp_path: Path, tiles: list[TileRecord], paths: dict[str, Path]) -> int:
    return build_tile_cache(tiles, paths, tmp_path / "cache", PARAMS, quality=QUALITY, workers=2)


def test_path_is_country_folder_plus_image_stem_and_grid_cell() -> None:
    tile = make_tiles()[1]
    assert tile_cache_path(Path("cache"), tile) == Path("cache/Czech/a_r0c1.jpg")


def test_each_cached_tile_shows_the_part_of_the_frame_it_claims_to(tmp_path: Path) -> None:
    """A wrong crop position or a swapped width/height would show as the wrong colour."""
    tiles = make_tiles()
    assert build(tmp_path, tiles, {IMAGE_ID: save_source(tmp_path)}) == 4
    for tile in tiles:
        with Image.open(tile_cache_path(tmp_path / "cache", tile)) as crop:
            assert crop.size == (50, 50)
            mean = ImageStat.Stat(crop.convert("RGB")).mean
        assert mean == pytest.approx(COLOURS[(tile.row, tile.col)], abs=12)


def test_second_build_writes_nothing_and_a_missing_tile_is_rebuilt(tmp_path: Path) -> None:
    tiles, paths = make_tiles(), {IMAGE_ID: save_source(tmp_path)}
    assert build(tmp_path, tiles, paths) == 4
    first = tile_cache_path(tmp_path / "cache", tiles[0])
    stamp = first.stat().st_mtime_ns
    assert build(tmp_path, tiles, paths) == 0
    assert first.stat().st_mtime_ns == stamp
    first.unlink()
    assert build(tmp_path, tiles, paths) == 1
    assert first.is_file()


def test_empty_leftover_file_counts_as_missing_and_is_rebuilt(tmp_path: Path) -> None:
    tiles, paths = make_tiles(), {IMAGE_ID: save_source(tmp_path)}
    build(tmp_path, tiles, paths)
    target = tile_cache_path(tmp_path / "cache", tiles[2])
    target.write_bytes(b"")
    assert missing_tiles(tiles, tmp_path / "cache") == [tiles[2]]
    assert build(tmp_path, tiles, paths) == 1
    assert missing_tiles(tiles, tmp_path / "cache") == []


def test_only_the_requested_tiles_are_written(tmp_path: Path) -> None:
    tiles = make_tiles()
    assert build(tmp_path, tiles[:1], {IMAGE_ID: save_source(tmp_path)}) == 1
    assert missing_tiles(tiles, tmp_path / "cache") == tiles[1:]


def test_cache_for_other_tile_parameters_or_quality_is_refused(tmp_path: Path) -> None:
    tiles, paths = make_tiles(), {IMAGE_ID: save_source(tmp_path)}
    build(tmp_path, tiles, paths)
    other = TileParams(tile_size=40, overlap=0.0, positive_overlap=0.25, negative_overlap=0.05)
    with pytest.raises(ValueError, match="cache"):
        build_tile_cache(tiles, paths, tmp_path / "cache", other, quality=QUALITY, workers=1)
    with pytest.raises(ValueError, match="cache"):
        build_tile_cache(tiles, paths, tmp_path / "cache", PARAMS, quality=80, workers=1)


def test_unreadable_source_image_is_an_error_naming_the_image(tmp_path: Path) -> None:
    broken = tmp_path / "src" / "a.png"
    broken.parent.mkdir()
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match=IMAGE_ID):
        build(tmp_path, make_tiles(), {IMAGE_ID: broken})


def test_source_smaller_than_the_index_says_is_an_error_not_black_padding(tmp_path: Path) -> None:
    small = save_source(tmp_path, image=Image.new("RGB", (60, 60)))
    with pytest.raises(ValueError, match="outside"):
        build(tmp_path, make_tiles(), {IMAGE_ID: small})


def test_unknown_image_in_the_tile_list_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match=IMAGE_ID):
        build(tmp_path, make_tiles(), {})


def test_two_images_with_the_same_file_name_cannot_share_a_cache_file(tmp_path: Path) -> None:
    other_id = "Czech/val/annotations/xmls/a"
    tiles = [*make_tiles(), *make_tiles(other_id)]
    paths = {IMAGE_ID: save_source(tmp_path), other_id: save_source(tmp_path, "b")}
    with pytest.raises(ValueError, match="collision"):
        build(tmp_path, tiles, paths)


def test_cli_caches_training_tiles_of_non_test_images_only(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    ids, rows, entries = [], [], {}
    for n in range(20):
        image_id = f"Czech/train/annotations/xmls/img{n:02d}"
        path = root / "images" / f"img{n:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        quadrant_image().save(path)
        ids.append(image_id)
        rows.append({"id": image_id, "image_path": f"raw/images/img{n:02d}.png", "labels": [0] * 4})
        entries[image_id] = entry_from_tiles((100, 100), make_tiles(image_id))
    sample_index = tmp_path / "sample_index.json"
    sample_index.write_text(json.dumps(rows))
    tile_index = tmp_path / "tile_index.json"
    write_tile_index(TileIndex(PARAMS, entries), tile_index)
    splits = make_splits(ids, lambda i: i, test_fraction=0.15, n_folds=5, val_fraction=0.1, seed=42)
    splits_file = tmp_path / "splits.json"
    write_splits(splits, splits_file)
    config = tmp_path / "cfg.yaml"
    config.write_text(
        yaml.safe_dump(
            {"clean_per_positive": 1.0, "min_clean_per_image": 1, "cache_jpeg_quality": 95}
        )
    )
    cache = tmp_path / "cache"
    argv = ["--sample-index", str(sample_index), "--tile-index", str(tile_index)]
    argv += ["--splits", str(splits_file), "--config", str(config), "--cache-dir", str(cache)]
    assert main(argv) == 0
    names = [p.name for p in cache.rglob("*.jpg")]

    def stem(image_id: str) -> str:
        return image_id.rsplit("/", 1)[1] + "_r"

    non_test = {i for part in splits.folds[0] for i in part}
    assert non_test and not non_test & set(splits.test)
    for image_id in splits.test:
        assert not any(name.startswith(stem(image_id)) for name in names)
    for image_id in non_test:  # every image is all clean, so min_clean_per_image keeps one tile
        assert sum(name.startswith(stem(image_id)) for name in names) == 1
    assert json.loads((cache / "cache.json").read_text())["jpeg_quality"] == 95
