"""Tile crop cache: each tile the model trains on is cropped once and saved as a native-size JPEG.

Purpose: training would otherwise decode a whole image (and Norway's are 4040x2035) for every
tile it reads. Cropping once makes a training read a small JPEG decode. Tiles are saved at native
resolution and not resized, so the augmentation that rescales them still sees every source pixel.

Safety: a cache built with other tile parameters or JPEG quality is refused (cache.json records
them); a crop whose box lies outside its image is an error, because PIL would silently pad it
with black; files are written to a temporary name and renamed, so an interrupted build never
leaves a half-written tile; a zero-byte file counts as missing. Cached files are RDD2022 pixels:
keep the folder under the gitignored data/ and never commit it.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from defects.tile_index import TileParams, TileRecord
from defects.voc import ALL

CACHE_VERSION = 1
MANIFEST_NAME = "cache.json"
DEFAULT_WORKERS = 8


def tile_cache_path(cache_dir: Path, tile: TileRecord) -> Path:
    """cache_dir / country / <image file stem>_r<row>c<col>.jpg"""
    country, slash, _ = tile.image_id.partition("/")
    stem = tile.image_id.rsplit("/", 1)[-1]
    return cache_dir / (country if slash else ALL) / f"{stem}_r{tile.row}c{tile.col}.jpg"


def missing_tiles(tiles: Sequence[TileRecord], cache_dir: Path) -> list[TileRecord]:
    """Tiles with no cached file, or an empty one."""
    return [
        t
        for t in tiles
        if not (path := tile_cache_path(cache_dir, t)).is_file() or path.stat().st_size == 0
    ]


def _check_manifest(cache_dir: Path, params: TileParams, quality: int) -> None:
    expected: dict[str, Any] = {
        "version": CACHE_VERSION,
        "params": asdict(params),
        "jpeg_quality": quality,
    }
    manifest = cache_dir / MANIFEST_NAME
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError(f"tile cache at {cache_dir} was built with different settings")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")


def _check_paths_are_unique(tiles: Sequence[TileRecord], cache_dir: Path) -> None:
    owner: dict[Path, str] = {}
    for tile in tiles:
        path = tile_cache_path(cache_dir, tile)
        if owner.setdefault(path, tile.tile_id) != tile.tile_id:
            raise ValueError(f"tile cache path collision: {owner[path]} and {tile.tile_id}")


def _write_image(
    image_id: str, source: Path, tiles: Sequence[TileRecord], cache_dir: Path, quality: int
) -> int:
    """Crop and save every listed tile of one image, decoding the image once."""
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError(f"{image_id}: unreadable image {source} ({exc})") from exc
    width, height = image.size
    for tile in tiles:
        x1, y1, x2, y2 = tile.box
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(
                f"{image_id}: tile {tile.tile_id} box {tile.box} is outside the "
                f"{width}x{height} image"
            )
        path = tile_cache_path(cache_dir, tile)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        image.crop(tile.box).save(temporary, format="JPEG", quality=quality)
        os.replace(temporary, path)
    return len(tiles)


def build_tile_cache(
    tiles: Sequence[TileRecord],
    image_paths: Mapping[str, Path],
    cache_dir: Path,
    params: TileParams,
    *,
    quality: int,
    workers: int = DEFAULT_WORKERS,
) -> int:
    """Write the tiles not yet cached; returns how many were written (0 on a repeat run)."""
    _check_manifest(cache_dir, params, quality)
    _check_paths_are_unique(tiles, cache_dir)
    by_image: dict[str, list[TileRecord]] = defaultdict(list)
    for tile in missing_tiles(tiles, cache_dir):
        by_image[tile.image_id].append(tile)
    sources = {image_id: image_paths[image_id] for image_id in by_image}  # unknown id: KeyError

    def write(image_id: str) -> int:
        return _write_image(image_id, sources[image_id], by_image[image_id], cache_dir, quality)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(write, by_image))
