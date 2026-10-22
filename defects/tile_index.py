"""Tile index: how every image is tiled, how each tile is labeled, and which tiles train the model.

Purpose: the classifier trains on tiles (defects/patches.py) labeled from RDD2022 boxes
(defects/tile_labels.py). This module fixes that data once, so a run is reproducible, and applies
the selection policy that keeps Norway's ~90 tiles per image from swamping training.

Compact storage: the grid is a pure function of the image size and the tile parameters, so the
index keeps only each image's size, its positive tiles with their labels, and its ignored tiles.
Every other tile is clean. Selection is per image and depends only on that image's tiles and the
seed, so adding or removing other images never changes it, and tiles inherit their image's split
because pools are built from lists of image ids.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from defects.labels import CODES
from defects.patches import IntBox, tile_grid
from defects.tile_labels import IGNORE, NEGATIVE, POSITIVE, Status, label_tile
from defects.voc import Box

INDEX_VERSION = 1
Cell = tuple[int, int]  # (row, col) in the tile grid


@dataclass(frozen=True)
class TileParams:
    """Everything the grid and the labels depend on; stored in the index so it cannot drift."""

    tile_size: int
    overlap: float
    positive_overlap: float
    negative_overlap: float

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> TileParams:
        return cls(
            tile_size=int(cfg["tile_size"]),
            overlap=float(cfg["tile_overlap"]),
            positive_overlap=float(cfg["positive_overlap"]),
            negative_overlap=float(cfg["negative_overlap"]),
        )


@dataclass(frozen=True)
class TileRecord:
    """One tile of one image: its place in the grid, its frame rectangle and its label."""

    image_id: str
    row: int
    col: int
    box: IntBox
    status: Status
    labels: tuple[int, ...]

    @property
    def tile_id(self) -> str:
        return f"{self.image_id}#r{self.row}c{self.col}"


@dataclass(frozen=True)
class ImageEntry:
    """What the index stores for one image; clean tiles are everything not listed here."""

    size: tuple[int, int]  # (width, height)
    positives: dict[Cell, tuple[int, ...]]
    ignored: frozenset[Cell]


@dataclass(frozen=True)
class TileIndex:
    params: TileParams
    images: dict[str, ImageEntry]


def label_image_tiles(
    image_id: str, image_size: tuple[int, int], boxes: Sequence[Box], params: TileParams
) -> list[TileRecord]:
    """Every tile of the image with its status; boxes include out-of-scope codes."""
    tiles = []
    for tile in tile_grid(image_size, params.tile_size, params.overlap):
        label = label_tile(
            tile.box,
            boxes,
            positive_overlap=params.positive_overlap,
            negative_overlap=params.negative_overlap,
        )
        tiles.append(TileRecord(image_id, tile.row, tile.col, tile.box, label.status, label.labels))
    return tiles


def entry_from_tiles(image_size: tuple[int, int], tiles: Sequence[TileRecord]) -> ImageEntry:
    return ImageEntry(
        size=image_size,
        positives={(t.row, t.col): t.labels for t in tiles if t.status == POSITIVE},
        ignored=frozenset((t.row, t.col) for t in tiles if t.status == IGNORE),
    )


def image_tiles(params: TileParams, image_id: str, entry: ImageEntry) -> list[TileRecord]:
    """Rebuild all tiles of an image from its entry; refuses an entry from other parameters."""
    grid = tile_grid(entry.size, params.tile_size, params.overlap)
    cells = {(t.row, t.col) for t in grid}
    if not (set(entry.positives) | entry.ignored) <= cells:
        raise ValueError(f"{image_id}: entry does not match the tile parameters {params}")
    clean = (0,) * len(CODES)
    tiles = []
    for tile in grid:
        cell = (tile.row, tile.col)
        if cell in entry.positives:
            status, labels = POSITIVE, entry.positives[cell]
        else:
            status, labels = (IGNORE if cell in entry.ignored else NEGATIVE), clean
        tiles.append(TileRecord(image_id, tile.row, tile.col, tile.box, status, labels))
    return tiles


def _rank(seed: int, tile: TileRecord) -> bytes:
    """A seeded, order-independent shuffle key: the same tile always ranks the same."""
    return hashlib.sha256(f"{seed}|{tile.tile_id}".encode()).digest()


def select_training_tiles(
    tiles: Sequence[TileRecord], *, clean_per_positive: float, min_clean: int, seed: int
) -> list[TileRecord]:
    """All positive tiles plus a capped, seeded sample of clean ones; ignored tiles are dropped.

    The clean count is ceil(clean_per_positive * positives), at least min_clean, at most the
    number available, so an image with no defect still contributes min_clean clean tiles.
    """
    if clean_per_positive < 0 or min_clean < 0:
        raise ValueError("clean_per_positive and min_clean_per_image must not be negative")
    positives = [t for t in tiles if t.status == POSITIVE]
    clean = sorted((t for t in tiles if t.status == NEGATIVE), key=lambda t: _rank(seed, t))
    keep = min(len(clean), max(min_clean, math.ceil(clean_per_positive * len(positives))))
    return sorted([*positives, *clean[:keep]], key=lambda t: (t.row, t.col))


def evaluation_tiles(tiles: Sequence[TileRecord]) -> list[TileRecord]:
    """Every positive and clean tile, uncapped, so scores reflect the natural mix."""
    return [t for t in tiles if t.status != IGNORE]


def _tiles_of(index: TileIndex, image_id: str) -> list[TileRecord]:
    try:
        entry = index.images[image_id]
    except KeyError:
        raise KeyError(f"{image_id} is not in the tile index") from None
    return image_tiles(index.params, image_id, entry)


def training_pool(
    index: TileIndex,
    image_ids: Sequence[str],
    *,
    clean_per_positive: float,
    min_clean: int,
    seed: int,
) -> list[TileRecord]:
    """Training tiles for exactly these images, in the order given."""
    return [
        tile
        for image_id in image_ids
        for tile in select_training_tiles(
            _tiles_of(index, image_id),
            clean_per_positive=clean_per_positive,
            min_clean=min_clean,
            seed=seed,
        )
    ]


def evaluation_pool(index: TileIndex, image_ids: Sequence[str]) -> list[TileRecord]:
    """Validation or held-out tiles for exactly these images, uncapped."""
    return [tile for image_id in image_ids for tile in evaluation_tiles(_tiles_of(index, image_id))]


def write_tile_index(index: TileIndex, path: Path) -> None:
    images = {
        image_id: {
            "size": list(entry.size),
            "positives": [[r, c, *labels] for (r, c), labels in sorted(entry.positives.items())],
            "ignored": [[r, c] for r, c in sorted(entry.ignored)],
        }
        for image_id, entry in index.images.items()
    }
    payload = {"version": INDEX_VERSION, "params": asdict(index.params), "images": images}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_tile_index(path: Path) -> TileIndex:
    payload = json.loads(path.read_text())
    if payload.get("version") != INDEX_VERSION:
        raise ValueError(f"Unsupported tile index version {payload.get('version')!r}")
    images = {
        image_id: ImageEntry(
            size=(raw["size"][0], raw["size"][1]),
            positives={(r, c): tuple(labels) for r, c, *labels in raw["positives"]},
            ignored=frozenset((r, c) for r, c in raw["ignored"]),
        )
        for image_id, raw in payload["images"].items()
    }
    return TileIndex(TileParams(**payload["params"]), images)
