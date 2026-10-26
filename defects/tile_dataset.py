"""Tile training data: the pools of tiles a fold trains and validates on, and a dataset over them.

Only the training pool (every positive tile plus a capped, seeded sample of clean ones, see
defects/tile_index.py) is cached on disk, so validation during training uses the same capped pool
for its images. That mix is roughly balanced; natural-mix scores come from scoring whole images
tile by tile at the end (defects/tile_predict.py), not from these pools.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence, Set
from pathlib import Path
from typing import Any

from PIL import Image

from defects.dataset import DatasetBase, image_transform
from defects.labels import CODES
from defects.tile_cache import missing_tiles, tile_cache_path
from defects.tile_index import TileIndex, TileRecord, image_tiles, training_pool
from defects.tile_labels import NEGATIVE


class TileDataset(DatasetBase):
    """Cached native-size tile crops through the same transform as whole images."""

    def __init__(
        self, tiles: Sequence[TileRecord], cache_dir: Path, cfg: dict[str, Any], *, training: bool
    ) -> None:
        self.tiles = list(tiles)
        self.cache_dir = cache_dir
        self.transform = image_transform(cfg, training=training)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        tile = self.tiles[index]
        with Image.open(tile_cache_path(self.cache_dir, tile)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(tile.labels, dtype=torch.float32)


def tile_pools(
    index: TileIndex,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    cfg: dict[str, Any],
    *,
    seed: int,
) -> tuple[list[TileRecord], list[TileRecord]]:
    """Training and validation tiles for a fold; tiles inherit their image's split."""

    def pool(ids: Sequence[str]) -> list[TileRecord]:
        return training_pool(
            index,
            ids,
            clean_per_positive=float(cfg["clean_per_positive"]),
            min_clean=int(cfg["min_clean_per_image"]),
            seed=seed,
        )

    return pool(train_ids), pool(val_ids)


def _entry(index_in_file: int, row: object) -> tuple[str, int, int]:
    """(image_id, row, col) of one file entry, or a ValueError naming the entry."""
    where = f"extra negatives entry {index_in_file}"
    if not isinstance(row, dict):
        raise ValueError(f"{where}: expected an object with image_id, row and col, got {row!r}")
    image_id, r, c = row.get("image_id"), row.get("row"), row.get("col")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError(f"{where}: image_id must be a non-empty string")
    for name, value in (("row", r), ("col", c)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: {name} must be an integer, got {value!r}")
    return image_id, int(r), int(c)  # type: ignore[arg-type]


def extra_negative_tiles(
    index: TileIndex, path: Path, allowed_ids: Set[str], *, repeat: int
) -> list[TileRecord]:
    """Annotator-confirmed clean tiles, each repeated `repeat` times, for training only.

    The file is a JSON list of {"image_id", "row", "col"}. Tiles of images outside allowed_ids
    (a fold's validation or held-out images, or the final test set) are skipped, so they cannot
    leak into training. Each listed tile is used once however often it is listed. An image missing
    from the index, a tile outside its grid, or a tile the index labels positive or ignored is an
    error: only clean negatives belong here.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, got {repeat}")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path} must hold a JSON list of tiles, got {type(rows).__name__}")
    wanted = dict.fromkeys(_entry(n, row) for n, row in enumerate(rows))  # unique, in file order
    grids: dict[str, dict[tuple[int, int], TileRecord]] = {}
    tiles: list[TileRecord] = []
    for image_id, r, c in wanted:
        if image_id not in index.images:
            raise ValueError(f"{image_id} is not in the tile index")
        if image_id not in allowed_ids:
            continue
        if image_id not in grids:
            grid = image_tiles(index.params, image_id, index.images[image_id])
            grids[image_id] = {(t.row, t.col): t for t in grid}
        tile = grids[image_id].get((r, c))
        if tile is None:
            raise ValueError(f"{image_id} row {r} col {c} is not in the tile grid")
        if tile.status != NEGATIVE:
            raise ValueError(f"{image_id} row {r} col {c} is not a clean negative ({tile.status})")
        tiles.extend([tile] * repeat)
    return tiles


def check_cached(tiles: Sequence[TileRecord], cache_dir: Path) -> None:
    """Fail before training, not mid-epoch, if any tile has no cached crop."""
    missing = missing_tiles(tiles, cache_dir)
    if missing:
        raise ValueError(
            f"{len(missing)} tiles are missing from the tile cache at {cache_dir} "
            "(run tools.build_tile_cache)"
        )


def image_scores(
    tiles: Sequence[TileRecord], probabilities: Sequence[Sequence[float]]
) -> dict[str, list[float]]:
    """One score per class for each image: the highest probability over its tiles."""
    if len(tiles) != len(probabilities):
        raise ValueError("need one probability row per tile")
    best: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(CODES))
    for tile, row in zip(tiles, probabilities, strict=True):
        if len(row) != len(CODES):
            raise ValueError(f"need {len(CODES)} class probabilities per tile, got {len(row)}")
        scores = best[tile.image_id]
        for k, value in enumerate(row):
            scores[k] = max(scores[k], float(value))
    return dict(best)
