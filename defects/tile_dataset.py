"""Tile training data: the pools of tiles a fold trains and validates on, and a dataset over them.

Only the training pool (every positive tile plus a capped, seeded sample of clean ones, see
defects/tile_index.py) is cached on disk, so validation during training uses the same capped pool
for its images. That mix is roughly balanced; natural-mix scores come from scoring whole images
tile by tile at the end (defects/tile_predict.py), not from these pools.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from defects.dataset import DatasetBase, image_transform
from defects.labels import CODES
from defects.tile_cache import missing_tiles, tile_cache_path
from defects.tile_index import TileIndex, TileRecord, training_pool


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
