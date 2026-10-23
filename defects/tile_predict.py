"""Score whole images tile by tile, the way the classifier is used on a driving frame.

Every tile of an image's grid (defects/patches.py) is cropped at native resolution and goes
through the same transform as the cached training tiles; the image's score for a class is the
highest probability over its tiles. Unlike the training pools, nothing is capped or skipped, so the
mix of clean and defective tiles is the natural one. No tile index or cache is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from defects.dataset import DatasetBase, Sample, image_transform
from defects.patches import tile_grid
from defects.runtime import forward_logits, select_device


class TiledImages(DatasetBase):
    """One item per image: all of its tiles stacked as an (n_tiles, 3, size, size) tensor."""

    def __init__(
        self, paths: Sequence[Path], cfg: dict[str, Any], *, tile_size: int, overlap: float
    ) -> None:
        self.paths = list(paths)
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = image_transform(cfg, training=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        import torch

        with Image.open(self.paths[index]) as opened:
            image = opened.convert("RGB")
        tiles = tile_grid(image.size, self.tile_size, self.overlap)
        return torch.stack([self.transform(image.crop(tile.box)) for tile in tiles])


def predict_tiled(
    model: Any,
    samples: Sequence[Sample],
    cfg: dict[str, Any],
    *,
    tile_size: int,
    overlap: float,
    device: str | None = None,
) -> list[list[float]]:
    """Per-class max tile probability for each sample, in the order given."""
    import torch
    from torch.utils.data import DataLoader

    target = device or select_device()
    model.to(target).eval()
    loader: DataLoader[Any] = DataLoader(
        TiledImages([s.image_path for s in samples], cfg, tile_size=tile_size, overlap=overlap),
        batch_size=None,  # one image (all its tiles) per item; workers decode images in parallel
        num_workers=cfg["num_workers"],
    )
    result: list[list[float]] = []
    with torch.no_grad():
        for tiles in loader:
            probabilities = [
                forward_logits(model, chunk, target, cfg).sigmoid()
                for chunk in tiles.split(cfg["batch_size"])
            ]
            result.append(torch.cat(probabilities).max(dim=0).values.cpu().tolist())
    return result
