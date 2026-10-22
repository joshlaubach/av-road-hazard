"""Tile geometry: overlapping square tiles, cropping, and tile <-> frame box mapping.

Purpose: replace the source's fixed 8x8 patch grid (ADR 0002). Pure geometry; it decides nothing
about labels or the road mask, so the same functions serve RDD2022 training tiles and inference
tiles on a driving frame.

Conventions (CONTRIBUTING / contract): image_size is (width, height); a box is (x1, y1, x2, y2)
with x along the width; a Tile.box is half-open in frame pixels, so x2 - x1 is its width.
A tile is cropped at native resolution and resized to the classifier input (380), so a box
inside the tile maps back to the frame by a single scale of tile_side / input_size.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

IntBox = tuple[int, int, int, int]
FloatBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Tile:
    """One crop of the frame: its grid position and its pixel rectangle in the frame."""

    row: int
    col: int
    box: IntBox

    @property
    def side(self) -> int:
        return self.box[2] - self.box[0]


def _starts(low: int, high: int, side: int, stride: int) -> list[int]:
    """Tile origins along one axis; the last tile is pushed back so it ends exactly on the edge."""
    last = high - side
    return [*range(low, last, stride), last]


def tile_grid(
    image_size: tuple[int, int],
    tile_size: int,
    overlap: float,
    *,
    region: IntBox | None = None,
) -> list[Tile]:
    """Square tiles covering `region` (default: the whole image) in reading order.

    The side is tile_size, shrunk to the region's shorter side when the region is smaller, so
    every tile is square, equal in size, and inside the region. overlap is the fraction of a tile
    shared with its neighbour. Rows and columns count from 0 inside the region.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive (width, height), got {image_size}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    x1, y1, x2, y2 = region if region is not None else (0, 0, width, height)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"region {region} is empty or outside the {width}x{height} image")
    side = min(tile_size, x2 - x1, y2 - y1)
    stride = max(1, round(side * (1.0 - overlap)))
    return [
        Tile(row, col, (x, y, x + side, y + side))
        for row, y in enumerate(_starts(y1, y2, side, stride))
        for col, x in enumerate(_starts(x1, x2, side, stride))
    ]


def crop_tile(image: Image.Image, tile: Tile, input_size: int) -> Image.Image:
    """Crop the tile at native resolution and resize it to the classifier input."""
    return image.crop(tile.box).resize((input_size, input_size), Image.Resampling.BICUBIC)


def tile_to_frame_box(box: FloatBox, tile: Tile, input_size: int) -> FloatBox:
    """Map a box in the resized tile (0..input_size) back to frame pixels."""
    scale = tile.side / input_size
    x1, y1, x2, y2 = box
    ox, oy = tile.box[0], tile.box[1]
    return (ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale)


def frame_to_tile_box(box: FloatBox, tile: Tile, input_size: int) -> FloatBox:
    """Map a frame box into the resized tile; the result is not clipped to the tile."""
    scale = input_size / tile.side
    x1, y1, x2, y2 = box
    ox, oy = tile.box[0], tile.box[1]
    return ((x1 - ox) * scale, (y1 - oy) * scale, (x2 - ox) * scale, (y2 - oy) * scale)
