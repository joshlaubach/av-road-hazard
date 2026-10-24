"""How much of a tile lies on the road, and dropping tiles that are mostly off it.

Purpose: the tile classifier fires on grass, snow banks and scenery, which are not road defects.
Skipping a tile that is mostly off the road keeps those hits out of the defect list without
touching any pixels (see docs/adr/0002-road-masked-tiling.md).

Pure geometry: the mask is a bool array of shape (height, width), True on road, from any source
(a segmenter or ground truth), and tiles come from defects/patches.py. Callers with no mask should
skip this filter: a missing mask means the road is unknown, never that the tile is off it.
"""

from __future__ import annotations

from collections.abc import Sequence

from defects.patches import Tile
from fusion.fuse import RoadMask

# A tile needs at least this share of its area on the road to be scored. PROVISIONAL: a round
# starting point, to be set from measurement on annotated tiles, not a tuned value.
DEFAULT_MIN_ROAD = 0.5


def _check_mask(mask: RoadMask) -> None:
    if mask.ndim != 2:
        raise ValueError(f"road mask must be 2-D (height, width), got shape {mask.shape}")
    if mask.dtype != bool:
        raise ValueError(f"road mask must have dtype bool, got {mask.dtype}")


def road_coverage(tile: Tile, mask: RoadMask) -> float:
    """Share of the tile's area on road pixels, in [0, 1].

    Raises ValueError for a mask that is not a 2-D bool array, or a tile that is not inside it.
    """
    _check_mask(mask)
    x1, y1, x2, y2 = tile.box
    height, width = mask.shape
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"tile {tile.box} is not inside the mask of height {height} and width {width}"
        )
    return float(mask[y1:y2, x1:x2].mean())


def drop_offroad_tiles(
    tiles: Sequence[Tile], mask: RoadMask, *, min_road: float = DEFAULT_MIN_ROAD
) -> list[Tile]:
    """The tiles whose road coverage is at least min_road, in their original order."""
    if not 0.0 <= min_road <= 1.0:  # also False for NaN
        raise ValueError(f"min_road must be in [0, 1], got {min_road}")
    _check_mask(mask)
    return [tile for tile in tiles if road_coverage(tile, mask) >= min_road]
