"""How much of a tile lies under detected objects, and dropping tiles that are mostly object.

Purpose: the classifier fires on cars and people that sit on the road. Skipping a tile that is
mostly object keeps those out of the defect list without touching any pixels; painting the objects
out instead made new false alarms at the painted edges (see the nuScenes mask experiment).

Pure geometry: boxes are frame pixels (x1, y1, x2, y2) from any source (a detector or ground
truth), and tiles come from defects/patches.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from defects.patches import Tile

FloatBox = tuple[float, float, float, float]

# Each object box grows by this fraction of its own size per side, so the mask covers the object's
# edges and shadow as well as the box a detector drew.
DEFAULT_PAD = 0.05
# A tile with more than this share under objects is not scored.
DEFAULT_MAX_COVER = 0.30


def _check_boxes(boxes: Sequence[FloatBox]) -> None:
    """Reject boxes that would crash the rounding or silently shrink under padding.

    The Object schema checks corner order but not finiteness, so a NaN can reach this far.
    """
    for box in boxes:
        if not all(math.isfinite(c) for c in box):
            raise ValueError(f"box has a non-finite coordinate: {box}")
        if box[0] > box[2] or box[1] > box[3]:
            raise ValueError(f"box must satisfy x1<=x2 and y1<=y2, got {box}")


def object_coverage(tile: Tile, boxes: Sequence[FloatBox], *, pad: float = DEFAULT_PAD) -> float:
    """Share of the tile's area under the union of the boxes, in [0, 1].

    Boxes are clipped to the tile and fractional edges round outward, so coverage is never
    understated. Overlapping boxes count once.
    """
    if pad < 0:
        raise ValueError(f"pad must not be negative, got {pad}")
    _check_boxes(boxes)
    x1, y1, x2, y2 = tile.box
    covered = np.zeros((y2 - y1, x2 - x1), dtype=np.bool_)
    for bx1, by1, bx2, by2 in boxes:
        grow_x, grow_y = pad * (bx2 - bx1), pad * (by2 - by1)
        left = max(math.floor(bx1 - grow_x), x1) - x1
        top = max(math.floor(by1 - grow_y), y1) - y1
        right = min(math.ceil(bx2 + grow_x), x2) - x1
        bottom = min(math.ceil(by2 + grow_y), y2) - y1
        if right > left and bottom > top:
            covered[top:bottom, left:right] = True
    return float(covered.mean())


def drop_covered_tiles(
    tiles: Sequence[Tile],
    boxes: Sequence[FloatBox],
    *,
    max_cover: float = DEFAULT_MAX_COVER,
    pad: float = DEFAULT_PAD,
) -> list[Tile]:
    """The tiles whose object coverage is at most max_cover, in their original order."""
    if not 0.0 <= max_cover <= 1.0:
        raise ValueError(f"max_cover must be in [0, 1], got {max_cover}")
    _check_boxes(boxes)  # also catches a bad box when there are no tiles to compare it with
    return [tile for tile in tiles if object_coverage(tile, boxes, pad=pad) <= max_cover]
