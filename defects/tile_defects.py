"""Frame in, Defect list out: tile the road region, skip object tiles, apply per-class thresholds.

This is the real body of defects.infer.find_defects: everything except model and checkpoint
loading, so it can be tested without either. A Defect's box is its tile's
rectangle in frame pixels, because the classifier scores whole tiles, not boxes inside them. The
name is the RDD2022 class code (for example "D00"); the human-readable names are in
defects/labels.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from defects.labels import CODES
from defects.object_mask import DEFAULT_MAX_COVER, drop_covered_tiles
from defects.patches import IntBox, Tile, tile_grid
from defects.road_region import DEFAULT_MIN_ROAD, drop_offroad_tiles
from defects.runtime import forward_logits, select_device
from fusion.fuse import RoadMask
from fusion.schema import Defect, Object


def tiles_to_defects(
    tiles: Sequence[Tile],
    probabilities: Sequence[Sequence[float]],
    thresholds: Sequence[float],
) -> list[Defect]:
    """One Defect per tile and class whose probability reaches that class's threshold.

    probabilities has one row per tile and one column per class in CODES order; thresholds has one
    value per class. Defects come out in tile order, then class order.
    """
    if len(thresholds) != len(CODES):
        raise ValueError(f"need {len(CODES)} thresholds, got {len(thresholds)}")
    if not all(0.0 <= value <= 1.0 for value in thresholds):  # also False for NaN
        raise ValueError(f"every threshold must be a probability in [0, 1], got {thresholds}")
    if len(tiles) != len(probabilities):
        raise ValueError("need one probability row per tile")
    defects: list[Defect] = []
    for tile, row in zip(tiles, probabilities, strict=True):
        if len(row) != len(CODES):
            raise ValueError(f"need {len(CODES)} class probabilities per tile, got {len(row)}")
        for k, (code, probability, threshold) in enumerate(
            zip(CODES, row, thresholds, strict=True)
        ):
            if probability >= threshold and not math.isnan(probability):
                x1, y1, x2, y2 = tile.box
                defects.append(
                    Defect(
                        cls=k + 1,
                        name=code,
                        confidence=float(probability),
                        box=(float(x1), float(y1), float(x2), float(y2)),
                        patch_row=tile.row,
                        patch_col=tile.col,
                    )
                )
    return defects


def find_frame_defects(
    model: Any,
    frame: np.ndarray,
    objects: Sequence[Object],
    cfg: dict[str, Any],
    *,
    thresholds: Sequence[float],
    region: IntBox | None = None,
    max_object_cover: float = DEFAULT_MAX_COVER,
    road_mask: RoadMask | None = None,
    min_road_cover: float = DEFAULT_MIN_ROAD,
    device: str | None = None,
) -> list[Defect]:
    """Defects in an RGB uint8 frame (H, W, 3), scoring only tiles that are mostly free of objects.

    road_mask is an optional bool array of shape (H, W), True on road: tiles with less than
    min_road_cover of their area on road are not scored, in addition to the object filter. With
    no mask the road is unknown, and no tile is treated as off-road.

    region is (x1, y1, x2, y2) in frame pixels (default: the whole frame); tile rows and columns
    count from its top-left corner. A region shorter or narrower than tile_size gets smaller
    tiles (see tile_grid), which the model then sees at a higher zoom than it was trained at.
    cfg needs the tile_size, tile_overlap, input_size and batch_size the checkpoint was trained
    with. The frame must be RGB channel order, not OpenCV's BGR. The model is moved to the device
    and put in eval mode, as in predict_tiled.
    """
    import torch

    from defects.dataset import image_transform

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            f"frame must be an RGB uint8 image of shape (H, W, 3), got {frame.shape} {frame.dtype}"
        )
    height, width = frame.shape[:2]
    if road_mask is not None and road_mask.shape != (height, width):
        raise ValueError(
            f"road_mask shape {road_mask.shape} must match the frame's (height, width) "
            f"{(height, width)}"
        )
    tiles = tile_grid(
        (width, height), int(cfg["tile_size"]), float(cfg["tile_overlap"]), region=region
    )
    kept = drop_covered_tiles(tiles, [o.box for o in objects], max_cover=max_object_cover)
    if road_mask is not None:
        kept = drop_offroad_tiles(kept, road_mask, min_road=min_road_cover)
    if not kept:
        return []
    image = Image.fromarray(frame)
    transform = image_transform(cfg, training=False)
    batch: Any = torch.stack([transform(image.crop(tile.box)) for tile in kept])
    target = device or select_device()
    model.to(target).eval()
    with torch.no_grad():
        probabilities = torch.cat(
            [
                forward_logits(model, chunk, target, cfg).sigmoid()
                for chunk in batch.split(cfg["batch_size"])
            ]
        )
    return tiles_to_defects(kept, probabilities.cpu().tolist(), thresholds)
