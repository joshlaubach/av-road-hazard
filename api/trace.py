"""Build the trace of an analysis: the road-mask preview, the tile story and the fusion steps.

Only reads what the pipeline already computed. Nothing here changes a record or a score.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence

import numpy as np
from PIL import Image

from api.models import FusionStep, RoadMaskPreview, TileTrace, Trace
from defects.infer import TracedDefects
from defects.labels import CODES
from fusion.config import FusionConfig
from fusion.fuse import RoadMask, proximity, road_multiplier
from fusion.schema import HazardRecord, Object

PREVIEW_MAX_SIDE = 160


def road_mask_preview(
    mask: RoadMask | None, max_side: int = PREVIEW_MAX_SIDE
) -> RoadMaskPreview | None:
    """The road mask as a small grayscale PNG (white is road), or None without a mask."""
    if mask is None:
        return None
    height, width = mask.shape
    scale = min(1.0, max_side / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if size != (width, height):
        image = image.resize(size, Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.point(lambda v: 255 if v >= 128 else 0).save(buffer, format="PNG")
    return RoadMaskPreview(
        width=size[0],
        height=size[1],
        png_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def fusion_steps(
    record: HazardRecord,
    objects: Sequence[Object],
    merged_from: Sequence[int],
    config: FusionConfig,
) -> list[FusionStep]:
    """One step per defect of the record: the factors that multiply into its score."""
    steps: list[FusionStep] = []
    for defect, tiles in zip(record.defects, merged_from, strict=True):
        near = proximity(defect.box, objects, config)
        weight = config.severity_weights[defect.severity]
        multiplier = road_multiplier(defect.in_road)
        steps.append(
            FusionStep(
                name=defect.name,
                merged_from=tiles,
                in_road=defect.in_road,
                road_multiplier=multiplier,
                severity=defect.severity,
                severity_weight=weight,
                confidence=defect.confidence,
                proximity=near,
                score=min(1.0, multiplier * weight * defect.confidence * (0.5 + 0.5 * near)),
            )
        )
    return steps


def build_trace(
    *,
    record: HazardRecord,
    objects: Sequence[Object],
    road_mask: RoadMask | None,
    traced: TracedDefects,
    config: FusionConfig,
) -> Trace:
    """The trace for one analysis: tiles, thresholds, the road mask preview and the fusion steps."""
    cut = dict(zip(CODES, traced.thresholds, strict=True))
    tiles: list[TileTrace] = []
    for scored in traced.tiles:
        probabilities = (
            dict(zip(CODES, scored.probabilities, strict=True)) if scored.probabilities else None
        )
        tiles.append(
            TileTrace(
                row=scored.tile.row,
                col=scored.tile.col,
                box=scored.tile.box,
                object_cover=scored.object_cover,
                road_cover=scored.road_cover,
                status=scored.status,
                probabilities=probabilities,
                flagged=[c for c, p in (probabilities or {}).items() if p >= cut[c]],
            )
        )
    return Trace(
        image_size=record.image_size,
        tile_size=traced.tile_size,
        tile_overlap=traced.tile_overlap,
        thresholds=cut,
        max_object_cover=traced.max_object_cover,
        min_road_cover=traced.min_road_cover,
        tiles=tiles,
        road_mask=road_mask_preview(road_mask),
        fusion=fusion_steps(record, objects, [len(g) for g in traced.groups], config),
    )
