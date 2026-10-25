"""Fusion: turn objects, raw defects and an optional road mask into a HazardRecord.

Pure functions, no model imports, deterministic. The formula and its limits are documented in
fusion/README.md. In short, per defect:

    defect_score = road_multiplier * severity_weight * defect_confidence * (0.5 + 0.5 * proximity)

and the frame's hazard_score is the maximum over defects (0.0 with none). It is an urgency ranking,
not a collision probability.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from common.geometry import box_iou as box_iou
from fusion.config import FusionConfig, load_fusion_config
from fusion.schema import (
    Box,
    Defect,
    DefectRecord,
    HazardRecord,
    Meta,
    NearObject,
    Object,
    Severity,
    Source,
)

SeverityFn = Callable[[Defect], Severity]
RoadMask = npt.NDArray[np.bool_]  # shape (height, width), True where the road is


def unknown_severity(_: Defect) -> Severity:
    """Default severity: unknown for every defect until the class mapping is verified."""
    return "unknown"


def object_distance_px(obj_box: Box, defect_box: Box) -> float:
    """Pixel distance from the object's bottom-center to the nearest point of the defect box.

    The bottom-center is a rough ground-contact point (where a person's feet or a car's wheels
    are). The distance is 0 when that point lies inside the defect box.
    """
    cx, cy = (obj_box[0] + obj_box[2]) / 2, obj_box[3]
    dx = max(defect_box[0] - cx, 0.0, cx - defect_box[2])
    dy = max(defect_box[1] - cy, 0.0, cy - defect_box[3])
    return math.hypot(dx, dy)


def normalized_distance(obj_box: Box, defect_box: Box, cfg: FusionConfig) -> float:
    """Distance in object heights: pixel distance over the object's height (with a floor).

    A perspective heuristic: a distant object is small in the image, so dividing by its own height
    keeps "one person-height away" meaning roughly the same near and far. It is not physical
    distance.
    """
    height = max(obj_box[3] - obj_box[1], cfg.min_object_height_px)
    return object_distance_px(obj_box, defect_box) / height


def in_road_state(box: Box, road_mask: RoadMask | None, cfg: FusionConfig) -> bool | None:
    """True / False when a road mask decides it, None (unknown) when it cannot.

    Unknown covers no mask and a box with no in-image area. Missing road information must not
    become False, because that would silently erase a possibly urgent hazard.

    The box is clipped to the image in continuous coordinates, and coverage is the share of that
    clipped area lying on road pixels. A pixel the box only partly overlaps counts in proportion
    to the overlap, so fractional edges are not rounded outward (which would distort the share).
    """
    if road_mask is None:
        return None
    height, width = road_mask.shape
    x1, y1 = max(box[0], 0.0), max(box[1], 0.0)
    x2, y2 = min(box[2], float(width)), min(box[3], float(height))
    if x2 <= x1 or y2 <= y1:  # no area inside the image (also covers zero-width/height boxes)
        return None
    ix1, iy1, ix2, iy2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    cols, rows = np.arange(ix1, ix2), np.arange(iy1, iy2)
    # Overlap length of the box with each pixel column / row, between 0 and 1.
    wx = np.clip(np.minimum(x2, cols + 1) - np.maximum(x1, cols), 0.0, 1.0)
    wy = np.clip(np.minimum(y2, rows + 1) - np.maximum(y1, rows), 0.0, 1.0)
    weights = np.outer(wy, wx)
    covered = float((weights * road_mask[iy1:iy2, ix1:ix2]).sum()) / float(weights.sum())
    # 1e-9 absorbs float error so a share that is exactly the threshold in theory is accepted.
    return bool(covered + 1e-9 >= cfg.road_overlap_threshold)


def road_multiplier(in_road: bool | None) -> float:
    """False -> 0 (confirmed off-road), True -> 1, None (unknown) -> 1, so unknown stays urgent."""
    return 0.0 if in_road is False else 1.0


def near_objects_for(
    defect_box: Box, objects: Sequence[Object], cfg: FusionConfig
) -> list[NearObject]:
    """Objects within near_radius object heights of the defect, nearest first.

    A local report for people reading a record. It is NOT a full explanation of the score: the
    score uses every object, so an object outside the radius can still contribute a little.
    """
    found: list[tuple[float, str, NearObject]] = []
    for obj in objects:
        d = normalized_distance(obj.box, defect_box, cfg)
        if d <= cfg.near_radius:
            near = NearObject(
                label=obj.label,
                iou=box_iou(obj.box, defect_box),
                dist_px=object_distance_px(obj.box, defect_box),
            )
            found.append((d, obj.label, near))
    found.sort(key=lambda item: (item[0], item[1]))
    return [near for _, _, near in found]


def proximity(defect_box: Box, objects: Sequence[Object], cfg: FusionConfig) -> float:
    """Strongest object relevance in [0, 1]; 0.0 with no objects.

    Each object contributes vulnerability * confidence * exp(-distance / scale). Taking the
    maximum, not the sum, stops duplicate detections or a crowd from inflating the score.
    """
    best = 0.0
    for obj in objects:
        d = normalized_distance(obj.box, defect_box, cfg)
        relevance = cfg.vulnerability(obj.label) * obj.score * math.exp(-d / cfg.distance_scale)
        best = max(best, relevance)
    return best


def defect_score(
    in_road: bool | None,
    severity: Severity,
    confidence: float,
    prox: float,
    cfg: FusionConfig,
) -> float:
    """Urgency of one defect in [0, 1]. The 0.5 baseline means an empty road can still be urgent."""
    return (
        road_multiplier(in_road) * cfg.severity_weights[severity] * confidence * (0.5 + 0.5 * prox)
    )


def fuse(
    *,
    frame_id: str,
    source: Source,
    image_size: tuple[int, int],
    objects: Sequence[Object],
    defects: Sequence[Defect],
    road_mask: RoadMask | None = None,
    meta: Meta | None = None,
    config: FusionConfig | None = None,
    severity_fn: SeverityFn = unknown_severity,
) -> HazardRecord:
    """Build the HazardRecord for one frame.

    image_size is (width, height); road_mask, if given, has shape (height, width) and must match.
    Raises ValueError on a size mismatch (a classic width/height mix-up) or a non-finite box.
    Severity comes from severity_fn, which stays "unknown" until the RDD2022 class mapping is
    verified.
    """
    cfg = config or load_fusion_config()
    width, height = image_size
    if road_mask is not None and road_mask.shape != (height, width):
        raise ValueError(
            f"road_mask shape {road_mask.shape} does not match image_size {image_size} "
            "(expected (height, width))"
        )
    # The schema checks box ordering but not finiteness; NaN/inf would crash the geometry or
    # leak into the record, so reject them here.
    boxes = [o.box for o in objects] + [d.box for d in defects]
    for box in boxes:
        if not all(math.isfinite(c) for c in box):
            raise ValueError(f"box has a non-finite coordinate: {box}")
    records: list[DefectRecord] = []
    scores: list[float] = []
    for defect in defects:
        in_road = in_road_state(defect.box, road_mask, cfg)
        severity = severity_fn(defect)
        scores.append(
            defect_score(
                in_road,
                severity,
                defect.confidence,
                proximity(defect.box, objects, cfg),
                cfg,
            )
        )
        records.append(
            DefectRecord(
                name=defect.name,
                confidence=defect.confidence,
                box=defect.box,
                severity=severity,
                in_road=in_road,
                near_objects=near_objects_for(defect.box, objects, cfg),
            )
        )
    return HazardRecord(
        frame_id=frame_id,
        source=source,
        image_size=image_size,
        objects=list(objects),
        defects=records,
        hazard_score=min(1.0, max(scores, default=0.0)),
        meta=meta or Meta(),
    )
