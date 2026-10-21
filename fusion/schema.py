"""Contract types for av-road-hazard. Frozen after Sprint 0; see CONTRACT_CHANGELOG.md.

Conventions (pinned here, the single source of truth):
- Boxes are pixel coordinates (x1, y1, x2, y2) with x1 <= x2 and y1 <= y2, origin top-left.
- image_size is (width, height) in pixels. NOT (height, width).
- Scores and confidences are probabilities in [0, 1].
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

Severity = Literal["high", "medium", "low", "unknown"]
Source = Literal["carla", "rdd2022", "upload", "nuscenes"]
TimeOfDay = Literal["day", "night", "unknown"]
Prob = Annotated[float, Field(ge=0.0, le=1.0)]


# Base class for every contract model. extra="forbid" rejects unknown fields, so a typo or a field
# from a newer version fails loudly instead of being silently dropped.
# (Comments, not docstrings, where a model has none: docstrings land in openapi.json.)
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Reject boxes whose corners are the wrong way round (min must come before max).

    A flipped box usually means an x/y or width/height mix-up upstream, so fail at the boundary.
    """
    x1, y1, x2, y2 = box
    if x1 > x2 or y1 > y2:
        raise ValueError(f"box must satisfy x1<=x2 and y1<=y2, got {box}")
    return box


# Box type used by every model: four floats, validated by _check_box when a model is built.
Box = Annotated[tuple[float, float, float, float], AfterValidator(_check_box)]


class Object(StrictModel):
    """One detected object (output of perception.infer.detect)."""

    label: str
    score: Prob
    box: Box


class Defect(StrictModel):
    """One raw defect detection (output of defects.infer.find_defects).

    patch_row and patch_col index the tile grid that produced the detection; the names are kept
    from the source repo's patch grid. cls is the 1-indexed class id; name is its label.
    """

    cls: int = Field(ge=1)
    name: str
    confidence: Prob
    box: Box
    patch_row: int = Field(ge=0)
    patch_col: int = Field(ge=0)


# An object close to a defect: its label, the box overlap (IoU) with the defect, and the pixel
# distance between them. Filled in by fusion (workstream C).
class NearObject(StrictModel):
    label: str
    iou: Prob
    dist_px: float = Field(ge=0.0)


class DefectRecord(StrictModel):
    """A defect as it appears in a HazardRecord, after fusion."""

    name: str
    confidence: Prob
    box: Box
    severity: Severity
    # True: on the road. False: confirmed off the road. None: unknown (no usable road mask).
    # Fusion scores unknown like True so uncertainty never hides a hazard. Readers must not test
    # truthiness: None is falsy.
    in_road: bool | None
    near_objects: list[NearObject] = []


# Optional per-frame context. Both fields default to "unknown"/None when a source cannot tell.
class Meta(StrictModel):
    time_of_day: TimeOfDay = "unknown"
    location: str | None = None


class HazardRecord(StrictModel):
    """One record per frame: objects, fused defects, hazard score, metadata."""

    frame_id: str = Field(min_length=1)
    source: Source
    image_size: tuple[int, int] = Field(description="(width, height) in pixels")
    objects: list[Object] = []
    defects: list[DefectRecord] = []
    hazard_score: float = Field(ge=0.0, description="Deterministic formula; see fusion/README.md")
    meta: Meta = Meta()

    @model_validator(mode="after")
    def _size_ok(self) -> HazardRecord:
        """Check image_size is positive. It is (width, height), the opposite order of numpy's
        shape, which is a classic source of coordinate bugs."""
        w, h = self.image_size
        if w <= 0 or h <= 0:
            raise ValueError(f"image_size must be positive (width, height), got {self.image_size}")
        return self
