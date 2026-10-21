"""Contract types for av-road-hazard. Frozen after Sprint 0; see CONTRACT_CHANGELOG.md.

Conventions (pinned here, the single source of truth):
- Boxes are pixel coordinates (x1, y1, x2, y2) with x1 <= x2 and y1 <= y2, origin top-left.
- image_size is (width, height) in pixels. NOT (height, width).
- Scores and confidences are probabilities in [0, 1].
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Box = tuple[float, float, float, float]
Severity = Literal["high", "medium", "low", "unknown"]
Source = Literal["carla", "rdd2022", "upload", "nuscenes"]
TimeOfDay = Literal["day", "night", "unknown"]
Prob = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_box(box: Box) -> Box:
    x1, y1, x2, y2 = box
    if x1 > x2 or y1 > y2:
        raise ValueError(f"box must satisfy x1<=x2 and y1<=y2, got {box}")
    return box


class Object(_Strict):
    """One detected object (output of perception.infer.detect)."""

    label: str
    score: Prob
    box: Box

    @model_validator(mode="after")
    def _box_ok(self) -> Object:
        _check_box(self.box)
        return self


class Defect(_Strict):
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

    @model_validator(mode="after")
    def _box_ok(self) -> Defect:
        _check_box(self.box)
        return self


class NearObject(_Strict):
    label: str
    iou: Prob
    dist_px: float = Field(ge=0.0)


class DefectRecord(_Strict):
    """A defect as it appears in a HazardRecord, after fusion."""

    name: str
    confidence: Prob
    box: Box
    severity: Severity
    in_road: bool
    near_objects: list[NearObject] = []

    @model_validator(mode="after")
    def _box_ok(self) -> DefectRecord:
        _check_box(self.box)
        return self


class Meta(_Strict):
    time_of_day: TimeOfDay = "unknown"
    location: str | None = None


class HazardRecord(_Strict):
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
        w, h = self.image_size
        if w <= 0 or h <= 0:
            raise ValueError(f"image_size must be positive (width, height), got {self.image_size}")
        return self
