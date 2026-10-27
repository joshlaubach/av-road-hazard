"""Request/response models for the HTTP API (the rest of the contract is in fusion/schema.py)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from fusion.schema import Box, HazardRecord, Prob, Severity, Source, StrictModel, TimeOfDay


# Response of POST /index: the frame ids that were analyzed and stored.
# (Comments, not docstrings, on these models: docstrings land in openapi.json, a frozen contract.)
class IndexResponse(StrictModel):
    indexed: list[str]


# Plain conditions for POST /query; every set field must hold. Typed so a bad value is a 422 at the
# edge and never reaches the index.
class QueryFilters(StrictModel):
    source: Source | None = None
    time_of_day: TimeOfDay | None = None
    min_hazard_score: float | None = Field(default=None, ge=0.0)
    min_severity: Literal["low", "medium", "high"] | None = None
    has_object: str | None = Field(default=None, pattern=r"^[a-z][a-z _-]{0,31}$")
    has_defect: Literal["D00", "D10", "D20", "D40"] | None = None


# Body of POST /query. At least one of text / image_b64 is required; that rule is checked in the
# route, not here. The sizes are bounded so one request cannot ask for unbounded work: text to 500
# characters, the image to the upload cap in base64 (10 MiB of image), the result list to 50.
class QueryRequest(StrictModel):
    text: str | None = Field(default=None, max_length=500)
    image_b64: str | None = Field(
        default=None, max_length=14_000_000, description="Base64-encoded image query"
    )
    filters: QueryFilters | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    llm: Literal["llama", "claude"] = "llama"


# Response of POST /query: the answer text, which LLM backend produced it, the records it used, the
# model that wrote the answer (null when none did), the frame ids the answer cites (all among
# `records`), and a notice when the written answer was skipped (limit reached or model unavailable).
class QueryResponse(StrictModel):
    answer: str
    llm: Literal["llama", "claude"]
    records: list[HazardRecord]
    model: str | None = None
    cited: list[str] = Field(default_factory=list)
    notice: str | None = None


# Response of GET /health: overall status plus per-component state, so a deploy check can see which
# models or services are missing.
class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    stubs: bool
    models: dict[str, str]
    llm_backend: str
    qdrant: str
    versions: dict[str, str]


# What the pipeline did on the way to a record, for POST /analyze/trace. The record is exactly what
# POST /analyze returns; the trace is extra and never changes it.
#
# One tile of the classifier's grid: where it is, how much of it is under objects and on the road
# (road_cover is null when there was no road mask), whether it was scored or skipped and why, and
# for a scored tile its probability for each damage class and the classes at or above their cutoff.
class TileTrace(StrictModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    box: Box
    object_cover: Prob
    road_cover: Prob | None
    status: Literal["scored", "skipped_object", "skipped_off_road"]
    probabilities: dict[str, Prob] | None
    flagged: list[str]


# The road mask shrunk for display: a grayscale PNG (white is road), base64 encoded, same aspect
# ratio as the photo, at most 160 pixels on its longer side.
class RoadMaskPreview(StrictModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    png_base64: str


# How one merged defect became a score: score = road_multiplier x severity_weight x confidence x
# (0.5 + 0.5 x proximity). merged_from is the number of flagged tiles that were merged into it.
class FusionStep(StrictModel):
    name: str
    merged_from: int = Field(ge=1)
    in_road: bool | None
    road_multiplier: Prob
    severity: Severity
    severity_weight: Prob
    confidence: Prob
    proximity: Prob
    score: Prob


class Trace(StrictModel):
    image_size: tuple[int, int]
    tile_size: int = Field(ge=1)
    tile_overlap: Prob
    thresholds: dict[str, Prob]
    max_object_cover: Prob
    min_road_cover: Prob
    tiles: list[TileTrace]
    road_mask: RoadMaskPreview | None
    fusion: list[FusionStep]


# Response of POST /analyze/trace.
class TracedAnalysis(StrictModel):
    record: HazardRecord
    trace: Trace
