"""Request/response models for the HTTP API (the rest of the contract is in fusion/schema.py)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from fusion.schema import HazardRecord, StrictModel


class IndexResponse(StrictModel):
    indexed: list[str]


class QueryRequest(StrictModel):
    text: str | None = None
    image_b64: str | None = Field(default=None, description="Base64-encoded image query")
    filters: dict[str, Any] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    llm: Literal["llama", "claude"] = "llama"


class QueryResponse(StrictModel):
    answer: str
    llm: Literal["llama", "claude"]
    records: list[HazardRecord]


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    stubs: bool
    models: dict[str, str]
    llm_backend: str
    qdrant: str
    versions: dict[str, str]
