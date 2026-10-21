"""Request/response models for the HTTP API (the rest of the contract is in fusion/schema.py)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fusion.schema import HazardRecord


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexResponse(_Strict):
    indexed: list[str]


class QueryRequest(_Strict):
    text: str | None = None
    image_b64: str | None = Field(default=None, description="Base64-encoded image query")
    filters: dict[str, Any] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    llm: Literal["llama", "claude"] = "llama"


class QueryResponse(_Strict):
    answer: str
    llm: Literal["llama", "claude"]
    records: list[HazardRecord]


class HealthResponse(_Strict):
    status: Literal["ok", "degraded"]
    stubs: bool
    models: dict[str, str]
    llm_backend: str
    qdrant: str
    versions: dict[str, str]
