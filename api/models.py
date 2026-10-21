"""Request/response models for the HTTP API (the rest of the contract is in fusion/schema.py)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from fusion.schema import HazardRecord, StrictModel


# Response of POST /index: the frame ids that were analyzed and stored.
# (Comments, not docstrings, on these models: docstrings land in openapi.json, a frozen contract.)
class IndexResponse(StrictModel):
    indexed: list[str]


# Body of POST /query. At least one of text / image_b64 is required; that rule is checked in the
# route, not here. top_k is bounded so a query cannot ask for an unbounded result list.
class QueryRequest(StrictModel):
    text: str | None = None
    image_b64: str | None = Field(default=None, description="Base64-encoded image query")
    filters: dict[str, Any] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    llm: Literal["llama", "claude"] = "llama"


# Response of POST /query: the answer text, which LLM backend produced it, and the records it used.
class QueryResponse(StrictModel):
    answer: str
    llm: Literal["llama", "claude"]
    records: list[HazardRecord]


# Response of GET /health: overall status plus per-component state, so a deploy check can see which
# models or services are missing.
class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    stubs: bool
    models: dict[str, str]
    llm_backend: str
    qdrant: str
    versions: dict[str, str]
