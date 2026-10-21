"""FastAPI app. All routes return stub data until later workstreams replace their parts."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from importlib import metadata

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from api.models import HealthResponse, IndexResponse, QueryRequest, QueryResponse
from api.pipeline import InvalidImage, Pipeline
from common.config import get_config
from common.logging import configure_logging, request_id_var
from fusion.schema import HazardRecord

log = logging.getLogger("api")


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


def create_app() -> FastAPI:
    cfg = get_config()
    configure_logging(cfg["logging"]["level"])
    app = FastAPI(title="av-road-hazard", version="0.1.0")
    pipeline = Pipeline()
    store: dict[str, HazardRecord] = {}  # in-memory until workstream D (Qdrant)
    api_cfg = cfg["api"]

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        log.info(
            "request",
            extra={
                "route": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    async def _read_image(file: UploadFile) -> bytes:
        if file.content_type not in api_cfg["allowed_content_types"]:
            raise HTTPException(415, f"unsupported content type: {file.content_type}")
        data = await file.read(api_cfg["max_upload_bytes"] + 1)
        if len(data) > api_cfg["max_upload_bytes"]:
            raise HTTPException(413, "file too large")
        return data

    @app.exception_handler(InvalidImage)
    async def invalid_image(_: Request, exc: InvalidImage) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.post("/analyze", response_model=HazardRecord)
    async def analyze(file: UploadFile = File(...)) -> HazardRecord:  # noqa: B008
        record = pipeline.analyze(await _read_image(file))
        store[record.frame_id] = record
        return record

    @app.post("/index", response_model=IndexResponse)
    async def index(files: list[UploadFile] = File(...)) -> IndexResponse:  # noqa: B008
        if len(files) > api_cfg["max_batch_files"]:
            raise HTTPException(413, "too many files")
        ids: list[str] = []
        for f in files:
            record = pipeline.analyze(await _read_image(f))
            store[record.frame_id] = record
            ids.append(record.frame_id)
        return IndexResponse(indexed=ids)

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        if req.text is None and req.image_b64 is None:
            raise HTTPException(422, "provide text or image_b64")
        records = list(store.values())[: req.top_k]
        return QueryResponse(
            answer="Stub answer: retrieval and LLM generation arrive with workstream D.",
            llm=req.llm,
            records=records,
        )

    @app.get("/records/{record_id}", response_model=HazardRecord)
    def get_record(record_id: str) -> HazardRecord:
        if record_id not in store:
            raise HTTPException(404, "record not found")
        return store[record_id]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        stubs = bool(cfg["pipeline"]["use_stubs"])
        state = "stub" if stubs else "not_implemented"
        return HealthResponse(
            status="ok" if stubs else "degraded",
            stubs=stubs,
            models={"perception": state, "defects": state, "fusion": "stub", "rag": "stub"},
            llm_backend=str(cfg["llm"]["backend"]),
            qdrant="not_configured",
            versions={"av-road-hazard": _version("av-road-hazard"), "fastapi": _version("fastapi")},
        )

    return app


app = create_app()
