"""FastAPI app. All routes return stub data until later workstreams replace their parts."""

from __future__ import annotations

import logging
import re
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
# A client-supplied x-request-id is only accepted if it looks like this. The id ends up in log lines
# and in debug-dump folder names, so letters, digits and . _ - (max 64 chars) keep path tricks out.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _version(pkg: str) -> str:
    """Installed version of a package, for /health. Returns "unknown" instead of raising when the
    package is not installed (for example when running from a source checkout without install)."""
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


def create_app() -> FastAPI:
    """Build the FastAPI app. All routes live inside this function so they can share the config,
    the pipeline and the in-memory record store through closures.

    A function (not module-level routes) means tests can build a fresh app, and importing this
    module stays cheap. The module-level `app` at the bottom is what uvicorn serves.
    """
    cfg = get_config()
    configure_logging(cfg["logging"]["level"])
    app = FastAPI(title="av-road-hazard", version="0.1.0")
    pipeline = Pipeline()
    # frame_id -> record. Lost on restart; replaced by Qdrant in workstream D.
    store: dict[str, HazardRecord] = {}  # in-memory until workstream D (Qdrant)
    api_cfg = cfg["api"]

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Give every request an id, expose it to the logs, and log one line per request.

        The id is stored in a context variable so any code running during the request (logging,
        debug dumps) can read it without it being passed through every function.
        """
        incoming = request.headers.get("x-request-id", "")
        # Never trust the header as-is: it can end up in file paths (debug dumps) and logs.
        rid = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        # set() returns a token that lets us restore the previous value afterwards.
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
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
        finally:
            # Always restore, even if the route raised, so an id never leaks into the next request.
            request_id_var.reset(token)

    async def _read_image(file: UploadFile) -> bytes:
        """Validate an upload and return its bytes: allowed content type, and not over the size cap.

        The content type is only what the client claims; the real check that the bytes are an image
        happens later in decode_image. Reading max + 1 bytes is enough to detect "too large" without
        pulling a huge body into memory.
        """
        if file.content_type not in api_cfg["allowed_content_types"]:
            raise HTTPException(415, f"unsupported content type: {file.content_type}")
        data = await file.read(api_cfg["max_upload_bytes"] + 1)
        if len(data) > api_cfg["max_upload_bytes"]:
            raise HTTPException(413, "file too large")
        return data

    @app.exception_handler(InvalidImage)
    async def invalid_image(_: Request, exc: InvalidImage) -> JSONResponse:
        """Turn a bad image anywhere in the pipeline into a 422 instead of a 500.

        Registered once here so routes do not each need their own try/except.
        """
        return JSONResponse({"detail": str(exc)}, status_code=422)

    # POST /analyze: one image in, one HazardRecord out. The record is also kept in the store so it
    # can be fetched later with GET /records/{id}. (Comments, not docstrings, on the routes below:
    # FastAPI copies docstrings into openapi.json, which is a frozen contract.)
    @app.post("/analyze", response_model=HazardRecord)
    async def analyze(file: UploadFile = File(...)) -> HazardRecord:  # noqa: B008
        record = pipeline.analyze(await _read_image(file))
        store[record.frame_id] = record
        return record

    # POST /index: analyze a batch of images and keep the records. Capped by max_batch_files so one
    # request cannot hold the server for long. Returns the frame ids that were stored.
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

    # POST /query: stub. Needs text or an image; returns the first top_k stored records and a fixed
    # answer. Real retrieval and LLM generation arrive with workstream D.
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

    # GET /records/{id}: fetch a stored record, or 404.
    @app.get("/records/{record_id}", response_model=HazardRecord)
    def get_record(record_id: str) -> HazardRecord:
        if record_id not in store:
            raise HTTPException(404, "record not found")
        return store[record_id]

    # GET /health: what is running. While stubs are on, status is "ok"; with stubs off the real
    # models do not exist yet, so status is "degraded" and each model reports "not_implemented".
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


# The ASGI app uvicorn serves: `uvicorn api.main:app`.
app = create_app()
