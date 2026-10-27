"""FastAPI app: analysis routes, and photo search and answers per visitor session."""

from __future__ import annotations

import base64
import binascii
import logging
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from importlib import metadata

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from api.models import (
    HealthResponse,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    TracedAnalysis,
)
from api.pipeline import InvalidImage, Pipeline, decode_image, model_status
from api.session import client_address, session_for
from common.config import get_config
from common.logging import configure_logging, request_id_var
from defects.infer import DefectsUnavailable
from fusion.schema import HazardRecord
from perception.detector import DetectorUnavailable
from rag.guards import Busy, RateLimited
from rag.index import Filters
from rag.service import QuotaExceeded, RagService

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
    api_cfg = cfg["api"]
    rag_holder: list[RagService] = []
    rag_lock = threading.Lock()

    def rag() -> RagService:
        """The search service, built on first use (cheap: CLIP itself loads on its first query)."""
        with rag_lock:
            if not rag_holder:
                rag_holder.append(RagService(cfg))
            return rag_holder[0]

    app.state.rag = rag  # lets tests reach the service (to reset the limits they share)

    def caller(request: Request, response: Response, limiter: str) -> tuple[RagService, str]:
        """Who is asking: their session (issued here), after counting this request against the
        limits for the session and for the address. Over the limit is a 429 with Retry-After."""
        svc = rag()
        session = session_for(request, response)
        svc.touch(session)
        gate = svc.query_limiter if limiter == "query" else svc.index_limiter
        try:
            gate.check(f"s:{session}")
            gate.check(f"a:{client_address(request)}")
        except RateLimited as exc:
            raise HTTPException(
                429, "too many requests", headers={"Retry-After": str(exc.retry_after)}
            ) from exc
        return svc, session

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

    @app.exception_handler(DetectorUnavailable)
    async def detector_unavailable(_: Request, exc: DetectorUnavailable) -> JSONResponse:
        """Fail loudly when the detector cannot run: a record built without object evidence would
        look safer than it is. The reason (paths, hashes) goes to the log, not to the client."""
        log.error("object detector unavailable: %s", exc)
        return JSONResponse({"detail": "object detector unavailable"}, status_code=503)

    @app.exception_handler(DefectsUnavailable)
    async def defects_unavailable(_: Request, exc: DefectsUnavailable) -> JSONResponse:
        """Same rule as the detector: no defects found is not the same as no classifier."""
        log.error("defect classifier unavailable: %s", exc)
        return JSONResponse({"detail": "defect classifier unavailable"}, status_code=503)

    # POST /analyze: one image in, one HazardRecord out. The record is also kept in the store so it
    # can be fetched later with GET /records/{id}. (Comments, not docstrings, on the routes below:
    # FastAPI copies docstrings into openapi.json, which is a frozen contract.)
    @app.post("/analyze", response_model=HazardRecord)
    async def analyze(
        request: Request,
        response: Response,
        file: UploadFile = File(...),  # noqa: B008
    ) -> HazardRecord:
        svc, session = caller(request, response, "upload")
        record = pipeline.analyze(await _read_image(file))
        svc.remember(session, record)
        return record

    # POST /analyze/trace: the same record as /analyze, plus the trace of what the pipeline did
    # (tile grid, skipped tiles, class probabilities, road mask preview, fusion factors). The record
    # is stored like /analyze's. Meant for the UI's walkthrough, not for bulk use.
    @app.post("/analyze/trace", response_model=TracedAnalysis)
    async def analyze_trace(
        request: Request,
        response: Response,
        file: UploadFile = File(...),  # noqa: B008
    ) -> TracedAnalysis:
        svc, session = caller(request, response, "upload")
        result = pipeline.analyze_traced(await _read_image(file))
        svc.remember(session, result.record)
        return result

    # POST /index: analyze a batch of images, keep them in the caller's session and make them
    # searchable. Capped by max_batch_files and by the photos one session may hold. Returns the
    # frame ids that were stored.
    @app.post("/index", response_model=IndexResponse)
    async def index(
        request: Request,
        response: Response,
        files: list[UploadFile] = File(...),  # noqa: B008
    ) -> IndexResponse:
        svc, session = caller(request, response, "upload")
        if len(files) > api_cfg["max_batch_files"]:
            raise HTTPException(413, "too many files")
        records: list[HazardRecord] = []
        frames: list[np.ndarray] = []
        for f in files:
            data = await _read_image(f)
            records.append(pipeline.analyze(data))
            frames.append(decode_image(data))
        try:
            svc.add_photos(session, records, frames)
        except QuotaExceeded as exc:
            raise HTTPException(429, "this session already holds as many photos as it may") from exc
        return IndexResponse(indexed=[r.frame_id for r in records])

    def _query_image(image_b64: str) -> np.ndarray:
        try:
            data = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(422, "image_b64 is not valid base64") from exc
        if len(data) > api_cfg["max_upload_bytes"]:
            raise HTTPException(413, "image too large")
        return decode_image(data)

    # POST /query: search the shared photos and the caller's own, then write an answer from what
    # was found (text queries only; a photo query returns the most similar photos). The caller is
    # the cookie's session; the body cannot name another one.
    @app.post("/query", response_model=QueryResponse)
    def query(request: Request, response: Response, req: QueryRequest) -> QueryResponse:
        if req.text is None and req.image_b64 is None:
            raise HTTPException(422, "provide text or image_b64")
        svc, session = caller(request, response, "query")
        frame = _query_image(req.image_b64) if req.image_b64 is not None else None
        filters = Filters(**req.filters.model_dump()) if req.filters is not None else None
        try:
            found = svc.query(session, req.text, frame, filters, req.top_k, req.llm)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Busy as exc:
            raise HTTPException(503, "the server is busy; try again shortly") from exc
        return QueryResponse(
            answer=found.answer,
            llm=req.llm,
            records=found.records,
            model=found.model,
            cited=found.cited,
            notice=found.notice,
        )

    # GET /records/{id}: fetch a record from the shared photos or the caller's own, or 404. Another
    # visitor's record is a 404, the same as one that does not exist.
    @app.get("/records/{record_id}", response_model=HazardRecord)
    def get_record(record_id: str, request: Request, response: Response) -> HazardRecord:
        found = rag().get_record(record_id, session_for(request, response))
        if found is None:
            raise HTTPException(404, "record not found")
        return found

    # GET /health: what is running. Each model is "stub", "ready" or "available" (its files are
    # there), or "unavailable"; the status is "degraded" if any model is unavailable.
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        models = model_status(cfg)
        return HealthResponse(
            status="degraded" if "unavailable" in models.values() else "ok",
            stubs=bool(cfg["pipeline"]["use_stubs"]),
            models=models,
            llm_backend=str(cfg["llm"]["backend"]),
            qdrant="configured" if cfg["qdrant"]["url"] else "in_memory",
            versions={"av-road-hazard": _version("av-road-hazard"), "fastapi": _version("fastapi")},
        )

    return app


# The ASGI app uvicorn serves: `uvicorn api.main:app`.
app = create_app()
