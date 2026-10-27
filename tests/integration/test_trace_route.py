"""POST /analyze/trace: the same record as /analyze, plus what the pipeline did on the way."""

from typing import Any

from fastapi.testclient import TestClient

from api.main import app
from api.models import TracedAnalysis

client = TestClient(app)


def _post(path: str, frame_bytes: bytes) -> dict[str, Any]:
    r = client.post(path, files={"file": ("f.png", frame_bytes, "image/png")})
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def test_the_trace_route_returns_the_same_record_as_analyze(frame_bytes: bytes) -> None:
    plain = _post("/analyze", frame_bytes)
    traced = TracedAnalysis.model_validate(_post("/analyze/trace", frame_bytes))
    assert traced.record.model_dump(mode="json") == plain


def test_the_trace_describes_the_tiles_and_the_fusion(frame_bytes: bytes) -> None:
    traced = TracedAnalysis.model_validate(_post("/analyze/trace", frame_bytes))
    trace = traced.trace
    assert trace.image_size == traced.record.image_size == (640, 360)
    assert trace.tiles and set(trace.thresholds) == {"D00", "D10", "D20", "D40"}
    assert len(trace.fusion) == len(traced.record.defects)
    top = max((s.score for s in trace.fusion), default=0.0)
    assert abs(top - traced.record.hazard_score) < 1e-9


def test_the_trace_route_keeps_the_record_for_later_lookup(frame_bytes: bytes) -> None:
    traced = _post("/analyze/trace", frame_bytes)
    got = client.get(f"/records/{traced['record']['frame_id']}")
    assert got.status_code == 200 and got.json() == traced["record"]


def test_the_trace_route_rejects_what_analyze_rejects() -> None:
    bad_type = client.post("/analyze/trace", files={"file": ("f.txt", b"hi", "text/plain")})
    assert bad_type.status_code == 415
    not_image = client.post("/analyze/trace", files={"file": ("f.png", b"nope", "image/png")})
    assert not_image.status_code == 422
