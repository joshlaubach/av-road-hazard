from fastapi.testclient import TestClient

from api.main import app
from fusion.schema import HazardRecord
from tools.export_openapi import TARGET, render

client = TestClient(app)


def test_openapi_json_is_current() -> None:
    assert TARGET.read_text() == render(), "run `python -m tools.export_openapi`"


def test_expected_routes_present() -> None:
    paths = app.openapi()["paths"]
    for route in ["/analyze", "/index", "/query", "/records/{record_id}", "/health"]:
        assert route in paths


def test_analyze_response_validates_against_schema(frame_bytes: bytes) -> None:
    r = client.post("/analyze", files={"file": ("f.png", frame_bytes, "image/png")})
    assert r.status_code == 200
    rec = HazardRecord.model_validate(r.json())
    assert rec.image_size == (640, 360)  # (width, height)
    assert r.headers["x-request-id"]


def test_health_reports_stubs() -> None:
    body = client.get("/health").json()
    assert body["stubs"] is True
    assert body["models"]["perception"] == "stub"
