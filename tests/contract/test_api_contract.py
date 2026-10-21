from fastapi.testclient import TestClient

from api.main import app
from fusion.schema import HazardRecord
from tools.export_openapi import TARGET, render

client = TestClient(app)


def test_openapi_json_is_current() -> None:
    """openapi.json matches the running app; regenerate it after changing the API."""
    assert TARGET.read_text() == render(), "run `python -m tools.export_openapi`"


def test_expected_routes_present() -> None:
    """All five contract routes exist."""
    paths = app.openapi()["paths"]
    for route in ["/analyze", "/index", "/query", "/records/{record_id}", "/health"]:
        assert route in paths


def test_analyze_response_validates_against_schema(frame_bytes: bytes) -> None:
    """The /analyze response parses as a HazardRecord and carries a request id."""
    r = client.post("/analyze", files={"file": ("f.png", frame_bytes, "image/png")})
    assert r.status_code == 200
    rec = HazardRecord.model_validate(r.json())
    assert rec.image_size == (640, 360)  # (width, height)
    assert r.headers["x-request-id"]


def test_health_reports_stubs() -> None:
    """/health reports that stubs are on and which models are stubbed."""
    body = client.get("/health").json()
    assert body["stubs"] is True
    assert body["models"]["perception"] == "stub"
