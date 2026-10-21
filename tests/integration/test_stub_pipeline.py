from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_analyze_index_query_records_flow(frame_bytes: bytes) -> None:
    """The happy path across routes: analyze, fetch the record, index the same image, query it."""
    rec = client.post("/analyze", files={"file": ("f.png", frame_bytes, "image/png")}).json()
    assert rec["objects"] and rec["defects"]

    got = client.get(f"/records/{rec['frame_id']}")
    assert got.status_code == 200 and got.json() == rec

    idx = client.post("/index", files=[("files", ("f.png", frame_bytes, "image/png"))])
    assert idx.json()["indexed"] == [rec["frame_id"]]

    q = client.post("/query", json={"text": "pothole near pedestrian", "top_k": 3})
    assert q.status_code == 200
    assert q.json()["records"][0]["frame_id"] == rec["frame_id"]


def test_rejects_non_image_and_bad_content_type(frame_bytes: bytes) -> None:
    """Wrong content type gives 415, unreadable bytes 422, unknown record 404, empty query 422."""
    assert (
        client.post("/analyze", files={"file": ("f.txt", b"hi", "text/plain")}).status_code == 415
    )
    assert (
        client.post("/analyze", files={"file": ("f.png", b"nope", "image/png")}).status_code == 422
    )
    assert client.get("/records/missing").status_code == 404
    assert client.post("/query", json={"top_k": 2}).status_code == 422


def test_rejects_decoded_image_over_pixel_cap() -> None:
    """A small file that decodes to a huge image is refused before it is decoded."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("1", (6000, 6000)).save(buf, "PNG")  # tiny file, 36M pixels once decoded
    assert len(buf.getvalue()) < 100_000
    r = client.post("/analyze", files={"file": ("big.png", buf.getvalue(), "image/png")})
    assert r.status_code == 422
    assert "too large" in r.json()["detail"]


def test_request_id_header_is_sanitized_and_logged() -> None:
    """A safe x-request-id is kept and logged; an unsafe one (path characters) is replaced."""
    import json
    import logging

    from common.logging import JsonFormatter

    lines: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(JsonFormatter().format(record))

    handler = Capture()
    logging.getLogger("api").addHandler(handler)
    try:
        ok = client.get("/health", headers={"x-request-id": "abc-123"})
        evil = client.get("/health", headers={"x-request-id": "../../etc/x"})
    finally:
        logging.getLogger("api").removeHandler(handler)
    assert ok.headers["x-request-id"] == "abc-123"
    assert evil.headers["x-request-id"] not in ("../../etc/x", "")
    assert "/" not in evil.headers["x-request-id"]
    logged = [json.loads(line) for line in lines]
    assert next(r["request_id"] for r in logged if r["route"] == "/health") == "abc-123"
