"""Whole-request size limits, applied before the body is read or parsed, and no GPS in records."""

from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main as main_module
from api.pipeline import Pipeline
from common import config

UPLOAD_CAP = 2_000  # bytes; small so the tests stay fast
OVERHEAD = 1_048_576  # what a multipart request may add on top of its files (see api/limits.py)
SMALL_BODY = 65_536  # the limit for any route that takes no upload


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    cfg = config.get_config()
    monkeypatch.setitem(cfg["api"], "max_upload_bytes", UPLOAD_CAP)
    monkeypatch.setitem(cfg["api"], "max_batch_files", 3)
    yield TestClient(main_module.create_app())


@pytest.fixture
def never_analyze(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("the body was read: analyze ran")

    monkeypatch.setattr(Pipeline, "analyze", boom)
    monkeypatch.setattr(Pipeline, "analyze_traced", boom)


@pytest.mark.usefixtures("never_analyze")
@pytest.mark.parametrize("path", ["/analyze", "/analyze/trace"])
def test_an_upload_far_over_the_cap_is_refused_before_analysis(
    client: TestClient, path: str
) -> None:
    big = b"x" * (UPLOAD_CAP + OVERHEAD + 1)
    response = client.post(path, files={"file": ("f.png", big, "image/png")})
    assert response.status_code == 413


@pytest.mark.usefixtures("never_analyze")
def test_a_batch_over_its_whole_body_limit_is_refused(client: TestClient) -> None:
    big = b"x" * (3 * UPLOAD_CAP + OVERHEAD + 1)
    response = client.post("/index", files=[("files", ("f.png", big, "image/png"))])
    assert response.status_code == 413


def test_a_query_with_an_oversized_json_body_is_refused(client: TestClient) -> None:
    body = '{"text": "' + "a" * (2 * UPLOAD_CAP + SMALL_BODY) + '"}'
    response = client.post("/query", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_a_route_that_takes_no_upload_has_a_small_body_limit(client: TestClient) -> None:
    response = client.request("GET", "/health", content=b"x" * (SMALL_BODY + 1))
    assert response.status_code == 413


@pytest.mark.usefixtures("never_analyze")
def test_a_streamed_body_over_the_limit_is_cut_off_without_a_length_header(
    client: TestClient,
) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(80):  # 80 KiB, over the /query limit (about 66 KiB with the small test cap)
            yield b"x" * 1024

    response = client.post("/query", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_a_normal_upload_under_the_limits_still_works(client: TestClient) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (90, 90, 90)).save(buf, "PNG")
    assert len(buf.getvalue()) < UPLOAD_CAP
    response = client.post("/analyze", files={"file": ("f.png", buf.getvalue(), "image/png")})
    assert response.status_code == 200


def test_records_are_rate_limited_like_queries(client: TestClient) -> None:
    per_minute = int(config.get_config()["rag"]["queries_per_minute"])
    codes = [client.get("/records/missing").status_code for _ in range(per_minute + 1)]
    assert codes[:per_minute] == [404] * per_minute
    assert codes[per_minute] == 429


def _jpeg_with_gps() -> bytes:
    image = Image.new("RGB", (64, 64), (120, 120, 120))
    exif = Image.Exif()
    exif[0x8825] = {1: "N", 2: (37.0, 46.0, 29.7), 3: "W", 4: (122.0, 25.0, 9.8)}  # GPS block
    buf = io.BytesIO()
    image.save(buf, "JPEG", exif=exif)
    assert b"Exif" in buf.getvalue()
    return buf.getvalue()


def test_gps_in_a_photo_never_reaches_the_record(client: TestClient) -> None:
    """Photos are decoded to pixels and dropped; only the analysis is kept, with no location."""
    data = _jpeg_with_gps()
    record = client.post("/analyze", files={"file": ("f.jpg", data, "image/jpeg")}).json()
    assert record["meta"]["location"] is None
    indexed = client.post("/index", files=[("files", ("f.jpg", data, "image/jpeg"))]).json()
    stored = client.get(f"/records/{indexed['indexed'][0]}")
    assert stored.status_code == 200
    text = stored.text
    assert "37.7" not in text and "122.4" not in text and "GPS" not in text


def test_an_oversized_content_length_is_refused_without_reading_a_byte() -> None:
    """The check uses the header alone: the receive channel must never be touched."""
    import asyncio

    app = main_module.create_app()
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        raise AssertionError("the app tried to read the body")

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/analyze",
        "raw_path": b"/analyze",
        "query_string": b"",
        "headers": [(b"content-length", str(10**9).encode()), (b"host", b"test")],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    asyncio.run(app(scope, receive, send))  # type: ignore[arg-type]
    assert sent[0]["status"] == 413
