"""Search and answers through the HTTP API: sessions, limits and input checks (stub pipeline)."""

from __future__ import annotations

import base64
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import app
from rag.llm import LLMUnavailable
from rag.service import RagService

PHOTO = "image/png"


class Scripted:
    name = "llama"
    model = "scripted"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    def generate(self, system: str, prompt: str) -> str:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def svc() -> Iterator[RagService]:
    """The app's search service with a scripted model, so no test talks to Ollama or Claude."""
    service: RagService = app.state.rag()
    original = service._tiers_for
    service._tiers_for = lambda kind: [Scripted("It is in the photo.")]
    yield service
    service._tiers_for = original


def upload(client: TestClient, frame_bytes: bytes) -> str:
    response = client.post("/index", files=[("files", ("f.png", frame_bytes, PHOTO))])
    assert response.status_code == 200, response.text
    return str(response.json()["indexed"][0])


def test_the_server_issues_a_private_session_cookie(frame_bytes: bytes, svc: RagService) -> None:
    client = TestClient(app)
    upload(client, frame_bytes)
    header = client.post("/query", json={"text": "road"}).headers.get("set-cookie", "")
    assert header == ""  # the cookie was already set by the first call and is reused
    first = TestClient(app).post("/index", files=[("files", ("f.png", frame_bytes, PHOTO))])
    cookie = first.headers["set-cookie"].lower()
    assert "rh_session=s_" in cookie and "httponly" in cookie and "samesite=lax" in cookie


def test_one_visitor_cannot_see_or_fetch_another_visitors_photo(
    frame_bytes: bytes, svc: RagService
) -> None:
    alice, bob = TestClient(app), TestClient(app)
    frame_id = upload(alice, frame_bytes)
    assert alice.get(f"/records/{frame_id}").status_code == 200
    assert bob.get(f"/records/{frame_id}").status_code == 404
    assert bob.post("/query", json={"text": "road", "top_k": 50}).json()["records"] == []
    mine = alice.post("/query", json={"text": "road", "top_k": 50}).json()
    assert [r["frame_id"] for r in mine["records"]] == [frame_id]


def test_a_forged_session_cookie_gets_a_new_session_and_cannot_reach_the_corpus(
    frame_bytes: bytes, svc: RagService
) -> None:
    alice = TestClient(app)
    frame_id = upload(alice, frame_bytes)
    for forged in ("corpus", "s_" + "A" * 30, "../x", ""):
        mallory = TestClient(app, cookies={"rh_session": forged})
        assert mallory.get(f"/records/{frame_id}").status_code == 404
        assert mallory.post("/query", json={"text": "road"}).json()["records"] == []
    assert svc.index.count(["corpus"]) == 0  # nothing could be written into the shared photos


def test_a_photo_query_works_and_gets_no_written_answer(
    frame_bytes: bytes, svc: RagService
) -> None:
    client = TestClient(app)
    frame_id = upload(client, frame_bytes)
    body = {"image_b64": base64.b64encode(frame_bytes).decode(), "top_k": 3}
    found = client.post("/query", json=body).json()
    assert [r["frame_id"] for r in found["records"]] == [frame_id]
    assert found["model"] is None and found["cited"] == []


def test_a_text_query_returns_the_models_answer_and_checked_citations(
    frame_bytes: bytes, svc: RagService
) -> None:
    client = TestClient(app)
    frame_id = upload(client, frame_bytes)
    svc._tiers_for = lambda kind: [Scripted(f"A car is in [{frame_id}] and [ghost].")]
    found = client.post("/query", json={"text": "cars", "llm": "claude"}).json()
    assert found["cited"] == [frame_id] and found["model"] == "scripted"
    assert "ghost" not in found["answer"] and "[unverified source]" in found["answer"]


def test_a_model_failure_is_a_notice_and_never_leaks_the_reason(
    frame_bytes: bytes, svc: RagService
) -> None:
    client = TestClient(app)
    upload(client, frame_bytes)
    svc._tiers_for = lambda kind: [Scripted(LLMUnavailable("key sk-secret rejected"))]
    found = client.post("/query", json={"text": "cars"})
    assert found.status_code == 200 and found.json()["notice"]
    assert "secret" not in found.text and found.json()["records"]


def test_too_many_requests_get_a_429_with_retry_after(
    frame_bytes: bytes, svc: RagService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(svc.query_limiter, "limit", 2)
    client = TestClient(app)
    codes = [client.post("/query", json={"text": "road"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    limited = client.post("/query", json={"text": "road"})
    assert int(limited.headers["retry-after"]) >= 1


def test_the_address_limit_stops_a_visitor_who_keeps_dropping_the_cookie(
    svc: RagService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(svc.query_limiter, "limit", 2)
    codes = [TestClient(app).post("/query", json={"text": "road"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_a_refused_request_does_not_evict_another_visitors_photos(
    frame_bytes: bytes, svc: RagService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(svc._rag, "max_sessions", 1)
    monkeypatch.setattr(svc.query_limiter, "limit", 1)
    victim = TestClient(app)
    upload(victim, frame_bytes)
    assert victim.post("/query", json={"text": "road"}).status_code == 200  # uses the address's one
    session = victim.cookies["rh_session"]
    attacker = TestClient(app)  # no cookie: a fresh session id on every request
    assert attacker.post("/query", json={"text": "road"}).status_code == 429
    assert svc.index.count([session]) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"text": "x" * 501},
        {"text": "road", "top_k": 51},
        {"text": "road", "filters": {"min_severity": "catastrophic"}},
        {"text": "road", "filters": {"has_defect": "D99"}},
        {"text": "road", "filters": {"session": "corpus"}},
        {"text": "road", "filters": {"has_object": "a" * 40}},
        {"text": "road", "llm": "gpt"},
        {"text": "road", "session": "corpus"},
        {"image_b64": "A" * 14_000_001},
    ],
)
def test_bad_query_bodies_are_refused_before_any_work(
    body: dict[str, object], svc: RagService
) -> None:
    assert TestClient(app).post("/query", json=body).status_code == 422


def test_bad_images_in_a_query_are_refused(svc: RagService) -> None:
    client = TestClient(app)
    assert client.post("/query", json={"image_b64": "not base64!!"}).status_code == 422
    junk = base64.b64encode(b"definitely not an image").decode()
    assert client.post("/query", json={"image_b64": junk}).status_code == 422


def test_filters_narrow_the_search(frame_bytes: bytes, svc: RagService) -> None:
    client = TestClient(app)
    upload(client, frame_bytes)
    none = client.post("/query", json={"text": "road", "filters": {"time_of_day": "night"}})
    assert none.status_code == 200 and none.json()["records"] == []


def test_the_photo_limit_per_session_is_a_429(
    frame_bytes: bytes, svc: RagService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(svc._rag, "max_records_per_session", 0)
    client = TestClient(app)
    response = client.post("/index", files=[("files", ("f.png", frame_bytes, PHOTO))])
    assert response.status_code == 429


def test_health_says_where_the_index_lives() -> None:
    assert TestClient(app).get("/health").json()["qdrant"] in {"in_memory", "configured"}
