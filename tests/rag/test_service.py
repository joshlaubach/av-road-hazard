from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from common.config import load_config
from fusion.schema import DefectRecord, HazardRecord, Meta, Object
from rag.embed import HashEmbedder
from rag.guards import Busy
from rag.index import CORPUS, Filters, new_session_id
from rag.llm import LLMUnavailable
from rag.service import (
    BUDGET_NOTICE,
    IMAGE_ONLY_ANSWER,
    UNAVAILABLE_NOTICE,
    QuotaExceeded,
    RagService,
)


def config(**rag: Any) -> dict[str, Any]:
    cfg = load_config()
    cfg["pipeline"]["use_stubs"] = True
    cfg["rag"].update(rag)
    return cfg


def record(frame_id: str, *, damage: str | None = None, night: bool = False) -> HazardRecord:
    defects = (
        [DefectRecord(name=damage, confidence=0.9, box=(0, 0, 1, 1), severity="high", in_road=True)]
        if damage
        else []
    )
    return HazardRecord(
        frame_id=frame_id,
        source="upload",
        image_size=(32, 32),
        objects=[Object(label="car", score=0.9, box=(0, 0, 1, 1))],
        defects=defects,
        hazard_score=0.5 if damage else 0.0,
        meta=Meta(time_of_day="night" if night else "day"),
    )


def frame(value: int) -> np.ndarray:
    out = np.zeros((32, 32, 3), dtype=np.uint8)
    out[::2] = value
    return out


class Scripted:
    name = "claude"

    def __init__(self, reply: object, model: str = "m") -> None:
        self.reply, self.model, self.calls = reply, model, 0

    def generate(self, system: str, prompt: str) -> str:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return str(self.reply)


def service(reply: object = "A pothole [p1].", **rag: Any) -> tuple[RagService, Scripted]:
    llm = Scripted(reply)
    return RagService(config(**rag), HashEmbedder(), tiers_for=lambda kind: [llm]), llm


def test_a_visitor_finds_their_own_photos_and_cannot_find_anyone_elses() -> None:
    svc, _ = service()
    mine, theirs = new_session_id(), new_session_id()
    svc.add_photos(mine, [record("p1", damage="D40")], [frame(200)])
    svc.add_photos(theirs, [record("t1", damage="D40")], [frame(50)])
    got = svc.query(mine, "pothole", None, None, 5, "llama")
    assert [r.frame_id for r in got.records] == ["p1"]
    assert svc.get_record("p1", mine) is not None
    assert svc.get_record("t1", mine) is None
    assert svc.get_record("p1", theirs) is None


def test_the_answer_comes_from_the_model_with_checked_citations() -> None:
    svc, llm = service("There is a pothole in [p1].")
    sid = new_session_id()
    svc.add_photos(sid, [record("p1", damage="D40")], [frame(200)])
    got = svc.query(sid, "Where is the pothole?", None, None, 5, "claude")
    assert (
        got.answer == "There is a pothole in [p1]." and got.cited == ["p1"] and got.notice is None
    )
    assert got.model == "m" and llm.calls == 1


def test_a_photo_query_returns_similar_photos_without_calling_a_model() -> None:
    svc, llm = service()
    sid = new_session_id()
    svc.add_photos(sid, [record("a"), record("b")], [frame(200), frame(20)])
    got = svc.query(sid, None, frame(200), None, 1, "claude")
    assert [r.frame_id for r in got.records] == ["a"]
    assert got.answer == IMAGE_ONLY_ANSWER and llm.calls == 0


def test_filters_apply_before_the_search() -> None:
    svc, _ = service()
    sid = new_session_id()
    svc.add_photos(sid, [record("d"), record("n", night=True)], [frame(10), frame(30)])
    got = svc.query(sid, "road", None, Filters(time_of_day="night"), 5, "llama")
    assert [r.frame_id for r in got.records] == ["n"]


def test_the_shared_corpus_is_searchable_by_everyone_but_not_writable_by_a_session() -> None:
    svc, _ = service()
    svc.index.add(
        [record("c1")], np.ones((1, 512), dtype=np.float32), np.ones((1, 512), dtype=np.float32)
    )
    sid = new_session_id()
    assert svc.get_record("c1", sid) is not None
    assert [r.frame_id for r in svc.query(sid, "road", None, None, 5, "llama").records] == ["c1"]
    with pytest.raises(ValueError):
        svc.add_photos(CORPUS, [record("x")], [frame(1)])


def test_a_session_cannot_hold_more_photos_than_the_limit() -> None:
    svc, _ = service(max_records_per_session=2)
    sid = new_session_id()
    svc.add_photos(sid, [record("a"), record("b")], [frame(1), frame(2)])
    svc.add_photos(sid, [record("a")], [frame(1)])  # the same photo again is not a new one
    with pytest.raises(QuotaExceeded):
        svc.add_photos(sid, [record("c")], [frame(3)])


def test_idle_sessions_are_deleted_and_the_oldest_go_first_when_there_are_too_many() -> None:
    now = [0.0]
    svc = RagService(
        config(session_ttl_s=100, max_sessions=2),
        HashEmbedder(),
        tiers_for=lambda k: [],
        clock=lambda: now[0],
    )
    a, b, c = new_session_id(), new_session_id(), new_session_id()
    for sid, name in ((a, "a"), (b, "b")):
        svc.touch(sid)
        svc.add_photos(sid, [record(name)], [frame(5)])
    svc.touch(c)  # a third session pushes out the least recently used (a)
    assert svc.index.count([a]) == 0 and svc.index.count([b]) == 1
    now[0] = 500.0
    svc.touch(c)  # b has been idle for longer than the limit
    assert svc.index.count([b]) == 0 and svc.get_record("b", b) is None


def test_when_the_spend_limit_is_reached_the_photos_still_come_back() -> None:
    svc, llm = service(daily_usd=0.0)
    sid = new_session_id()
    svc.add_photos(sid, [record("p1")], [frame(9)])
    got = svc.query(sid, "road", None, None, 5, "claude")
    assert got.answer == BUDGET_NOTICE and got.notice == BUDGET_NOTICE and llm.calls == 0
    assert [r.frame_id for r in got.records] == ["p1"]


def test_a_model_failure_gives_a_plain_notice_not_an_error() -> None:
    svc, _ = service(LLMUnavailable("secret detail about the key"))
    sid = new_session_id()
    svc.add_photos(sid, [record("p1")], [frame(9)])
    got = svc.query(sid, "road", None, None, 5, "llama")
    assert got.answer == UNAVAILABLE_NOTICE and "secret" not in got.answer
    assert [r.frame_id for r in got.records] == ["p1"]


def test_too_many_jobs_at_once_is_a_busy_error() -> None:
    svc, _ = service(parallel_jobs=1)
    sid = new_session_id()
    svc.add_photos(sid, [record("p1")], [frame(9)])
    with svc.slots, pytest.raises(Busy):
        svc.query(sid, "road", None, None, 5, "llama")


def test_an_empty_question_and_a_missing_query_are_errors() -> None:
    svc, _ = service()
    sid = new_session_id()
    with pytest.raises(ValueError, match="text or an image"):
        svc.search(sid, None, None, None, 5)
    with pytest.raises(ValueError, match="empty"):
        svc.query(sid, "\n\t", None, None, 5, "llama")


def test_routes_that_never_search_do_not_need_the_qdrant_server() -> None:
    cfg = config()
    cfg["qdrant"] = {"url": "http://127.0.0.1:9", "api_key": None}  # nothing listens here
    svc = RagService(cfg, HashEmbedder(), tiers_for=lambda kind: [])
    sid = new_session_id()
    svc.touch(sid)
    svc.remember(sid, record("r1"))
    assert svc.get_record("r1", sid) is not None
