"""The search service the API uses: stores a visitor's photos, searches, and writes the answer.

It ties together the embedder, the Qdrant index, the language models and the limits (rag/guards.py).
It knows nothing about HTTP; the API turns its exceptions into status codes. Every photo belongs to
a session (rag/index.py), and every call here names the session it acts for.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fusion.schema import HazardRecord
from rag.embed import DIM, ClipEmbedder, Embedder, HashEmbedder
from rag.guards import BudgetExceeded, RateLimiter, Slots, SpendGuard
from rag.index import CORPUS, Filters, Hit, Index, check_session
from rag.llm import (
    LLM,
    Answer,
    LLMUnavailable,
    StubLLM,
    answer_with_escalation,
    clean_question,
    make_tiers,
)
from rag.text import describe

log = logging.getLogger("rag")

SWEEP_EVERY_S = 60  # how often expired photos are physically deleted (they are hidden at once)
IMAGE_ONLY_ANSWER = "These are the photos most similar to yours."
BUDGET_NOTICE = "Written answers are paused for now; here are the matching photos."
UNAVAILABLE_NOTICE = "The answer service is unavailable right now; here are the matching photos."


class QuotaExceeded(RuntimeError):
    """The visitor has stored as many photos as one session may hold."""


@dataclass(frozen=True)
class QueryResult:
    answer: str
    llm: str
    model: str | None
    records: list[HazardRecord]
    cited: list[str]
    notice: str | None = None


class RagService:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        embedder: Embedder | None = None,
        tiers_for: Callable[[str], list[LLM]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        rag = cfg["rag"]
        self._rag = rag
        self._clock = clock
        self._stubs = bool(cfg["pipeline"]["use_stubs"])
        self._embedder = embedder
        self._embedder_lock = threading.Lock()
        self._qdrant = cfg["qdrant"]
        self._index: Index | None = None
        self._index_lock = threading.Lock()
        self.spend = SpendGuard(float(rag["daily_usd"]), float(rag["monthly_usd"]))
        self.query_limiter = RateLimiter(int(rag["queries_per_minute"]))
        self.index_limiter = RateLimiter(int(rag["uploads_per_minute"]))
        self.slots = Slots(int(rag["parallel_jobs"]))
        llm_cfg = cfg["llm"]
        self._default_stub = self._stubs and tiers_for is None
        self._tiers_for = tiers_for or (
            (lambda kind: [StubLLM(kind)])
            if self._default_stub
            else (
                lambda kind: make_tiers(
                    kind, llm_cfg, on_usage=self.spend.record, before_call=self.spend.check
                )
            )
        )
        # (session, frame id) -> (record, the time it stops being served)
        self._recent: OrderedDict[tuple[str, str], tuple[HazardRecord, float]] = OrderedDict()
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._last_sweep = float("-inf")
        self._quota_lock = threading.Lock()
        self._reserved: dict[str, int] = {}  # places promised to uploads that are still running

    @property
    def index(self) -> Index:
        """Connected on first use, so routes that never store or search (/analyze, /health) do not
        need the Qdrant server to be up."""
        with self._index_lock:
            if self._index is None:
                self._index = Index(DIM, self._qdrant["url"], self._qdrant.get("api_key"))
            return self._index

    @property
    def embedder(self) -> Embedder:
        """Built on first use: loading CLIP takes seconds and memory that /health must not pay."""
        with self._embedder_lock:
            if self._embedder is None:
                device = self._rag["device"]
                self._embedder = HashEmbedder() if self._stubs else ClipEmbedder(device)
            return self._embedder

    # ------------------------------------------------------------------ sessions

    def touch(self, session: str) -> None:
        """Note that the session is in use, and forget sessions that have gone quiet or that
        exceed the number of sessions kept (the least recently used go first)."""
        check_session(session, allow_corpus=False)
        now = self._clock()
        self._seen[session] = now
        self._seen.move_to_end(session)
        ttl = float(self._rag["session_ttl_s"])
        limit = int(self._rag["max_sessions"])
        while self._seen:
            oldest, seen_at = next(iter(self._seen.items()))
            if now - seen_at <= ttl and len(self._seen) <= limit:
                break
            self._forget(oldest)
        if self._index is not None and now - self._last_sweep >= SWEEP_EVERY_S:
            self._last_sweep = now
            try:
                self._index.delete_expired()
            except Exception:  # housekeeping only: searches already ignore expired photos
                log.exception("could not delete expired photos")

    def _forget(self, session: str) -> None:
        self._seen.pop(session, None)
        for key in [k for k in self._recent if k[0] == session]:
            del self._recent[key]
        if self._index is not None:  # nothing was ever stored if the index was never opened
            self._index.delete_session(session)

    def remember(self, session: str, record: HazardRecord) -> None:
        """Keep a record from /analyze so GET /records can return it (bounded per session)."""
        ttl = float(self._rag["session_ttl_s"])
        self._recent[(session, record.frame_id)] = (record, self._clock() + ttl)
        self._recent.move_to_end((session, record.frame_id))
        mine = [k for k in self._recent if k[0] == session]
        for key in mine[: max(0, len(mine) - int(self._rag["max_records_per_session"]))]:
            del self._recent[key]

    # ------------------------------------------------------------------ photos

    def add_photos(
        self, session: str, records: Sequence[HazardRecord], frames: Sequence[np.ndarray]
    ) -> None:
        new = len({r.frame_id for r in records} - self._ids_in(session, records))
        with self._quota_lock:  # check and reserve as one step, so two uploads cannot both fit
            held = self.index.count([session]) + self._reserved.get(session, 0)
            if held + new > int(self._rag["max_records_per_session"]):
                raise QuotaExceeded("this session already holds as many photos as it may")
            self._reserved[session] = self._reserved.get(session, 0) + new
        try:
            images = self.embedder.embed_images(frames)
            texts = self.embedder.embed_texts([describe(r) for r in records])
            ttl = float(self._rag["session_ttl_s"])
            self.index.add_upload(records, images, texts, session, ttl)
            for record in records:
                self.remember(session, record)
        finally:
            with self._quota_lock:
                left = self._reserved.get(session, 0) - new
                if left > 0:
                    self._reserved[session] = left
                else:
                    self._reserved.pop(session, None)

    def _ids_in(self, session: str, records: Sequence[HazardRecord]) -> set[str]:
        return {r.frame_id for r in records if self.index.get(r.frame_id, [session]) is not None}

    def get_record(self, frame_id: str, session: str) -> HazardRecord | None:
        recent = self._recent.get((session, frame_id))
        if recent is not None and recent[1] > self._clock():
            return recent[0]
        return self.index.get(frame_id, [CORPUS, session])

    # ------------------------------------------------------------------ search and answer

    def search(
        self,
        session: str,
        text: str | None,
        frame: np.ndarray | None,
        filters: Filters | None,
        top_k: int,
    ) -> list[Hit]:
        queries: dict[str, np.ndarray] = {}
        if text:
            vector = self.embedder.embed_texts([text])[0]
            queries["text"] = vector
            queries["image"] = vector
        if frame is not None:
            queries["image"] = self.embedder.embed_images([frame])[0]
        if not queries:
            raise ValueError("a search needs text or an image")
        return self.index.search(queries, [CORPUS, session], filters, top_k)

    def query(
        self,
        session: str,
        text: str | None,
        frame: np.ndarray | None,
        filters: Filters | None,
        top_k: int,
        llm: str,
    ) -> QueryResult:
        question = clean_question(text) if text else None
        if text is not None and not question:
            raise ValueError("the question is empty")
        with self.slots:  # embedding the question and writing the answer are both model work
            hits = self.search(session, question, frame, filters, top_k)
            records = [h.record for h in hits]
            if not question:
                return QueryResult(IMAGE_ONLY_ANSWER, llm, None, records, [])
            try:
                answer = self._answer(llm, question, hits)
            except BudgetExceeded as exc:
                log.warning("answer skipped: %s", exc)
                return QueryResult(BUDGET_NOTICE, llm, None, records, [], BUDGET_NOTICE)
            except LLMUnavailable as exc:
                log.error("language model unavailable: %s", exc)
                return QueryResult(UNAVAILABLE_NOTICE, llm, None, records, [], UNAVAILABLE_NOTICE)
            return QueryResult(answer.text, llm, answer.model or None, records, list(answer.cited))

    def _answer(self, llm: str, question: str, hits: Sequence[Hit]) -> Answer:
        tiers = self._tiers_for(llm)
        if llm == "claude" and hits and not self._default_stub:
            self.spend.check()
        return answer_with_escalation(tiers, question, hits)
