"""The Qdrant index: one point per photo, two vectors (the photo and its description) and metadata.

Every point belongs to a session ("corpus" for the shared demo photos, a session id for a visitor's
uploads). Every search names the sessions it may see, so one visitor's photos never come back for
another.
"""

from __future__ import annotations

import json
import re
import secrets
import time
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from qdrant_client import QdrantClient, models

from fusion.schema import HazardRecord
from rag.text import SEVERITY_RANK

COLLECTION = "frames"
IMAGE_VECTOR = "image"
TEXT_VECTOR = "text"
CORPUS = "corpus"
# A visitor's session id: issued by the server (new_session_id), never chosen by the client. It is
# long and random so it cannot be guessed, and it can never equal CORPUS.
SESSION_ID = re.compile(r"s_[A-Za-z0-9_-]{22,64}")
RRF_K = 60  # reciprocal rank fusion constant; 60 is the value from the original paper
POOL = 100  # how many hits each vector search contributes before fusing
KEYWORD_FIELDS = ("session", "source", "time_of_day", "objects", "defects")


@dataclass(frozen=True)
class Filters:
    """Plain conditions on the metadata. Every set field must hold."""

    source: str | None = None
    time_of_day: str | None = None
    min_hazard_score: float | None = None
    min_severity: str | None = None
    has_object: str | None = None
    has_defect: str | None = None


@dataclass(frozen=True)
class Hit:
    record: HazardRecord
    score: float
    session: str = CORPUS
    ranks: dict[str, int] = field(default_factory=dict)


def new_session_id() -> str:
    """A fresh unguessable session id (32 random characters after the prefix)."""
    return "s_" + secrets.token_urlsafe(24)


def check_session(session: str, *, allow_corpus: bool = True) -> str:
    """The session if it is well formed, else ValueError. Sessions are used as keys into the
    shared index, so anything that is not the corpus or a server-issued id is refused."""
    if (allow_corpus and session == CORPUS) or SESSION_ID.fullmatch(session):
        return session
    raise ValueError("not a valid session id")


def point_id(session: str, frame_id: str) -> str:
    """Stable id for (session, frame). The pair is encoded as JSON so that no two different pairs
    give the same text ("a/b" + "c" and "a" + "b/c" must not collide)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps([session, frame_id])))


def build_filter(
    sessions: Collection[str], filters: Filters | None, live_at: float | None = None
) -> models.Filter:
    """Always restricts to `sessions`; an empty list is an error, never "everything".

    With `live_at` (a time), a visitor's photo only counts if it has an `expires_at` after that
    time: expired photos, and photos stored without an expiry, are invisible. The shared corpus
    never expires. Deleting and listing the corpus leave `live_at` out and see everything.
    """
    if not sessions:
        raise ValueError("a search must name at least one session")
    for session in sessions:
        check_session(session)
    visitors = sorted(s for s in sessions if s != CORPUS)
    session_match: models.Condition = models.FieldCondition(
        key="session", match=models.MatchAny(any=sorted(sessions))
    )
    if live_at is not None and visitors:
        alive = models.Filter(
            must=[
                models.FieldCondition(key="session", match=models.MatchAny(any=visitors)),
                models.FieldCondition(key="expires_at", range=models.Range(gt=live_at)),
            ]
        )
        options: list[models.Condition] = [alive]
        if CORPUS in sessions:
            options.append(
                models.FieldCondition(key="session", match=models.MatchValue(value=CORPUS))
            )
        session_match = models.Filter(should=options)
    must: list[models.Condition] = [session_match]
    f = filters or Filters()
    for key, value in (
        ("source", f.source),
        ("time_of_day", f.time_of_day),
        ("objects", f.has_object),
        ("defects", f.has_defect),
    ):
        if value is not None:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    if f.min_hazard_score is not None:
        must.append(
            models.FieldCondition(key="hazard_score", range=models.Range(gte=f.min_hazard_score))
        )
    if f.min_severity is not None:
        if f.min_severity not in SEVERITY_RANK:
            raise ValueError(f"min_severity must be one of {sorted(SEVERITY_RANK)}")
        rank = SEVERITY_RANK[f.min_severity]
        must.append(models.FieldCondition(key="severity_rank", range=models.Range(gte=rank)))
    return models.Filter(must=must)


def payload_for(
    record: HazardRecord, session: str, expires_at: float | None = None
) -> dict[str, object]:
    rank = max((SEVERITY_RANK[d.severity] for d in record.defects), default=0)
    expiry: dict[str, object] = {} if expires_at is None else {"expires_at": expires_at}
    return {
        **expiry,
        "frame_id": record.frame_id,
        "session": session,
        "source": record.source,
        "time_of_day": record.meta.time_of_day,
        "hazard_score": record.hazard_score,
        "severity_rank": rank,
        "objects": sorted({o.label for o in record.objects}),
        "defects": sorted({d.name for d in record.defects}),
        "record": record.model_dump(mode="json"),
    }


class Index:
    """`url` None keeps everything in memory; an http URL uses that server; else a folder path."""

    def __init__(
        self,
        dim: int,
        url: str | None = None,
        api_key: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.dim = dim
        self._clock = clock  # wall-clock seconds: stored expiry times must survive a restart
        if url is None:
            self.client = QdrantClient(":memory:")
        elif url.startswith("http"):
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=url)
        if not self.client.collection_exists(COLLECTION):
            params = models.VectorParams(size=dim, distance=models.Distance.COSINE)
            self.client.create_collection(
                COLLECTION, vectors_config={IMAGE_VECTOR: params, TEXT_VECTOR: params}
            )
            for key in KEYWORD_FIELDS if url is not None else ():  # the in-memory mode ignores them
                self.client.create_payload_index(COLLECTION, key, models.PayloadSchemaType.KEYWORD)

    def add(
        self,
        records: Sequence[HazardRecord],
        image_vectors: np.ndarray,
        text_vectors: np.ndarray,
        session: str = CORPUS,
        expires_at: float | None = None,
    ) -> None:
        check_session(session)
        if not (len(records) == len(image_vectors) == len(text_vectors)):
            raise ValueError("records and vectors must be the same length")
        points = [
            models.PointStruct(
                id=point_id(session, r.frame_id),
                vector={IMAGE_VECTOR: iv.tolist(), TEXT_VECTOR: tv.tolist()},
                payload=payload_for(r, session, expires_at),
            )
            for r, iv, tv in zip(records, image_vectors, text_vectors, strict=True)
        ]
        self.client.upsert(COLLECTION, points)

    def add_upload(
        self,
        records: Sequence[HazardRecord],
        image_vectors: np.ndarray,
        text_vectors: np.ndarray,
        session: str,
        ttl_s: float,
    ) -> None:
        """Store a visitor's photos for `ttl_s` seconds. Unlike add(), this refuses the shared
        corpus, so a request that names the session "corpus" cannot write into the demo photos."""
        check_session(session, allow_corpus=False)
        self.add(records, image_vectors, text_vectors, session, self._clock() + ttl_s)

    def delete_session(self, session: str) -> None:
        """Remove everything a visitor stored. The shared corpus cannot be deleted this way."""
        check_session(session, allow_corpus=False)
        self.client.delete(
            COLLECTION, points_selector=models.FilterSelector(filter=build_filter([session], None))
        )

    def clear_corpus(self) -> None:
        """Remove the shared demo photos. Only the corpus import should call this."""
        self.client.delete(
            COLLECTION,
            points_selector=models.FilterSelector(filter=build_filter([CORPUS], None)),
        )

    def corpus_frame_ids(self) -> set[str]:
        """The frame id of every shared demo photo, read back from the store."""
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                COLLECTION,
                scroll_filter=build_filter([CORPUS], None),
                limit=256,
                offset=offset,
                with_payload=["frame_id"],
                with_vectors=False,
            )
            found.update(str(p.payload["frame_id"]) for p in points if p.payload)
            if offset is None:
                return found

    def delete_expired(self) -> None:
        """Physically remove visitor photos that have expired or have no expiry (housekeeping:
        searches already ignore them). The shared corpus is never touched."""
        corpus = models.FieldCondition(key="session", match=models.MatchValue(value=CORPUS))
        expired = models.FieldCondition(key="expires_at", range=models.Range(lte=self._clock()))
        no_expiry = models.IsEmptyCondition(is_empty=models.PayloadField(key="expires_at"))
        self.client.delete(
            COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(must_not=[corpus], should=[expired, no_expiry])
            ),
        )

    def delete_visitor_points(self) -> None:
        """Remove every visitor's photos and keep the shared corpus (used when the app starts)."""
        corpus = models.FieldCondition(key="session", match=models.MatchValue(value=CORPUS))
        self.client.delete(
            COLLECTION,
            points_selector=models.FilterSelector(filter=models.Filter(must_not=[corpus])),
        )

    def count(self, sessions: Collection[str]) -> int:
        """How many photos the given sessions hold; naming at least one session is required."""
        flt = build_filter(sessions, None, self._clock())
        return int(self.client.count(COLLECTION, count_filter=flt, exact=True).count)

    def get(self, frame_id: str, sessions: Collection[str]) -> HazardRecord | None:
        """The record, only if it lives in one of the sessions the caller may see.

        The stored session and frame id are checked as well as the point id, so a mix-up in how
        ids are made could never hand back another session's record.
        """
        build_filter(sessions, None)  # an empty session list is an error here too
        now = self._clock()
        for session in sorted(sessions):
            found = self.client.retrieve(COLLECTION, [point_id(session, frame_id)])
            payload = found[0].payload if found else None
            if not payload or payload["session"] != session or payload["frame_id"] != frame_id:
                continue
            if session != CORPUS and float(payload.get("expires_at", 0)) <= now:
                continue  # expired, or stored without an expiry
            return HazardRecord.model_validate(payload["record"])
        return None

    def search(
        self,
        queries: dict[str, np.ndarray],
        sessions: Collection[str],
        filters: Filters | None = None,
        top_k: int = 5,
        weights: Mapping[str, float] | None = None,
    ) -> list[Hit]:
        """Filter first, search each named vector with its query vector, then fuse the rankings.

        `queries` maps a vector name to a query vector; each one gives a ranked list, and the lists
        are combined by reciprocal rank: a photo scores the sum of weight / (60 + rank) over the
        lists. `weights` maps a vector name to its weight; a name left out counts 1.0, so no
        weights means every list votes equally.
        """
        flt = build_filter(sessions, filters, self._clock())
        fused: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}
        payloads: dict[str, dict[str, object]] = {}
        for name, vector in queries.items():
            result = self.client.query_points(
                COLLECTION,
                query=vector.tolist(),
                using=name,
                query_filter=flt,
                limit=max(POOL, top_k),
            )
            for rank, point in enumerate(result.points, start=1):
                key = str(point.id)
                fused[key] = fused.get(key, 0.0) + (weights or {}).get(name, 1.0) / (RRF_K + rank)
                ranks.setdefault(key, {})[name] = rank
                payloads[key] = point.payload or {}
        best = sorted(fused, key=lambda k: (-fused[k], k))[:top_k]
        return [
            Hit(
                HazardRecord.model_validate(payloads[k]["record"]),
                fused[k],
                str(payloads[k]["session"]),
                ranks[k],
            )
            for k in best
        ]
