"""The Qdrant index: one point per photo, two vectors (the photo and its description) and metadata.

Every point belongs to a session ("corpus" for the shared demo photos, a session id for a visitor's
uploads). Every search names the sessions it may see, so one visitor's photos never come back for
another.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import numpy as np
from qdrant_client import QdrantClient, models

from fusion.schema import HazardRecord
from rag.text import SEVERITY_RANK

COLLECTION = "frames"
IMAGE_VECTOR = "image"
TEXT_VECTOR = "text"
CORPUS = "corpus"
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


def point_id(session: str, frame_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session}/{frame_id}"))


def build_filter(sessions: Collection[str], filters: Filters | None) -> models.Filter:
    """Always restricts to `sessions`; an empty list is an error, never "everything"."""
    if not sessions:
        raise ValueError("a search must name at least one session")
    must: list[models.Condition] = [
        models.FieldCondition(key="session", match=models.MatchAny(any=sorted(sessions)))
    ]
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
        rank = SEVERITY_RANK[f.min_severity]
        must.append(models.FieldCondition(key="severity_rank", range=models.Range(gte=rank)))
    return models.Filter(must=must)


def payload_for(record: HazardRecord, session: str) -> dict[str, object]:
    rank = max((SEVERITY_RANK[d.severity] for d in record.defects), default=0)
    return {
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

    def __init__(self, dim: int, url: str | None = None) -> None:
        self.dim = dim
        if url is None:
            self.client = QdrantClient(":memory:")
        elif url.startswith("http"):
            self.client = QdrantClient(url=url)
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
    ) -> None:
        if not (len(records) == len(image_vectors) == len(text_vectors)):
            raise ValueError("records and vectors must be the same length")
        points = [
            models.PointStruct(
                id=point_id(session, r.frame_id),
                vector={IMAGE_VECTOR: iv.tolist(), TEXT_VECTOR: tv.tolist()},
                payload=payload_for(r, session),
            )
            for r, iv, tv in zip(records, image_vectors, text_vectors, strict=True)
        ]
        self.client.upsert(COLLECTION, points)

    def count(self, session: str | None = None) -> int:
        flt = None if session is None else build_filter([session], None)
        return int(self.client.count(COLLECTION, count_filter=flt, exact=True).count)

    def get(self, frame_id: str, sessions: Collection[str]) -> HazardRecord | None:
        """The record, only if it lives in one of the sessions the caller may see."""
        for session in sorted(sessions):
            found = self.client.retrieve(COLLECTION, [point_id(session, frame_id)])
            if found and found[0].payload:
                return HazardRecord.model_validate(found[0].payload["record"])
        return None

    def search(
        self,
        queries: dict[str, np.ndarray],
        sessions: Collection[str],
        filters: Filters | None = None,
        top_k: int = 5,
    ) -> list[Hit]:
        """Filter first, search each named vector with its query vector, then fuse the rankings.

        `queries` maps a vector name to a query vector; each one gives a ranked list, and the lists
        are combined by reciprocal rank: a photo scores the sum of 1/(60 + rank) over the lists.
        """
        flt = build_filter(sessions, filters)
        fused: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}
        payloads: dict[str, dict[str, object]] = {}
        for name, vector in queries.items():
            result = self.client.query_points(
                COLLECTION, query=vector.tolist(), using=name, query_filter=flt, limit=POOL
            )
            for rank, point in enumerate(result.points, start=1):
                key = str(point.id)
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
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
