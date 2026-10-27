"""Search quality on queries generated from nuScenes annotations.

A query is a sentence such as "a road scene with traffic cones"; the frames that really contain
traffic cones (by the human annotations) are its relevant set. Recall at k is the share of the best
possible answer the top k contains: hits in the top k divided by min(k, number of relevant frames),
so a perfect search scores 1.0 even when more than k frames are relevant. Pure logic: stdlib and
numpy only.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Words a person would type for each object label.
PHRASES: dict[str, str] = {
    "pedestrian": "pedestrians",
    "cyclist": "cyclists",
    "motorcycle": "motorcycles",
    "car": "cars",
    "truck": "trucks",
    "bus": "buses",
    "barrier": "barriers",
    "traffic_cone": "traffic cones",
}
KS = (1, 5, 10)
MIN_RELEVANT = 3  # fewer relevant frames than this makes a noisy query
MAX_SHARE = 0.5  # a query that half the frames satisfy tests nothing


@dataclass(frozen=True)
class FrameFacts:
    """What the human annotations say about one frame."""

    frame_id: str
    night: bool
    labels: frozenset[str]  # object labels with at least one clearly visible annotated box


@dataclass(frozen=True)
class Query:
    text: str
    kind: str  # "object", "light" or "object+light"
    relevant: frozenset[str]


def make_queries(frames: Sequence[FrameFacts]) -> list[Query]:
    """Object, light and object-at-night queries with enough (but not too many) relevant frames."""
    total = len(frames)
    candidates: list[Query] = []
    for night, light in ((True, "at night"), (False, "in daylight")):
        candidates.append(
            Query(
                f"a road scene {light}",
                "light",
                frozenset(f.frame_id for f in frames if f.night == night),
            )
        )
    for label, plural in PHRASES.items():
        has = [f for f in frames if label in f.labels]
        candidates.append(
            Query(f"a road scene with {plural}", "object", frozenset(f.frame_id for f in has))
        )
        for night, light in ((True, "at night"), (False, "in daylight")):
            candidates.append(
                Query(
                    f"a road scene with {plural} {light}",
                    "object+light",
                    frozenset(f.frame_id for f in has if f.night == night),
                )
            )
    return [q for q in candidates if MIN_RELEVANT <= len(q.relevant) <= MAX_SHARE * total]


def cache_matches(
    stored: Mapping[str, Any], version: str, frame_ids: Sequence[str], config: Mapping[str, Any]
) -> bool:
    """Whether a cached set of records was made for exactly these frames, dataset and models."""
    return (
        stored.get("version") == version
        and stored.get("frame_ids") == list(frame_ids)
        and stored.get("config") == dict(config)
        and len(stored.get("records", ())) == len(frame_ids)
    )


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Hits in the top k over the most hits any search could get (min(k, |relevant|))."""
    if not relevant:
        raise ValueError("a query needs at least one relevant frame")
    hits = sum(1 for frame_id in ranked[:k] if frame_id in relevant)
    return hits / min(k, len(relevant))


Ranker = Callable[[Query], list[str]]


def score_ranker(
    ranker: Ranker, queries: Sequence[Query], ks: Sequence[int] = KS
) -> dict[int, list[float]]:
    """Per-query recall at each k."""
    out: dict[int, list[float]] = {k: [] for k in ks}
    for query in queries:
        ranked = ranker(query)
        for k in ks:
            out[k].append(recall_at_k(ranked, query.relevant, k))
    return out


def random_ranker(frame_ids: Sequence[str], seed: int = 0) -> Ranker:
    """A search that ignores the query: the floor every real search must beat."""
    rng = random.Random(seed)

    def rank(_: Query) -> list[str]:
        order = list(frame_ids)
        rng.shuffle(order)
        return order

    return rank


def mean_with_interval(
    values: Sequence[float], *, resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean over queries and a 90% range from resampling the queries with replacement."""
    if not values:
        raise ValueError("no values")
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(resamples, len(arr)))].mean(axis=1)
    low, high = np.percentile(means, [5, 95])
    return float(arr.mean()), float(low), float(high)


def summarize(
    scores: Mapping[int, Sequence[float]], queries: Sequence[Query]
) -> dict[str, dict[str, dict[str, float]]]:
    """{"all"|kind: {"recall@k": {mean, low, high}}} for the report."""
    groups = {"all": list(range(len(queries)))}
    for kind in sorted({q.kind for q in queries}):
        groups[kind] = [i for i, q in enumerate(queries) if q.kind == kind]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, rows in groups.items():
        out[name] = {}
        for k, values in scores.items():
            mean, low, high = mean_with_interval([values[i] for i in rows])
            out[name][f"recall@{k}"] = {
                "mean": round(mean, 4),
                "low": round(low, 4),
                "high": round(high, 4),
                "queries": len(rows),
            }
    return out
