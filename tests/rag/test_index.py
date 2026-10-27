from __future__ import annotations

import numpy as np
import pytest

from fusion.schema import DefectRecord, HazardRecord, Meta, Object
from rag.index import CORPUS, Filters, Index, build_filter
from rag.text import describe

DIM = 8


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def record(
    frame_id: str,
    *,
    night: bool = False,
    labels: tuple[str, ...] = (),
    defect: str | None = None,
    severity: str = "low",
    score: float = 0.0,
) -> HazardRecord:
    defects = (
        [
            DefectRecord(
                name=defect,
                confidence=0.9,
                box=(0, 0, 1, 1),
                severity=severity,
                in_road=True,
            )
        ]
        if defect
        else []
    )
    return HazardRecord(
        frame_id=frame_id,
        source="nuscenes",
        image_size=(100, 100),
        objects=[Object(label=label, score=0.9, box=(0, 0, 1, 1)) for label in labels],
        defects=defects,
        hazard_score=score,
        meta=Meta(time_of_day="night" if night else "day"),
    )


def filled() -> Index:
    index = Index(DIM)
    records = [
        record("a", labels=("car",), score=0.1),
        record("b", night=True, labels=("pedestrian",), defect="D40", severity="high", score=0.8),
        record("c", labels=("car", "car"), defect="D00", severity="low", score=0.3),
    ]
    vectors = np.stack([unit(0), unit(1), unit(2)])
    index.add(records, vectors, vectors)
    return index


def ids(hits: list) -> list[str]:  # type: ignore[type-arg]
    return [h.record.frame_id for h in hits]


def test_nearest_vector_comes_first() -> None:
    hits = filled().search({"image": unit(1)}, [CORPUS], top_k=3)
    assert ids(hits)[0] == "b"


def test_filters_run_before_the_vector_search() -> None:
    index = filled()
    assert ids(index.search({"image": unit(1)}, [CORPUS], Filters(time_of_day="day"))) != []
    assert "b" not in ids(index.search({"image": unit(1)}, [CORPUS], Filters(time_of_day="day")))
    assert ids(index.search({"image": unit(0)}, [CORPUS], Filters(min_severity="high"))) == ["b"]
    assert ids(index.search({"image": unit(0)}, [CORPUS], Filters(has_defect="D00"))) == ["c"]
    assert ids(index.search({"image": unit(0)}, [CORPUS], Filters(has_object="pedestrian"))) == [
        "b"
    ]
    assert ids(index.search({"image": unit(0)}, [CORPUS], Filters(min_hazard_score=0.5))) == ["b"]


def test_two_rankings_are_fused() -> None:
    hits = filled().search({"image": unit(0), "text": unit(1)}, [CORPUS], top_k=3)
    assert set(ids(hits)) == {"a", "b", "c"}
    assert hits[0].ranks.keys() == {"image", "text"}  # first in both lists
    assert hits[0].record.frame_id in {"a", "b"}


def test_a_session_cannot_see_another_sessions_photos() -> None:
    index = filled()
    index.add([record("mine")], unit(3)[None, :], unit(3)[None, :], session="s1")
    index.add([record("theirs")], unit(3)[None, :], unit(3)[None, :], session="s2")
    seen = ids(index.search({"image": unit(3)}, [CORPUS, "s1"], top_k=10))
    assert "mine" in seen and "theirs" not in seen
    assert index.get("theirs", [CORPUS, "s1"]) is None
    assert index.get("mine", [CORPUS, "s1"]) is not None
    assert index.get("mine", [CORPUS]) is None


def test_the_same_frame_id_in_two_sessions_is_two_points() -> None:
    index = Index(DIM)
    index.add([record("x")], unit(0)[None, :], unit(0)[None, :], session="s1")
    index.add([record("x")], unit(0)[None, :], unit(0)[None, :], session="s2")
    assert index.count() == 2 and index.count("s1") == 1


def test_a_search_must_name_a_session() -> None:
    with pytest.raises(ValueError):
        build_filter([], None)


def test_vectors_and_records_must_match_in_length() -> None:
    with pytest.raises(ValueError):
        Index(DIM).add([record("a")], np.zeros((2, DIM)), np.zeros((2, DIM)))


def test_describe_uses_search_words() -> None:
    text = describe(record("b", night=True, labels=("car", "car"), defect="D40", score=0.5))
    assert "nighttime" in text and "2 car" in text and "pothole" in text and "50 out of 100" in text
    assert "No road damage found" in describe(record("a"))
