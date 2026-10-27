from __future__ import annotations

import pytest

from eval.retrieval import (
    FrameFacts,
    Query,
    cache_matches,
    make_queries,
    mean_with_interval,
    random_expectation,
    recall_at_k,
    score_ranker,
    summarize,
)


def frames() -> list[FrameFacts]:
    out = []
    for i in range(20):
        labels = {"car"} if i < 8 else set()
        if i in (0, 1, 2, 15):
            labels.add("traffic_cone")
        out.append(FrameFacts(f"f{i}", night=i >= 15, labels=frozenset(labels)))
    return out


def test_perfect_search_scores_one_even_with_more_relevant_than_k() -> None:
    relevant = frozenset(f"f{i}" for i in range(8))
    ranked = [f"f{i}" for i in range(20)]
    assert recall_at_k(ranked, relevant, 5) == 1.0
    assert recall_at_k(list(reversed(ranked)), relevant, 5) == 0.0


def test_partial_hits_and_short_relevant_set() -> None:
    assert recall_at_k(["a", "x", "b"], frozenset({"a", "b"}), 2) == 0.5
    assert recall_at_k(["a"], frozenset({"a"}), 10) == 1.0


def test_a_query_needs_a_relevant_frame() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], frozenset(), 1)


def test_queries_come_from_the_annotations_and_skip_rare_and_trivial_ones() -> None:
    queries = {q.text: q for q in make_queries(frames())}
    assert queries["a road scene with cars"].relevant == {f"f{i}" for i in range(8)}
    assert queries["a road scene at night"].relevant == {f"f{i}" for i in range(15, 20)}
    assert queries["a road scene with traffic cones"].relevant == {"f0", "f1", "f2", "f15"}
    assert "a road scene with buses" not in queries  # none
    assert "a road scene in daylight" not in queries  # 15 of 20 frames is more than half
    assert "a road scene with traffic cones at night" not in queries  # only one frame


def test_the_random_floor_is_the_exact_average_over_all_orders() -> None:
    queries = [Query("q", "object", frozenset(f"f{i}" for i in range(10)))]
    # 100 frames, 10 relevant: the top 10 holds 1 relevant frame on average, out of at most 10.
    assert random_expectation(queries, 100, (1, 10)) == {1: [0.1], 10: [0.1]}
    few = [Query("q", "object", frozenset({"a", "b", "c"}))]
    assert random_expectation(few, 30, (5,))[5] == [pytest.approx(0.5 / 3)]


def test_interval_contains_the_mean_and_summary_groups_by_kind() -> None:
    mean, low, high = mean_with_interval([0.0, 1.0, 1.0, 0.5])
    assert low <= mean <= high
    qs = make_queries(frames())
    out = summarize(score_ranker(lambda q: sorted(q.relevant), qs, (1, 5)), qs)
    assert out["all"]["recall@5"]["mean"] == 1.0
    assert set(out) >= {"all", "object", "light"}


def test_cached_records_are_used_only_for_the_same_frames_dataset_and_models() -> None:
    config = {"files": {"detector": "aaa"}, "thresholds": {"x": 1}}
    records = [{"frame_id": "a"}, {"frame_id": "b"}]
    stored = {"version": "v1", "frame_ids": ["a", "b"], "config": config, "records": records}
    assert cache_matches(stored, "v1", ["a", "b"], config)
    assert not cache_matches(stored, "v1", ["b", "a"], config)  # same frames, different order
    assert not cache_matches(stored, "v1", ["a", "c"], config)
    assert not cache_matches(stored, "v2", ["a", "b"], config)
    assert not cache_matches(stored, "v1", ["a", "b"], {**config, "files": {"detector": "bbb"}})
    assert not cache_matches({**stored, "records": records[:1]}, "v1", ["a", "b"], config)
    swapped = [{"frame_id": "b"}, {"frame_id": "a"}]  # right frames, records in the wrong places
    assert not cache_matches({**stored, "records": swapped}, "v1", ["a", "b"], config)
    assert not cache_matches({}, "v1", ["a", "b"], config)
