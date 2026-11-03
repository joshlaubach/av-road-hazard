"""Tests for tools/eval_corpus_retrieval.py: the query builder and the pass rule, no models."""

from __future__ import annotations

from tools.eval_corpus_retrieval import MIN_MARGIN, corpus_queries, verdict


def image(name: str, *codes: str) -> dict[str, object]:
    return {"file": f"{name}.jpg", "boxes": [{"code": c} for c in codes]}


def test_relevance_comes_from_the_dataset_boxes() -> None:
    images = [image("a", "D40"), image("b", "D00", "D40"), image("c"), image("d", "D10")]
    queries = {q.text: q.relevant for q in corpus_queries(images)}
    assert queries["a road with a pothole"] == {"a", "b"}
    assert queries["a road with a longitudinal crack"] == {"b"}
    assert queries["a road with no damage"] == {"c"}


def test_a_class_with_no_images_gets_no_query() -> None:
    texts = [q.text for q in corpus_queries([image("a", "D40"), image("b")])]
    assert "a road with an alligator crack" not in texts and "a road with a pothole" in texts


def test_clean_query_is_labelled_clean() -> None:
    kinds = {q.text: q.kind for q in corpus_queries([image("a", "D10"), image("b")])}
    assert kinds["a road with no damage"] == "clean"
    assert kinds["a road with a transverse crack"] == "damage"


def scores(fused_mean: float, fused_low: float, random_mean: float) -> tuple[dict, dict]:
    fused = {"all": {"recall@5": {"mean": fused_mean, "low": fused_low, "high": 1.0}}}
    random = {"all": {"recall@5": {"mean": random_mean, "low": random_mean, "high": random_mean}}}
    return fused, random


def test_verdict_passes_with_margin_and_a_low_end_above_random() -> None:
    assert verdict(*scores(0.6, 0.4, 0.3))["passed"] is True


def test_verdict_fails_when_the_margin_is_too_small() -> None:
    assert verdict(*scores(0.3 + MIN_MARGIN - 0.01, 0.35, 0.3))["passed"] is False


def test_verdict_fails_when_the_range_reaches_down_to_random() -> None:
    assert verdict(*scores(0.7, 0.25, 0.3))["passed"] is False
