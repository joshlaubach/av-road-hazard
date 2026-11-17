"""eval/test_split.py: the pure parts of the sealed test-split scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from defects.labels import CODES
from defects.tile_index import ImageEntry, TileIndex, TileParams, image_tiles
from eval.test_split import (
    Protocol,
    hash_mismatches,
    measure,
    parse_protocol,
    score_tiles,
)

THRESHOLDS = (0.9, 0.75, 0.65, 0.75)
PARAMS = TileParams(tile_size=4, overlap=0.0, positive_overlap=0.25, negative_overlap=0.05)


def _protocol_raw(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "checkpoint_sha256": "a" * 64,
        "splits_sha256": "b" * 64,
        "tile_index_sha256": "c" * 64,
        "thresholds": dict(zip(CODES, THRESHOLDS, strict=True)),
        "pooled_sampling": {"seed": 42, "clean_per_positive": 1.0, "min_clean_per_image": 1},
        "measurements": ["pooled", "all_tiles"],
        "metrics": ["precision", "recall"],
        "rule": "Run once.",
        "registered_date": "2026-11-16",
    }
    return {**raw, **overrides}


def test_a_protocol_missing_a_field_is_refused() -> None:
    raw = _protocol_raw()
    del raw["splits_sha256"]
    with pytest.raises(ValueError, match="splits_sha256"):
        parse_protocol(raw)


def test_a_protocol_thresholds_must_name_exactly_the_classes() -> None:
    raw = _protocol_raw(thresholds={"D00": 0.9, "D10": 0.75, "D20": 0.65})
    with pytest.raises(ValueError, match="thresholds"):
        parse_protocol(raw)


def test_a_protocol_must_declare_the_two_measurements_in_order() -> None:
    with pytest.raises(ValueError, match="measurements"):
        parse_protocol(_protocol_raw(measurements=["all_tiles"]))


def test_a_protocol_parses_thresholds_in_class_order() -> None:
    protocol = parse_protocol(
        _protocol_raw(thresholds={"D40": 0.5, "D00": 0.9, "D10": 0.1, "D20": 0.2})
    )
    assert protocol == Protocol(
        checkpoint_sha256="a" * 64,
        splits_sha256="b" * 64,
        tile_index_sha256="c" * 64,
        thresholds=(0.9, 0.1, 0.2, 0.5),
        seed=42,
        clean_per_positive=1.0,
        min_clean=1,
    )


def test_hash_mismatches_name_each_file_that_differs() -> None:
    protocol = parse_protocol(_protocol_raw())
    actual = {"checkpoint": "a" * 64, "splits": "x" * 64, "tile_index": "c" * 64}
    assert hash_mismatches(protocol, actual) == ["splits"]
    assert hash_mismatches(protocol, {"checkpoint": "a" * 64, "splits": "b" * 64}) == ["tile_index"]


def _index() -> TileIndex:
    """One 16x16 image, 4x4 tiles (16 cells): two positives, one ignored cell, the rest clean."""
    entry = ImageEntry(
        size=(16, 16),
        positives={(0, 0): (1, 0, 0, 0), (0, 1): (0, 1, 0, 0)},
        ignored=frozenset({(3, 3)}),
    )
    return TileIndex(PARAMS, {"img": entry})


def _picture(_image_id: str) -> Image.Image:
    return Image.new("RGB", (16, 16), (90, 90, 90))


def test_scoring_calls_the_scorer_on_crops_in_batches_of_the_given_size() -> None:
    tiles = image_tiles(PARAMS, "img", _index().images["img"])
    tiles = [t for t in tiles if t.status != "ignore"]
    sizes: list[int] = []

    def scorer(crops: list[Image.Image]) -> np.ndarray:
        sizes.append(len(crops))
        return np.full((len(crops), len(CODES)), 0.1)

    probs = score_tiles(tiles, _index(), _picture, scorer, batch_size=4)
    assert len(probs) == 15
    assert sizes == [4, 4, 4, 3]


def test_scoring_refuses_a_scorer_that_returns_the_wrong_shape() -> None:
    tiles = [t for t in image_tiles(PARAMS, "img", _index().images["img"]) if t.status != "ignore"]

    def scorer(crops: list[Image.Image]) -> np.ndarray:
        return np.zeros((len(crops), 2))

    with pytest.raises(ValueError, match="shape"):
        score_tiles(tiles, _index(), _picture, scorer, batch_size=8)


def test_scoring_refuses_an_image_whose_size_differs_from_the_index() -> None:
    tiles = image_tiles(PARAMS, "img", _index().images["img"])

    def wrong_size(_image_id: str) -> Image.Image:
        return Image.new("RGB", (20, 16))

    with pytest.raises(ValueError, match="size"):
        score_tiles(tiles, _index(), wrong_size, lambda c: np.zeros((len(c), 4)), batch_size=8)


def test_measure_reports_no_precision_or_interval_where_there_is_nothing_to_divide() -> None:
    tiles = [t for t in image_tiles(PARAMS, "img", _index().images["img"]) if t.status != "ignore"]
    probs = {t.tile_id: np.zeros(len(CODES)) for t in tiles}  # nothing is ever flagged
    result = measure(tiles, probs, THRESHOLDS)
    d00 = result["classes"]["D00"]
    assert d00["flagged"] == 0 and d00["precision"] is None and d00["precision_interval"] is None
    assert d00["positives"] == 1 and d00["recall"] == 0.0
    assert d00["recall_interval"]["low90"] == 0.0
    assert result["tiles"] == 15 and d00["tiles"] == 15


def test_measure_gives_auroc_none_for_a_class_without_both_kinds_of_tile() -> None:
    tiles = [t for t in image_tiles(PARAMS, "img", _index().images["img"]) if t.status != "ignore"]
    probs = {t.tile_id: np.full(len(CODES), 0.95) for t in tiles}
    d20 = measure(tiles, probs, THRESHOLDS)["classes"]["D20"]
    assert d20["positives"] == 0 and d20["auroc"] is None
    assert d20["flagged"] == 15 and d20["precision"] == 0.0 and d20["recall"] is None
