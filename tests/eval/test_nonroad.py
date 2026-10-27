"""eval/nonroad.py: how often the tile classifier flags tiles that are not road."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from eval.nonroad import BANDS, non_road_report

CODES = ("D00", "D10", "D20", "D40")
THRESHOLDS = (0.9, 0.75, 0.65, 0.75)


def _probs(rows: Sequence[Sequence[float]]) -> np.ndarray:
    return np.array(rows, dtype=float)


def test_a_tile_is_flagged_when_any_class_reaches_its_own_threshold() -> None:
    probs = _probs([(0.95, 0.0, 0.0, 0.0), (0.85, 0.0, 0.0, 0.0), (0.0, 0.8, 0.0, 0.0)])
    cover = np.array([0.0, 0.0, 0.0])
    report = non_road_report(probs, cover, THRESHOLDS, CODES)
    band = report["clearly_not_road"]
    assert band["tiles"] == 3
    assert band["flagged"]["count"] == 2  # D00 0.95 >= 0.9 and D10 0.8 >= 0.75; D00 0.85 < 0.9
    assert band["per_class"]["D00"]["count"] == 1 and band["per_class"]["D10"]["count"] == 1
    assert band["per_class"]["D20"]["count"] == 0


def test_tiles_are_sorted_into_bands_by_road_cover_and_road_tiles_are_left_out() -> None:
    probs = _probs([(1.0, 0, 0, 0)] * 5)
    cover = np.array([0.0, 0.049, 0.05, 0.39, 0.4])
    report = non_road_report(probs, cover, THRESHOLDS, CODES)
    assert report["clearly_not_road"]["tiles"] == 2  # 0.0 and 0.049
    assert report["partly_road_skipped"]["tiles"] == 2  # 0.05 and 0.39
    assert sum(b["tiles"] for b in report.values() if isinstance(b, dict)) == 4  # 0.4 is road


def test_rates_come_with_a_90_percent_range_and_an_empty_band_has_no_rate() -> None:
    probs = _probs([(1.0, 0, 0, 0)] + [(0.0, 0, 0, 0)] * 19)
    cover = np.zeros(20)
    flagged = non_road_report(probs, cover, THRESHOLDS, CODES)["clearly_not_road"]["flagged"]
    assert flagged["count"] == 1 and flagged["total"] == 20 and flagged["rate"] == 0.05
    assert flagged["low90"] < 0.05 < flagged["high90"]
    empty = non_road_report(probs, np.full(20, 0.9), THRESHOLDS, CODES)["clearly_not_road"]
    assert empty["tiles"] == 0 and empty["flagged"]["rate"] is None


def test_inputs_that_do_not_line_up_are_refused() -> None:
    with pytest.raises(ValueError, match="one coverage per tile"):
        non_road_report(_probs([(0, 0, 0, 0)]), np.zeros(2), THRESHOLDS, CODES)
    with pytest.raises(ValueError, match="one threshold per class"):
        non_road_report(_probs([(0, 0, 0, 0)]), np.zeros(1), (0.5,), CODES)
    with pytest.raises(ValueError, match="one column per class"):
        non_road_report(np.zeros((1, 3)), np.zeros(1), THRESHOLDS, CODES)


def test_the_bands_end_where_the_pipeline_starts_scoring() -> None:
    from defects.road_region import DEFAULT_MIN_ROAD

    assert BANDS["partly_road_skipped"][1] == DEFAULT_MIN_ROAD
