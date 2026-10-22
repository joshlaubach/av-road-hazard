"""Tile labeling on plain numbers; no image is read. Tile (0, 100, 100, 200) has area 10,000."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from defects.tile_labels import IGNORE, NEGATIVE, POSITIVE, TileLabel, label_tile, overlap_fraction
from defects.voc import Box

TILE = (0, 100, 100, 200)
POS, NEG = 0.25, 0.05


def label(boxes: list[Box], tile: tuple[int, int, int, int] = TILE) -> TileLabel:
    return label_tile(tile, boxes, positive_overlap=POS, negative_overlap=NEG)


def strip(code: str, rows: int) -> Box:
    """A box wider and taller than the tile that covers `rows` rows of it: overlap = rows/100."""
    return (code, 0, 0, 1000, 100 + rows)


def test_overlap_is_intersection_over_the_smaller_area() -> None:
    assert overlap_fraction((10, 110, 30, 130), TILE) == 1.0  # box fully inside the tile
    assert overlap_fraction((0, 0, 1000, 1000), TILE) == 1.0  # tile fully inside the box
    assert overlap_fraction(strip("D00", 24)[1:], TILE) == pytest.approx(0.24)
    assert overlap_fraction((500, 500, 600, 600), TILE) == 0.0


def test_small_box_fully_inside_is_positive_for_its_class() -> None:
    result = label([("D40", 10, 110, 30, 130)])
    assert (result.status, result.labels) == (POSITIVE, (0, 0, 0, 1))


def test_box_covering_the_whole_tile_is_positive() -> None:
    assert label([("D20", 0, 0, 1000, 1000)]) == TileLabel(POSITIVE, (0, 0, 1, 0))


def test_long_thin_crack_crossing_a_large_tile_is_positive() -> None:
    """A Norway-style 4000x300 D00 box across a 512 px tile covers 59% of the tile."""
    result = label_tile(
        (1000, 0, 1512, 512),
        [("D00", 0, 200, 4000, 500)],
        positive_overlap=POS,
        negative_overlap=NEG,
    )
    assert result == TileLabel(POSITIVE, (1, 0, 0, 0))


def test_positive_threshold_is_inclusive() -> None:
    assert label([strip("D10", 25)]).status == POSITIVE
    assert label([strip("D10", 24)]).status == IGNORE


def test_negative_threshold_is_inclusive_and_the_gap_is_ignored() -> None:
    assert label([strip("D00", 5)]) == TileLabel(NEGATIVE, (0, 0, 0, 0))
    assert label([strip("D00", 6)]).status == IGNORE


def test_no_boxes_or_distant_boxes_are_a_clean_negative() -> None:
    assert label([]) == TileLabel(NEGATIVE, (0, 0, 0, 0))
    assert label([("D00", 500, 500, 600, 600)]) == TileLabel(NEGATIVE, (0, 0, 0, 0))


def test_two_classes_can_both_be_positive() -> None:
    result = label([("D00", 0, 0, 1000, 160), ("D40", 10, 150, 30, 170)])
    assert (result.status, result.labels) == (POSITIVE, (1, 0, 0, 1))


def test_repeated_class_positive_wins_over_its_own_ambiguous_box() -> None:
    assert label([strip("D00", 10), strip("D00", 30)]) == TileLabel(POSITIVE, (1, 0, 0, 0))


def test_ambiguous_other_class_makes_a_positive_tile_ignored() -> None:
    """D10 might be present at 10%: calling it 0 would be a guess, so the tile is not used."""
    assert label([strip("D00", 50), strip("D10", 10)]).status == IGNORE


@pytest.mark.parametrize("code", ["D44", "D50", "D43", "D01", "D11", "D0w0"])
def test_out_of_scope_box_stops_an_otherwise_clean_tile_being_called_clean(code: str) -> None:
    assert label([(code, 10, 110, 30, 130)]).status == IGNORE
    assert label([strip(code, 6)]).status == IGNORE


def test_out_of_scope_box_barely_touching_does_not_spoil_a_clean_tile() -> None:
    assert label([strip("D44", 5)]) == TileLabel(NEGATIVE, (0, 0, 0, 0))


def test_out_of_scope_box_does_not_change_a_positive_tile() -> None:
    result = label([("D20", 0, 0, 1000, 1000), ("D44", 10, 110, 30, 130)])
    assert result == TileLabel(POSITIVE, (0, 0, 1, 0))


def test_ignored_and_negative_tiles_carry_no_positive_labels() -> None:
    assert label([strip("D00", 10)]).labels == (0, 0, 0, 0)


def test_zero_area_box_is_skipped_not_a_division_error() -> None:
    assert label([("D00", 10, 110, 10, 130)]) == TileLabel(NEGATIVE, (0, 0, 0, 0))


@pytest.mark.parametrize(
    ("positive", "negative"), [(0.0, 0.0), (0.05, 0.05), (0.05, 0.1), (1.5, 0.05), (0.25, -0.1)]
)
def test_invalid_thresholds_are_rejected(positive: float, negative: float) -> None:
    with pytest.raises(ValueError, match="overlap"):
        label_tile(TILE, [], positive_overlap=positive, negative_overlap=negative)


def test_empty_tile_is_rejected() -> None:
    with pytest.raises(ValueError, match="tile"):
        label_tile((10, 10, 10, 50), [], positive_overlap=POS, negative_overlap=NEG)


def test_training_config_thresholds_are_valid_and_drive_the_rule() -> None:
    cfg = yaml.safe_load(Path("configs/defects_train.yaml").read_text())
    result = label_tile(
        TILE,
        [strip("D00", 30)],
        positive_overlap=cfg["positive_overlap"],
        negative_overlap=cfg["negative_overlap"],
    )
    assert 0 <= cfg["negative_overlap"] < cfg["positive_overlap"] <= 1
    assert result.status == POSITIVE
