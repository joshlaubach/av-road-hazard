"""Pins which RDD2022 code means which defect (docs/SOURCE_REPOS.md, S-11).

Source of truth: Arya et al., "RDD2022: A multi-national image dataset for automatic Road Damage
Detection" (arXiv 2209.08538), Figs. 2, 3 and 5: D00 longitudinal crack, D10 transverse crack, D20
alligator crack, D40 pothole. The source repo's own mapping (D00 pothole, D10 longitudinal, D20
transverse, D40 alligator) contradicts the paper and must not be ported.

Two kinds of check. The offline tests run everywhere. The data test compares the annotation files
on disk with the paper's per-country box counts, so a code that is not the paper's code (a
re-exported or edited dataset) fails here before it reaches training. It is opt-in: set RDD2022_ROOT
to the folder that holds the country folders.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from defects.labels import CODES, NAMES, class_index
from defects.voc import EXPECTED_CODES, country_of, iter_annotations, parse_annotation

PINNED_NAMES = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole",
}

# Boxes per code (D00, D10, D20, D40) in the released training annotations, Fig. 5 of the paper.
PAPER_TRAIN_BOXES = {
    "Japan": (4049, 3979, 6199, 2243),
    "India": (1555, 68, 2021, 3187),
    "Czech": (988, 399, 161, 197),
    "Norway": (8570, 1730, 468, 461),
    "United_States": (6750, 3295, 834, 135),
}


def test_the_four_codes_keep_their_order_because_model_columns_follow_it() -> None:
    assert CODES == EXPECTED_CODES == ("D00", "D10", "D20", "D40")
    assert [class_index(code) for code in CODES] == [0, 1, 2, 3]


def test_the_pinned_names_cover_exactly_the_four_codes() -> None:
    assert tuple(PINNED_NAMES) == CODES


def test_the_names_in_labels_are_the_pinned_ones() -> None:
    """The source repo's wrong mapping (D00 pothole, ...) must never come back."""
    assert NAMES == PINNED_NAMES


@pytest.mark.data
def test_annotation_box_counts_match_the_paper_per_country() -> None:
    root_value = os.environ.get("RDD2022_ROOT")
    if not root_value:
        pytest.skip("set RDD2022_ROOT to the RDD2022 folder to check the annotations")
    root = Path(root_value)
    counts: dict[str, Counter[str]] = {}
    for xml_path in iter_annotations(root):
        _, boxes = parse_annotation(xml_path)
        counts.setdefault(country_of(xml_path, root), Counter()).update(box[0] for box in boxes)
    present = sorted(set(PAPER_TRAIN_BOXES) & set(counts))
    if not present:
        pytest.skip(f"none of {sorted(PAPER_TRAIN_BOXES)} found under {root}")
    found = {c: tuple(counts[c][code] for code in CODES) for c in present}
    expected = {c: PAPER_TRAIN_BOXES[c] for c in present}
    assert found == expected
