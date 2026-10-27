"""defects.merge: neighboring flagged tiles of one class become one defect."""

from __future__ import annotations

import pytest

from defects.merge import merge_defects, merge_groups
from fusion.schema import Defect


def _d(
    box: tuple[float, float, float, float],
    *,
    name: str = "D00",
    confidence: float = 0.9,
    row: int = 0,
    col: int = 0,
) -> Defect:
    cls = {"D00": 1, "D10": 2, "D20": 3, "D40": 4}[name]
    return Defect(cls=cls, name=name, confidence=confidence, box=box, patch_row=row, patch_col=col)


def test_nothing_in_nothing_out_and_a_single_defect_is_unchanged() -> None:
    assert merge_defects([]) == []
    only = _d((0, 0, 10, 10))
    assert merge_defects([only]) == [only]


def test_two_overlapping_tiles_become_one_defect_with_the_union_box_and_top_confidence() -> None:
    weak = _d((0, 0, 100, 100), confidence=0.91, row=0, col=0)
    strong = _d((50, 0, 150, 100), confidence=0.97, row=0, col=1)
    (merged,) = merge_defects([weak, strong])
    assert merged.box == (0, 0, 150, 100)
    assert merged.confidence == 0.97
    assert (merged.name, merged.cls) == ("D00", 1)
    assert (merged.patch_row, merged.patch_col) == (0, 1)  # the strongest tile


def test_tiles_that_only_touch_edges_merge_but_a_gap_keeps_them_apart() -> None:
    a, b = _d((0, 0, 100, 100)), _d((100, 0, 200, 100))
    assert len(merge_defects([a, b])) == 1
    apart = _d((101, 0, 201, 100))
    assert len(merge_defects([a, apart])) == 2


def test_a_chain_merges_even_when_its_ends_do_not_touch() -> None:
    chain = [_d((0, 0, 100, 100)), _d((90, 0, 190, 100)), _d((180, 0, 280, 100))]
    (merged,) = merge_defects(chain)
    assert merged.box == (0, 0, 280, 100)


def test_the_result_does_not_depend_on_input_order() -> None:
    tiles = [_d((0, 0, 100, 100), col=0), _d((90, 0, 190, 100), col=1, confidence=0.95)]
    assert merge_defects(tiles)[0].box == merge_defects(tiles[::-1])[0].box
    assert merge_defects(tiles)[0].confidence == merge_defects(tiles[::-1])[0].confidence


def test_different_classes_never_merge_even_on_the_same_spot() -> None:
    crack, pothole = _d((0, 0, 100, 100), name="D00"), _d((0, 0, 100, 100), name="D40")
    merged = merge_defects([crack, pothole])
    assert sorted(d.name for d in merged) == ["D00", "D40"]


def test_two_separate_groups_stay_two_defects_in_order_of_first_appearance() -> None:
    left = [_d((0, 0, 100, 100)), _d((90, 0, 190, 100))]
    right = [_d((500, 0, 600, 100)), _d((590, 0, 690, 100))]
    merged = merge_defects([*right, *left])
    assert [d.box for d in merged] == [(500, 0, 690, 100), (0, 0, 190, 100)]


def test_a_strong_tile_tie_keeps_the_earliest_tile_and_inputs_are_not_changed() -> None:
    a = _d((0, 0, 100, 100), confidence=0.9, row=3, col=1)
    b = _d((50, 0, 150, 100), confidence=0.9, row=3, col=2)
    before = (a.model_copy(), b.model_copy())
    (merged,) = merge_defects([a, b])
    assert (merged.patch_row, merged.patch_col) == (3, 1)
    assert (a, b) == before


@pytest.mark.parametrize("count", [1, 5, 40])
def test_a_row_of_overlapping_tiles_is_always_one_defect(count: int) -> None:
    tiles = [_d((i * 50.0, 0, i * 50.0 + 100, 100)) for i in range(count)]
    assert len(merge_defects(tiles)) == 1


def test_the_groups_say_which_tiles_each_merged_defect_came_from() -> None:
    a, b = _d((0, 0, 100, 100)), _d((90, 0, 190, 100))
    far = _d((500, 0, 600, 100))
    groups = merge_groups([a, far, b])
    assert [len(g) for g in groups] == [2, 1]
    assert groups[0] == [a, b] and groups[1] == [far]
    assert [d.box for d in merge_defects([a, far, b])] == [(0, 0, 190, 100), (500, 0, 600, 100)]
