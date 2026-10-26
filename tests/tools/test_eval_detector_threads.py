"""Timing at a thread count must not change the process's thread count for later work."""

import pytest

from tools.eval_detector import ladder_rungs, thread_limit

torch = pytest.importorskip("torch")


def test_thread_limit_sets_and_restores_the_count() -> None:
    before = torch.get_num_threads()
    target = 1 if before != 1 else 2

    with thread_limit(target):
        assert torch.get_num_threads() == target

    assert torch.get_num_threads() == before


def test_thread_limit_restores_after_an_error() -> None:
    before = torch.get_num_threads()

    with pytest.raises(RuntimeError), thread_limit(1 if before != 1 else 2):
        raise RuntimeError("boom")

    assert torch.get_num_threads() == before


def test_thread_limit_none_changes_nothing() -> None:
    before = torch.get_num_threads()

    with thread_limit(None):
        assert torch.get_num_threads() == before


def test_ladder_rungs_are_unique_and_highest_first() -> None:
    assert ladder_rungs([4, 24, 4, 2]) == [24, 4, 2]


def test_ladder_rungs_drop_non_positive_counts() -> None:
    assert ladder_rungs([0, 2, -1]) == [2]
