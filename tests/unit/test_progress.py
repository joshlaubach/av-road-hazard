"""The progress bar helper: transparent to iteration, silent when output is not a terminal."""

import sys

import pytest

pytest.importorskip("tqdm")

from common.progress import make_bar


def test_bar_yields_every_item_unchanged() -> None:
    """Wrapping an iterable never drops, reorders or alters items."""
    assert list(make_bar([3, 1, 2], desc="t", enabled=False)) == [3, 1, 2]


def test_auto_mode_is_silent_without_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Redirected output gets no bar and no escape-code noise."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    bar = make_bar(range(3), desc="t")
    assert bar.disable is True
    list(bar)
    assert capsys.readouterr().err == ""


def test_auto_mode_draws_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a TTY on stderr the bar is on by default."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    bar = make_bar(range(3), desc="t")
    assert bar.disable is False
    bar.close()


def test_auto_mode_survives_a_missing_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """pythonw sets sys.stderr to None; that must read as 'not a terminal', not crash."""
    monkeypatch.setattr(sys, "stderr", None)
    bar = make_bar(range(2), desc="t")
    assert bar.disable is True


@pytest.mark.parametrize("enabled", [True, False])
def test_explicit_flag_overrides_terminal_detection(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """enabled=True forces a bar without a TTY; enabled=False silences one with a TTY."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: not enabled)
    bar = make_bar(range(2), desc="t", enabled=enabled)
    assert bar.disable is (not enabled)
    bar.close()


def test_bar_shows_live_values_in_the_postfix() -> None:
    """set_postfix is the hook the training loops use to show the running loss."""
    bar = make_bar(range(2), desc="t", enabled=True)
    bar.set_postfix(loss=0.5)
    assert "loss" in bar.postfix and "0.5" in bar.postfix
    bar.close()
