"""A progress bar for training loops that stays quiet when output is not a terminal."""

from __future__ import annotations

import sys
from collections.abc import Iterable

from tqdm import tqdm


def _stderr_is_terminal() -> bool:
    isatty = getattr(sys.stderr, "isatty", None)  # stderr is None under pythonw
    return bool(isatty()) if callable(isatty) else False


def make_bar[T](
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool | None = None,
    leave: bool = False,
) -> tqdm[T]:
    """Wrap an iterable in a tqdm bar.

    enabled=None draws only when stderr is a terminal, so redirected logs and pytest runs get no
    bar noise; True or False overrides that. Callers show live numbers with bar.set_postfix(...).
    """
    show = _stderr_is_terminal() if enabled is None else enabled
    return tqdm(iterable, desc=desc, total=total, disable=not show, leave=leave, dynamic_ncols=True)
