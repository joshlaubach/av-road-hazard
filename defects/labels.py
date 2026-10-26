"""Verified class codes; human-readable names await the label check."""

CODES = ("D00", "D10", "D20", "D40")
ALL = "ALL"  # pseudo-country used for the totals across all countries
NAMES: dict[str, str | None] = dict.fromkeys(CODES)


class LabelNamesNotVerified(ValueError):
    """A name was requested before its mapping was verified."""


def class_index(code: str) -> int:
    return CODES.index(code)


def name_of(code: str) -> str:
    name = NAMES[code]
    if name is None:
        raise LabelNamesNotVerified(f"Name for {code} has not been verified")
    return name
