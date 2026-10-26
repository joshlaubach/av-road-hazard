"""RDD2022 class codes and their names, verified against the paper and the annotation files."""

CODES = ("D00", "D10", "D20", "D40")
ALL = "ALL"  # pseudo-country used for the totals across all countries
# Arya et al., RDD2022 (arXiv 2209.08538). Pinned by tests/defects/test_label_mapping.py, which also
# compares the box counts in the annotation files with the paper.
NAMES: dict[str, str] = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole",
}


def class_index(code: str) -> int:
    return CODES.index(code)