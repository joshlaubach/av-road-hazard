"""Merge neighboring flagged tiles of one class into a single defect.

The classifier scores overlapping square tiles, so one long crack lights up several tiles in a row.
Two defects of the same class whose boxes overlap or touch are one defect, and so is a chain of
them: A touches B and B touches C makes one defect even if A and C are apart. The merged box is the
smallest rectangle that covers the group and may be large (a crack along the whole road gives a
wide box). Its confidence is the strongest tile's, not an average, so weak neighbors cannot dilute
a strong tile. Different classes never merge: a pothole and a crack on the same spot keep their own
severities.
"""

from __future__ import annotations

from collections.abc import Sequence

from fusion.schema import Defect


def _touch(a: Defect, b: Defect) -> bool:
    """Whether the closed boxes overlap or share an edge or corner."""
    return (
        a.box[0] <= b.box[2]
        and b.box[0] <= a.box[2]
        and a.box[1] <= b.box[3]
        and b.box[1] <= a.box[3]
    )


def _group_roots(defects: Sequence[Defect]) -> list[int]:
    """For each defect, the lowest index in its group of same-class, chained neighbors."""
    root = list(range(len(defects)))

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for i, a in enumerate(defects):
        for j in range(i + 1, len(defects)):
            b = defects[j]
            if a.cls == b.cls and _touch(a, b):
                low, high = sorted((find(i), find(j)))
                root[high] = low
    return [find(i) for i in range(len(defects))]


def merge_defects(defects: Sequence[Defect]) -> list[Defect]:
    """The defects with neighboring same-class tiles merged; groups keep their first-seen order."""
    groups: dict[int, list[Defect]] = {}
    for defect, root in zip(defects, _group_roots(defects), strict=True):
        groups.setdefault(root, []).append(defect)
    merged: list[Defect] = []
    for members in groups.values():
        strongest = max(members, key=lambda d: d.confidence)  # the first of equals wins
        merged.append(
            strongest.model_copy(
                update={
                    "box": (
                        min(d.box[0] for d in members),
                        min(d.box[1] for d in members),
                        max(d.box[2] for d in members),
                        max(d.box[3] for d in members),
                    )
                }
            )
        )
    return merged
