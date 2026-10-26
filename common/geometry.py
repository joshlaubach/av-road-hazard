"""Box geometry shared by evaluation and fusion."""

from __future__ import annotations

from collections.abc import Sequence


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes: overlap area divided by combined area (0 to 1).

    1.0 is identical boxes, 0.0 is no overlap. The max(..., 0) clamps handle disjoint boxes, whose
    raw intersection width or height would be negative.
    """
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
