"""Tile labels from box annotations: positive, clean negative, or ignored.

Purpose: RDD2022 labels say where a defect is in the whole photo, but the classifier trains on
tiles (defects/patches.py). A tile gets a label only when the boxes make it clear; an unclear tile
is ignored rather than guessed, because a wrong 0 teaches the model that a defect is not a defect.

Overlap of a box with a tile is intersection area / the smaller of the box area and the tile area.
That counts a small defect fully inside the tile as 1.0, and a long crack that covers the whole
tile as 1.0 too, which a plain IoU would not. Boxes are VOC boxes (code, x1, y1, x2, y2) in frame
pixels, the same units as Tile.box. Thresholds come from configs/defects_train.yaml.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from defects.labels import CODES, class_index
from defects.voc import Box

Status = Literal["positive", "negative", "ignore"]
POSITIVE: Status = "positive"
NEGATIVE: Status = "negative"
IGNORE: Status = "ignore"

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class TileLabel:
    """Status, plus one 0/1 per class code; labels are all zero unless the status is positive."""

    status: Status
    labels: tuple[int, ...]


def overlap_fraction(box: Rect, tile: Rect) -> float:
    """Intersection area over the smaller of the two areas; 0.0 when they do not overlap."""
    width = min(box[2], tile[2]) - max(box[0], tile[0])
    height = min(box[3], tile[3]) - max(box[1], tile[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((box[2] - box[0]) * (box[3] - box[1]), (tile[2] - tile[0]) * (tile[3] - tile[1]))
    return width * height / smaller


def label_tile(
    tile: Rect,
    boxes: Sequence[Box],
    *,
    positive_overlap: float,
    negative_overlap: float,
) -> TileLabel:
    """Label one tile from every box in its frame, including out-of-scope codes.

    A class is positive at overlap >= positive_overlap and clean at <= negative_overlap; in
    between it is unclear. A tile is then:
      - ignored if any in-scope class is unclear (calling it 0 would be a guess);
      - positive if any class is positive (out-of-scope boxes do not change that);
      - ignored if nothing is positive but an out-of-scope code (D44, D50, ...) overlaps more than
        negative_overlap, so a worn lane line or manhole cover never makes a "clean road";
      - a clean negative otherwise.
    """
    if not 0.0 <= negative_overlap < positive_overlap <= 1.0:
        raise ValueError(
            "need 0 <= negative_overlap < positive_overlap <= 1, "
            f"got negative_overlap={negative_overlap}, positive_overlap={positive_overlap}"
        )
    if tile[0] >= tile[2] or tile[1] >= tile[3]:
        raise ValueError(f"tile must have positive width and height, got {tile}")
    positive = [False] * len(CODES)
    unclear = [False] * len(CODES)
    out_of_scope_touches = False
    for code, x1, y1, x2, y2 in boxes:
        overlap = overlap_fraction((x1, y1, x2, y2), tile)
        if code not in CODES:
            out_of_scope_touches |= overlap > negative_overlap
        elif overlap >= positive_overlap:
            positive[class_index(code)] = True
        elif overlap > negative_overlap:
            unclear[class_index(code)] = True
    no_labels = (0,) * len(CODES)
    if any(u and not p for u, p in zip(unclear, positive, strict=True)):
        return TileLabel(IGNORE, no_labels)
    if any(positive):
        return TileLabel(POSITIVE, tuple(int(p) for p in positive))
    return TileLabel(IGNORE if out_of_scope_touches else NEGATIVE, no_labels)
