"""Structural VOC validation only; class meanings and source files are unchanged."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from defects.voc import Box


def _integer(text: str | None) -> int:
    value = float(text) if text is not None else float("nan")
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError("expected finite integer")
    return int(value)


def _coordinate(text: str | None) -> int:
    """Round a finite box coordinate to a pixel; Norway's annotations are sub-pixel (1138.46)."""
    value = float(text) if text is not None else float("nan")
    if not math.isfinite(value):
        raise ValueError("expected finite number")
    return round(value)


def read_annotation(
    path: Path,
) -> tuple[list[Box], tuple[int, int] | None, list[str]]:
    """Read every object, recording malformed annotations rather than dropping them.

    Missing size metadata is allowed: image dimensions are checked from the actual image.
    Missing boxes or coordinates are not converted into zeros or clean negatives. Fractional box
    coordinates are rounded to whole pixels (same as defects.voc); image sizes must be integers.
    """
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return [], None, ["unreadable_annotation"]
    if root.tag != "annotation":
        return [], None, ["invalid_annotation_root"]
    issues: list[str] = []
    size = None
    node = root.find("size")
    if node is not None:
        try:
            width, height = _integer(node.findtext("width")), _integer(node.findtext("height"))
            if width <= 0 or height <= 0:
                raise ValueError("nonpositive size")
            size = (width, height)
        except (ValueError, OverflowError):
            issues.append("invalid_image_size")
    boxes: list[Box] = []
    for obj in root.findall("object"):
        code = (obj.findtext("name") or "").strip()
        if not code:
            issues.append("missing_class_code")
        box = obj.find("bndbox")
        if box is None:
            issues.append("missing_box")
            continue
        try:
            x1, y1, x2, y2 = (
                _coordinate(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except (ValueError, OverflowError):
            issues.append("invalid_coordinates")
            continue
        if x1 >= x2 or y1 >= y2:
            issues.append("invalid_box")
        boxes.append((code, x1, y1, x2, y2))
    return boxes, size, issues


def image_issues(path: Path, size: tuple[int, int] | None, boxes: list[Box]) -> list[str]:
    """Verify image integrity, dimensions and box bounds without repairing annotations."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError):
        return ["unreadable_image"]
    issues = []
    if size is not None and size != (width, height):
        issues.append("image_size_mismatch")
    if any(x1 < 0 or y1 < 0 or x2 > width or y2 > height for _, x1, y1, x2, y2 in boxes):
        issues.append("box_outside_image")
    return issues
