"""Reading RDD2022 Pascal VOC annotations: codes, boxes and image lookup.

Purpose: the small shared layer under dataset building, annotation validation and the label-check
tool (tools/label_check.py). Library code only; it prints nothing and writes nothing.

The annotation files are Pascal VOC XML. Codes outside EXPECTED_CODES are the caller's to count and
report, never to drop silently.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

# The four defect codes the project uses. Deliberately only codes, no names: which code means which
# defect is exactly what this script helps verify (docs/SOURCE_REPOS.md, S-11).
EXPECTED_CODES = ("D00", "D10", "D20", "D40")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
ALL = "ALL"  # pseudo-country used for the totals across all countries

Box = tuple[str, int, int, int, int]  # (code, xmin, ymin, xmax, ymax)


def iter_annotations(root: Path) -> Iterator[Path]:
    """Every annotation file under root, in a stable order so runs are repeatable.

    rglob keeps this independent of the exact folder layout, which is not confirmed yet.
    """
    yield from sorted(root.rglob("*.xml"))


def country_of(xml_path: Path, root: Path) -> str:
    """The first folder below root (the country in the RDD2022 layout), or ALL for flat folders."""
    parts = xml_path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else ALL


def parse_annotation(xml_path: Path) -> tuple[tuple[int, int] | None, list[Box]]:
    """Return ((width, height) or None, boxes). Every <object> is returned, whatever its name."""
    # These are the dataset's own files, but the XML is still parsed defensively: missing tags give
    # 0 or None instead of an exception, and the caller counts such boxes as invalid.
    root = ET.parse(xml_path).getroot()
    size: tuple[int, int] | None = None
    size_node = root.find("size")
    if size_node is not None:
        w = _num(size_node.findtext("width"))
        h = _num(size_node.findtext("height"))
        if w > 0 and h > 0:
            size = (w, h)
    boxes: list[Box] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        boxes.append(
            (
                (obj.findtext("name") or "").strip(),
                _num(bnd.findtext("xmin")),
                _num(bnd.findtext("ymin")),
                _num(bnd.findtext("xmax")),
                _num(bnd.findtext("ymax")),
            )
        )
    return size, boxes


def _num(text: str | None) -> int:
    """Parse an XML number (it may be written as 12 or 12.0) to an int; missing or empty is 0."""
    return round(float(text)) if text else 0


def find_image(xml_path: Path) -> Path | None:
    """Look next to the XML, then in a sibling images/ folder (layout is confirmed on real data)."""
    # Candidate folders: next to the XML, then an `images` folder beside each of its three nearest
    # parents (annotations/xmls/x.xml -> train/images/x.jpg).
    for folder in (xml_path.parent, *(p / "images" for p in xml_path.parents[:3])):
        for suffix in IMAGE_SUFFIXES:
            candidate = folder / (xml_path.stem + suffix)
            if candidate.is_file():
                return candidate
    return None
