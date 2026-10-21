"""Per-code statistics and sample overlays for RDD2022 annotations.

Purpose: settle which D-code means which defect from the data, not from memory
(docs/SOURCE_REPOS.md, S-11). Run it where the dataset lives (RDD2022_ROOT), read the table and the
sample images, then pin the verified mapping in one place with a test.

Usage: python -m defects.label_check [RDD2022_ROOT] --out DIR [--samples 20]

The annotation files are Pascal VOC XML. Codes outside EXPECTED_CODES are counted and reported,
never dropped silently.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from common.config import get_config

# The four defect codes the project uses. Deliberately only codes, no names: which code means which
# defect is exactly what this script helps verify (docs/SOURCE_REPOS.md, S-11).
EXPECTED_CODES = ("D00", "D10", "D20", "D40")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
ALL = "ALL"  # pseudo-country used for the totals across all countries

Box = tuple[str, int, int, int, int]  # (code, xmin, ymin, xmax, ymax)


@dataclass
class CodeStats:
    """Running totals for one (country, code): counts plus the per-box shape measurements."""

    boxes: int = 0
    images: int = 0
    aspect_ratios: list[float] = field(default_factory=list)  # box width / box height
    area_shares: list[float] = field(default_factory=list)  # box area / image area


@dataclass
class Summary:
    """Everything summarize() learned: the stats table plus what it could not use."""

    stats: dict[tuple[str, str], CodeStats]  # (country, code); country ALL aggregates
    unexpected: Counter[str]  # object names outside EXPECTED_CODES
    invalid_boxes: int  # zero or negative width/height
    annotation_files: int


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


def summarize(root: Path) -> Summary:
    """Walk all annotations and collect, per (country, code), how many boxes and images there are
    and the shape of each box: aspect ratio (width / height) and share of the image area.

    Each box is added twice, once under its own country and once under ALL. Codes outside
    EXPECTED_CODES are counted in `unexpected` and zero-size boxes in `invalid_boxes`; neither is
    silently dropped, which was a problem in the source loader.
    """
    stats: dict[tuple[str, str], CodeStats] = {}
    unexpected: Counter[str] = Counter()
    invalid = 0
    files = 0
    for xml_path in iter_annotations(root):
        files += 1
        country = country_of(xml_path, root)
        size, boxes = parse_annotation(xml_path)
        seen: set[str] = set()
        for code, x1, y1, x2, y2 in boxes:
            if code not in EXPECTED_CODES:
                unexpected[code] += 1
                continue
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                invalid += 1
                continue
            for key in ((country, code), (ALL, code)):  # own country, and the overall total
                s = stats.setdefault(key, CodeStats())
                s.boxes += 1
                s.aspect_ratios.append(w / h)
                if size:  # area share needs the image size; skip it when the XML has none
                    s.area_shares.append(w * h / (size[0] * size[1]))
            seen.add(code)
        for code in seen:  # `images` counts each image once per code, however many boxes it has
            for key in ((country, code), (ALL, code)):
                stats[key].images += 1
    return Summary(stats, unexpected, invalid, files)


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """(25th percentile, median, 75th percentile). NaN for no data; the value itself for one point,
    since statistics.quantiles needs at least two."""
    if not values:
        return (float("nan"),) * 3
    if len(values) == 1:
        return (values[0],) * 3
    q1, q2, q3 = statistics.quantiles(values, n=4)
    return q1, q2, q3


HEADER = (
    "country", "code", "boxes", "images", "boxes_per_image",
    "aspect_p25", "aspect_median", "aspect_p75", "area_share_median",
)  # fmt: skip


def table_rows(summary: Summary) -> list[tuple[str, ...]]:
    """One row per (country, code) with the numbers already formatted as text for printing/CSV."""
    rows: list[tuple[str, ...]] = []
    for (country, code), s in sorted(summary.stats.items()):
        a25, a50, a75 = _quartiles(s.aspect_ratios)
        area = _quartiles(s.area_shares)[1]
        rows.append(
            (
                country, code, str(s.boxes), str(s.images), f"{s.boxes / s.images:.2f}",
                f"{a25:.2f}", f"{a50:.2f}", f"{a75:.2f}", f"{area:.4f}",
            )
        )  # fmt: skip
    return rows


def format_table(summary: Summary) -> str:
    """The table as aligned plain text, then file count, invalid boxes and unexpected codes."""
    rows = [HEADER, *table_rows(summary)]
    widths = [max(len(r[i]) for r in rows) for i in range(len(HEADER))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    lines.append("")
    lines.append(f"annotation files: {summary.annotation_files}")
    lines.append(f"zero/negative-size boxes skipped: {summary.invalid_boxes}")
    if summary.unexpected:
        names = ", ".join(f"{n or '<empty>'}={c}" for n, c in sorted(summary.unexpected.items()))
        lines.append(f"codes outside {EXPECTED_CODES} (not analysed): {names}")
    else:
        lines.append(f"codes outside {EXPECTED_CODES}: none")
    return "\n".join(lines)


def write_csv(summary: Summary, path: Path) -> None:
    """Save the same table as CSV so runs can be compared or loaded into a notebook."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(table_rows(summary))


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


def sample_overlays(
    root: Path, out_dir: Path, per_code: int = 20, seed: int | None = None
) -> dict[str, int]:
    """Save up to per_code images per code with that box drawn on it.

    Returns the number of files written per code. Boxes whose image cannot be found are skipped, so
    a count below per_code means missing images or too few boxes.
    """
    # A local Random instance, seeded: the same samples every run, and no change to global state.
    rng = random.Random(get_config()["seed"] if seed is None else seed)
    # Collect every valid box per code first, then shuffle and take the first per_code that have an
    # image on disk. Skipped boxes just mean fewer samples, reported by main().
    pool: dict[str, list[tuple[Path, Box]]] = {code: [] for code in EXPECTED_CODES}
    for xml_path in iter_annotations(root):
        for box in parse_annotation(xml_path)[1]:
            if box[0] in pool and box[3] > box[1] and box[4] > box[2]:
                pool[box[0]].append((xml_path, box))
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = dict.fromkeys(EXPECTED_CODES, 0)
    for code, candidates in pool.items():
        rng.shuffle(candidates)
        for xml_path, (_, x1, y1, x2, y2) in candidates:
            if written[code] >= per_code:
                break
            image_path = find_image(xml_path)
            if image_path is None:
                continue
            with Image.open(image_path) as img:
                canvas = img.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
            draw.text((x1 + 4, y1 + 4), f"{code} {country_of(xml_path, root)}", fill="red")
            canvas.save(out_dir / f"{code}_{written[code]:02d}.jpg")
            written[code] += 1
    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: print the table, write the CSV, save sample overlays.

    Returns 0 on success and 2 for a usage error (no folder given or not a folder). `argv` can be
    passed in by tests; None means use the real command line.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("root", nargs="?", help="RDD2022 folder (default: env RDD2022_ROOT)")
    parser.add_argument("--out", type=Path, required=True, help="folder for CSV and sample images")
    parser.add_argument("--samples", type=int, default=20, help="sample images per code")
    args = parser.parse_args(argv)
    raw_root = args.root or os.environ.get("RDD2022_ROOT")
    if not raw_root:
        print("error: pass the RDD2022 folder or set RDD2022_ROOT", file=sys.stderr)
        return 2
    root = Path(raw_root)
    if not root.is_dir():
        print(f"error: not a folder: {root}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(root)
    print(format_table(summary))
    write_csv(summary, args.out / "label_stats.csv")
    written = sample_overlays(root, args.out / "samples", args.samples)
    print("\nsample images written: " + ", ".join(f"{c}={n}" for c, n in written.items()))
    short = [c for c, n in written.items() if n < args.samples]
    if short:
        print(f"fewer than {args.samples} samples (missing images or too few boxes): {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
