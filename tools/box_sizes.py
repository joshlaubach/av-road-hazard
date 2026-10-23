"""How big are the defects, in pixels, in each country? Read-only.

Usage: python -m tools.box_sizes --root RDD2022_RAW --tile-size 384 [--out SIZES.json]
Tiles have the same pixel size everywhere but cover a different real-world area when cameras
differ. This reports, per country, the median defect width, height and longer side in pixels, the
median image width, and the share of defects whose longer side exceeds one tile. If a country's
defects are several times larger than the others', its tiles show only pieces of them. Only the
four in-scope codes with a positive width and height count. Prints a table and writes JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from defects.voc import ALL, EXPECTED_CODES, country_of, iter_annotations, parse_annotation


def _median(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def summarize_sizes(root: Path, *, tile_size: int) -> dict[str, dict[str, Any]]:
    """Defect size statistics per country folder under root, plus an ALL total."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    widths: dict[str, list[int]] = defaultdict(list)
    heights: dict[str, list[int]] = defaultdict(list)
    image_widths: dict[str, list[int]] = defaultdict(list)
    images: dict[str, int] = defaultdict(int)
    for xml_path in iter_annotations(root):
        size, boxes = parse_annotation(xml_path)
        for country in (country_of(xml_path, root), ALL):
            images[country] += 1
            if size is not None:
                image_widths[country].append(size[0])
            for code, x1, y1, x2, y2 in boxes:
                if code in EXPECTED_CODES and x2 > x1 and y2 > y1:
                    widths[country].append(x2 - x1)
                    heights[country].append(y2 - y1)
    report: dict[str, dict[str, Any]] = {}
    for country in sorted(images):
        sides = [max(w, h) for w, h in zip(widths[country], heights[country], strict=True)]
        report[country] = {
            "images": images[country],
            "boxes": len(sides),
            "median_width": _median(widths[country]),
            "median_height": _median(heights[country]),
            "median_side": _median(sides),
            "median_image_width": _median(image_widths[country]),
            "share_larger_than_tile": (
                sum(side > tile_size for side in sides) / len(sides) if sides else None
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = summarize_sizes(args.root, tile_size=args.tile_size)
    print(
        f"{'country':<14}{'images':>8}{'boxes':>8}{'med w':>8}{'med h':>8}{'med side':>10}"
        f"{'img w':>8}{'> tile':>8}"
    )
    for country, row in report.items():
        share = row["share_larger_than_tile"]
        print(
            f"{country:<14}{row['images']:>8}{row['boxes']:>8}{row['median_width'] or 0:>8.0f}"
            f"{row['median_height'] or 0:>8.0f}{row['median_side'] or 0:>10.0f}"
            f"{row['median_image_width'] or 0:>8.0f}{'-' if share is None else f'{share:.1%}':>8}"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
