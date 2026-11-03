"""Make the 20 Stellenbosch demo photos at full size with the car's dashboard cropped off.

Usage: python -m tools.prepare_stellenbosch --raw data/stellenbosch/raw
           [--fixtures tests/fixtures/stellenbosch] [--out data/stellenbosch/corpus]
The committed fixtures are 1280 px wide copies, and shrunk photos miss small damage (ROADMAP, the
real-photo check of 2026-10-26: "photos are analyzed at full size"). This picks the same 20 photos
by file name from the raw download, keeps the top CROP_HEIGHT rows of each (the lower part is the
dashboard), and writes JPEGs to --out, which is git-ignored (data/). The result is what
tools/index_corpus.py should index.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

# Rows kept from the 3680 by 2760 originals; the bottom 1000 rows show the dashboard. The fixtures
# are this crop shrunk to 1280 by 612 (1760 * 1280 / 3680 = 612), which fixes the number.
CROP_HEIGHT = 1760
JPEG_QUALITY = 95


def fixture_names(fixtures: Path) -> list[str]:
    return sorted(p.stem for p in fixtures.glob("*.jpg"))


def find_raw(raw: Path, names: Sequence[str]) -> dict[str, Path]:
    """The raw file for each name (any letter case of the extension); a missing one is an error."""
    by_stem = {p.stem: p for p in raw.rglob("*") if p.suffix.lower() == ".jpg"}
    missing = [n for n in names if n not in by_stem]
    if missing:
        raise SystemExit(f"not in the raw download: {', '.join(missing)}")
    return {n: by_stem[n] for n in names}


def crop_to_road(image: Image.Image) -> Image.Image:
    if image.height < CROP_HEIGHT:
        raise ValueError(f"image is only {image.height} rows tall, expected at least {CROP_HEIGHT}")
    return image.crop((0, 0, image.width, CROP_HEIGHT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/stellenbosch"))
    parser.add_argument("--out", type=Path, default=Path("data/stellenbosch/corpus"))
    args = parser.parse_args(argv)

    sources = find_raw(args.raw, fixture_names(args.fixtures))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        with Image.open(path) as image:
            crop_to_road(image.convert("RGB")).save(
                args.out / f"{name}.jpg", "JPEG", quality=JPEG_QUALITY
            )
    print(f"wrote {len(sources)} photos to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
