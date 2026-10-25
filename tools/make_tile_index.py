"""Build the tile index for the validated RDD2022 samples.

Usage: python -m tools.make_tile_index --root RAW --sample-index INDEX.json --config CFG.yaml
       --out TILE_INDEX.json
RAW is the dataset root the sample ids are relative to; the sample index is the file written by
tools/validate_annotations.py; the config supplies tile_size, tile_overlap, positive_overlap and
negative_overlap. Boxes are read from the annotation XML, including out-of-scope codes, because
those decide which tiles are ignored. The index holds only numbers and ids, never pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from defects.tile_index import (
    ImageEntry,
    TileIndex,
    TileParams,
    entry_from_tiles,
    label_image_tiles,
    write_tile_index,
)
from defects.validation import read_annotation


def build_entry(
    root: Path, sample_index: Path, row: dict[str, object], params: TileParams
) -> ImageEntry:
    """Tile one validated image; its size comes from the annotation, else from the image file."""
    image_id = str(row["id"])
    boxes, size, issues = read_annotation(root / f"{image_id}.xml")
    if issues:
        raise ValueError(f"{image_id}: sample index lists an invalid annotation: {issues}")
    if size is None:
        with Image.open(sample_index.parent / str(row["image_path"])) as image:
            size = image.size
    return entry_from_tiles(size, label_image_tiles(image_id, size, boxes, params))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error("root must be an existing dataset directory")
    if args.out.suffix.lower() != ".json" or args.out.resolve() == args.sample_index.resolve():
        parser.error("out must be a .json file different from the sample index")
    params = TileParams.from_config(yaml.safe_load(args.config.read_text()))
    rows = json.loads(args.sample_index.read_text())
    images = {
        str(row["id"]): build_entry(args.root, args.sample_index, row, params) for row in rows
    }
    write_tile_index(TileIndex(params, images), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
