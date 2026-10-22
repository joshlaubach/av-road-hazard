"""Crop and cache the training tiles of every non-test image.

Usage: python -m tools.build_tile_cache --sample-index INDEX.json --tile-index TILES.json
       --splits SPLITS.json --config CFG.yaml --cache-dir DIR
The tiles are the training pool (all positives plus the capped, seeded clean sample; see
defects/tile_index.py) of every image outside the final test set. Pools are chosen per image with
the global seed, so one cache serves every fold. Re-running only writes what is missing. The
cache holds RDD2022 pixels: keep it under the gitignored data/ folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from common.config import get_config
from defects.splits import read_splits
from defects.tile_cache import DEFAULT_WORKERS, build_tile_cache
from defects.tile_index import read_tile_index, training_pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--tile-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    index = read_tile_index(args.tile_index)
    splits = read_splits(args.splits)
    non_test = sorted({i for part in splits.folds[0] for i in part})
    if set(non_test) & set(splits.test):
        parser.error("splits are inconsistent: the test set overlaps the folds")
    pool = training_pool(
        index,
        non_test,
        clean_per_positive=float(cfg["clean_per_positive"]),
        min_clean=int(cfg["min_clean_per_image"]),
        seed=int(get_config()["seed"]),
    )
    rows = json.loads(args.sample_index.read_text())
    paths = {r["id"]: (args.sample_index.parent / r["image_path"]).resolve() for r in rows}
    written = build_tile_cache(
        pool,
        paths,
        args.cache_dir,
        index.params,
        quality=int(cfg["cache_jpeg_quality"]),
        workers=args.workers,
    )
    print(f"{len(pool)} training tiles from {len(non_test)} images; {written} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
