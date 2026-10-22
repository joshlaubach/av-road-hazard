"""Group samples by image content so near-duplicate images never straddle a split.

Purpose: `python -m defects.train splits` needs an id -> group mapping and has no safe default.
RDD2022 filenames are anonymized shuffled indices (consecutive ids are no more alike than random
pairs), so the only usable grouping signal is the pixels: images whose perceptual hashes are close
are chained into one group, and defects/splits.py keeps each group on one side of every split.

Usage: python -m tools.make_groups --sample-index INDEX.json --out GROUPS.json [--report R.json]
The sample index is the file written by defects/validate_annotations.py. Groups never span
countries (the first folder of the sample id). Library code never reads dataset images; only main()
does, and it never copies them.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from defects.voc import ALL

HASH_SIDE = 16  # difference hash of a 17x16 grayscale thumbnail: 16x16 comparisons
HASH_BITS = HASH_SIDE * HASH_SIDE
NO_NEIGHBOUR = HASH_BITS  # farther than any real distance; used when a country has one image
DEFAULT_MAX_BITS = 24  # 0.10 of the hash; check the report's nearest-distance histogram
BLOCK_ROWS = 128  # rows compared against all hashes at once; bounds memory to BLOCK_ROWS * n * 32 B
HISTOGRAM_STEP = 8
DEFAULT_WORKERS = 8
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

Hashes = NDArray[np.uint8]


def dhash_bits(image: Image.Image) -> Hashes:
    """Difference hash packed into HASH_BITS / 8 bytes: is each pixel brighter than its left one."""
    gray = image.convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    return np.packbits((pixels[:, 1:] > pixels[:, :-1]).ravel())


def cluster_near_duplicates(
    hashes: Hashes, max_bits: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Cluster labels (0..k-1, by first member) and each image's nearest-neighbour distance.

    Images within max_bits differing hash bits are linked, and links chain: A~B and B~C put A, B
    and C together. That is deliberate, since a split only needs one chain end to leak.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nearest = np.full(n, NO_NEIGHBOUR, dtype=np.int64)
    for start in range(0, n, BLOCK_ROWS):
        block = hashes[start : start + BLOCK_ROWS]
        dist = POPCOUNT[block[:, None, :] ^ hashes[None, :, :]].sum(axis=2, dtype=np.int64)
        rows = np.arange(len(block))
        dist[rows, start + rows] = NO_NEIGHBOUR  # an image is not its own neighbour
        nearest[start : start + len(block)] = dist.min(axis=1)
        for i, j in zip(*np.nonzero(dist <= max_bits), strict=True):
            parent[find(start + int(i))] = find(int(j))
    first_seen: dict[int, int] = {}
    labels = np.array([first_seen.setdefault(find(i), len(first_seen)) for i in range(n)])
    return labels, nearest


def _country(sample_id: str) -> str:
    return sample_id.split("/", 1)[0] if "/" in sample_id else ALL


def group_ids(
    ids: Sequence[str], hashes: Hashes, max_bits: int
) -> tuple[dict[str, str], NDArray[np.int64]]:
    """Map every id to a "country:first-member-id" group, plus nearest distances in id order."""
    if len(ids) != len(hashes):
        raise ValueError("Need one hash per id")
    by_country: dict[str, list[int]] = defaultdict(list)
    for position, sample_id in enumerate(ids):
        by_country[_country(sample_id)].append(position)
    groups: dict[str, str] = {}
    nearest = np.full(len(ids), NO_NEIGHBOUR, dtype=np.int64)
    for country, positions in by_country.items():
        labels, near = cluster_near_duplicates(hashes[positions], max_bits)
        members: dict[int, list[str]] = defaultdict(list)
        for position, label in zip(positions, labels, strict=True):
            members[int(label)].append(ids[position])
        for member_ids in members.values():
            name = f"{country}:{min(member_ids)}"
            groups.update(dict.fromkeys(member_ids, name))
        nearest[positions] = near
    return groups, nearest


def _hash_file(sample_id: str, path: Path) -> Hashes:
    """Hash one image; a draft decode keeps multi-megapixel JPEGs fast. Failures name the id."""
    try:
        with Image.open(path) as image:
            image.draft("L", (HASH_SIDE * 8, HASH_SIDE * 8))
            return dhash_bits(image)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError(f"{sample_id}: unreadable image {path} ({exc})") from exc


def read_index(path: Path) -> list[tuple[str, Path]]:
    """(id, image path) pairs from a sample index; image paths are relative to the index file."""
    rows = json.loads(path.read_text())
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Sample index has duplicate ids")
    return [(row["id"], path.parent / row["image_path"]) for row in rows]


def build_report(
    groups: dict[str, str], nearest: NDArray[np.int64], max_bits: int
) -> dict[str, Any]:
    """Counts plus the nearest-neighbour histogram used to check that max_bits sits in a valley."""
    sizes: dict[str, int] = defaultdict(int)
    for name in groups.values():
        sizes[name] += 1
    histogram = {f"{lo}-{lo + HISTOGRAM_STEP - 1}": 0 for lo in range(0, HASH_BITS, HISTOGRAM_STEP)}
    histogram["none"] = 0
    for value in nearest.tolist():
        key = "none" if value >= NO_NEIGHBOUR else _bin(value)
        histogram[key] += 1
    return {
        "images": len(groups),
        "groups": len(sizes),
        "max_bits": max_bits,
        "images_in_duplicate_groups": sum(n for n in sizes.values() if n > 1),
        "largest_group": max(sizes.values(), default=0),
        "nearest_distance_histogram": histogram,
    }


def _bin(value: int) -> str:
    lo = value // HISTOGRAM_STEP * HISTOGRAM_STEP
    return f"{lo}-{lo + HISTOGRAM_STEP - 1}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-bits", type=int, default=DEFAULT_MAX_BITS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    outputs = [p for p in (args.out, args.report) if p is not None]
    if any(p.suffix.lower() != ".json" for p in outputs):
        parser.error("output files must have a .json extension")
    if len({p.resolve() for p in [args.sample_index, *outputs]}) != len(outputs) + 1:
        parser.error("sample-index, out and report must be different files")
    if not 0 <= args.max_bits < HASH_BITS:
        parser.error(f"max-bits must be in [0, {HASH_BITS})")
    pairs = read_index(args.sample_index)
    ids = [sample_id for sample_id, _ in pairs]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = np.stack(list(pool.map(lambda pair: _hash_file(*pair), pairs)))
    groups, nearest = group_ids(ids, hashes, args.max_bits)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n")
    if args.report is not None:
        args.report.write_text(
            json.dumps(build_report(groups, nearest, args.max_bits), indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
