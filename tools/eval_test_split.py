"""Score the RDD2022 test split exactly once, with everything frozen.

See eval/test_split_protocol.json for the protocol.

Usage: python -m tools.eval_test_split --protocol eval/test_split_protocol.json --checkpoint PATH
           --splits PATH --tile-index PATH --sample-index PATH --out eval/test_split_result.json
Refuses, before any image or model is loaded, when: the output file already exists (sealed; there
is no --force); the protocol is not tracked by git or has uncommitted changes, or any tracked file
in its repository does; or the SHA-256 of the checkpoint, splits, tile index or sample index is not
the one the protocol records. Then it scores only the test images, in two measurements: "pooled"
(the training pool, comparable to the validation row) and "all_tiles" (every non-ignored tile).
Thresholds come from the protocol. The result holds numbers, hashes and git provenance only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.hashing import file_sha256
from defects.labels import CODES
from defects.splits import read_splits
from defects.tile_index import read_tile_index
from eval.provenance import provenance
from eval.test_split import (
    DEFAULT_BATCH,
    MEASUREMENTS,
    Protocol,
    Scorer,
    hash_mismatches,
    measure,
    parse_protocol,
    score_tiles,
    scored_tiles,
)

SCHEMA_VERSION = 1


class Refusal(Exception):
    """A guard said no; the message says which one and why. Nothing was loaded."""


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def check_out_absent(out: Path) -> None:
    if out.exists():
        raise Refusal(
            f"{out} already exists: the test split is scored once and the result is sealed"
        )


def check_protocol_committed(protocol: Path) -> Path:
    """Refuse unless the protocol is committed and nothing tracked has changed; returns the repo."""
    folder, name = protocol.parent, protocol.name
    tracked = _git(folder, "ls-files", "--error-unmatch", "--", name)
    if tracked is None or tracked.returncode != 0:
        raise Refusal(f"{protocol} is not tracked by git: commit the protocol before scoring")
    unchanged = _git(folder, "diff", "--quiet", "HEAD", "--", name)
    if unchanged is None or unchanged.returncode != 0:
        raise Refusal(f"{protocol} has uncommitted changes: commit it before scoring")
    top = _git(folder, "rev-parse", "--show-toplevel")
    if top is None or top.returncode != 0:
        raise Refusal("the protocol is not inside a git repository")
    repo = Path(top.stdout.strip())
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty is None or dirty.returncode != 0 or dirty.stdout.strip():
        raise Refusal("the working tree has uncommitted changes to tracked files: commit or stash")
    return repo


def _sample_paths(sample_index: Path) -> dict[str, Path]:
    rows = json.loads(sample_index.read_text())
    return {row["id"]: sample_index.parent / row["image_path"] for row in rows}


def _open_rgb(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def load_scorer(checkpoint: Path) -> Scorer:
    """The checkpoint with its sibling config, scored through the evaluation transform."""
    from defects.model import load_checkpoint
    from defects.runtime import select_device
    from tools.eval_classifier import Context, score_crops

    config = json.loads(checkpoint.with_suffix(".json").read_text())["config"]
    device = select_device()
    model = load_checkpoint(checkpoint, config, device=device).eval()
    context = Context(cfg=config, base=checkpoint.parent)
    return lambda crops: score_crops(model, device, crops, context)


def _protocol_from(path: Path) -> Protocol:
    try:
        return parse_protocol(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refusal(f"protocol could not be read: {exc}") from exc


def run(
    *,
    protocol_path: Path,
    checkpoint: Path,
    splits: Path,
    tile_index: Path,
    sample_index: Path,
    out: Path,
    load_scorer: Callable[[Path], Scorer] = load_scorer,
    open_image: Callable[[Path], Any] = _open_rgb,
    batch_size: int = DEFAULT_BATCH,
) -> dict[str, Any]:
    """Run the guards, then the one scoring; write out and return the result."""
    protocol_path = protocol_path.resolve()
    out = out.resolve()
    check_out_absent(out)
    repo = check_protocol_committed(protocol_path)
    protocol = _protocol_from(protocol_path)
    actual = {
        "checkpoint": file_sha256(checkpoint),
        "splits": file_sha256(splits),
        "tile_index": file_sha256(tile_index),
        "sample_index": file_sha256(sample_index),
    }
    mismatched = hash_mismatches(protocol, actual)
    if mismatched:
        raise Refusal(f"SHA-256 does not match the protocol for: {', '.join(mismatched)}")

    test_ids = sorted(read_splits(splits).test)
    index = read_tile_index(tile_index)
    absent = [i for i in test_ids if i not in index.images]
    if absent:
        raise Refusal(f"{len(absent)} test images are missing from the tile index")
    paths = _sample_paths(sample_index)
    unknown = [i for i in test_ids if i not in paths]
    if unknown:
        raise Refusal(f"{len(unknown)} test images are missing from the sample index")

    tiles = scored_tiles(index, test_ids, protocol)
    scorer = load_scorer(checkpoint)
    probs = score_tiles(
        tiles["all_tiles"],
        index,
        lambda image_id: open_image(paths[image_id]),
        scorer,
        batch_size=batch_size,
    )
    measurements = {name: measure(tiles[name], probs, protocol.thresholds) for name in MEASUREMENTS}
    result = {
        "schema_version": SCHEMA_VERSION,
        "images": len(test_ids),
        "measurements": measurements,
        "provenance": provenance(
            {
                "protocol": protocol_path,
                "checkpoint": checkpoint,
                "splits": splits,
                "tile_index": tile_index,
                "sample_index": sample_index,
            },
            thresholds=dict(zip(CODES, protocol.thresholds, strict=True)),
            repo=repo,
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:  # refuses to overwrite even if one appeared
        handle.write(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "--protocol",
        "--checkpoint",
        "--splits",
        "--tile-index",
        "--sample-index",
        "--out",
    ):
        parser.add_argument(flag, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(
            protocol_path=args.protocol,
            checkpoint=args.checkpoint,
            splits=args.splits,
            tile_index=args.tile_index,
            sample_index=args.sample_index,
            out=args.out,
        )
    except Refusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except FileExistsError:
        print(f"refused: {args.out} already exists: the test split is scored once", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
