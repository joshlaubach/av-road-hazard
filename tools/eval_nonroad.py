"""Measure how often the tile classifier flags tiles that are not road (nuScenes CAM_FRONT).

Usage: python -m tools.eval_nonroad --nuscenes-root DIR --checkpoint best.pt --out report.json
           [--version v1.0-mini]
Needs USE_STUBS=0 and the road model. Every frame is cut into the classifier's own tile grid, the
road model says how much of each tile is road, and tiles under the pipeline's road cut are scored
by the classifier at the shipped per-class thresholds (eval/nonroad.py defines the two bands).
The images stay on this machine; the report holds counts, rates and hashes only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.config import get_config
from defects.infer import defect_settings
from defects.labels import CODES
from defects.patches import tile_grid
from defects.road_region import DEFAULT_MIN_ROAD, road_coverage
from defects.tile_defects import tile_probabilities
from eval.nonroad import non_road_report
from eval.provenance import provenance


def score_frames(
    contexts: list[Any], root: Path, classifier: Any, thresholds: tuple[float, ...]
) -> dict[str, Any]:
    """Probabilities and road cover of every non-road tile in the frames, then the report."""
    probabilities: list[np.ndarray] = []
    cover: list[float] = []
    for context in contexts:
        mask = context.mask()
        with Image.open(root / context.path) as image:
            frame = np.asarray(image.convert("RGB"))
        height, width = frame.shape[:2]
        tiles = tile_grid(
            (width, height),
            int(classifier.cfg["tile_size"]),
            float(classifier.cfg["tile_overlap"]),
        )
        covers = [road_coverage(tile, mask) for tile in tiles]
        off = [t for t, c in zip(tiles, covers, strict=True) if c < DEFAULT_MIN_ROAD]
        if off:
            probabilities.append(tile_probabilities(classifier.model, frame, off, classifier.cfg))
            cover += [c for c in covers if c < DEFAULT_MIN_ROAD]
    table = np.concatenate(probabilities) if probabilities else np.zeros((0, len(CODES)))
    return {
        "frames": len(contexts),
        "non_road_tiles_scored": len(cover),
        **non_road_report(table, np.asarray(cover, dtype=float), thresholds, CODES),
    }


def main(argv: list[str] | None = None) -> int:
    from tools.eval_classifier import load_classifier, model_files, prepare_contexts

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    settings = defect_settings(get_config())
    classifier = load_classifier(args.checkpoint)
    contexts = prepare_contexts(args.nuscenes_root, args.version)
    report = score_frames(contexts, args.nuscenes_root, classifier, settings.thresholds)
    report["provenance"] = provenance(
        {"checkpoint": args.checkpoint, **model_files()},
        thresholds=dict(zip(CODES, settings.thresholds, strict=True)),
        repo=Path(__file__).resolve().parents[1],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("USE_STUBS", "0")
    sys.exit(main())
