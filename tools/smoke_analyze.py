"""Run the real pipeline over a folder of images and report what worked and what did not.

Usage: python -m tools.smoke_analyze IMAGE_FOLDER --out DIR
Needs USE_STUBS=0 and the models (detector, classifier checkpoint, road model). Writes, per image,
an overlay PNG and the record JSON, named after the full file name (so frame.jpg and frame.png
do not collide), plus summary.json. Counts are kept apart on purpose: a record
can be valid while the road model failed (every in_road is then unknown), and a missing model is a
failure, not an empty result. Exit code 0 only if every image gave a record with a road mask.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from api.pipeline import InvalidImage, Pipeline
from common.config import get_config
from defects.infer import DefectsUnavailable
from perception.detector import DetectorUnavailable
from tools.overlay import draw

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def run_smoke(pipeline: Pipeline, images: list[Path], out: Path) -> dict[str, Any]:
    """Analyze every image; save overlays and records; return the counts and per-image outcome."""
    out.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, str] = {}
    for path in images:
        try:
            analysis = pipeline.analyze_checked(path.read_bytes())
        except InvalidImage:
            outcome[path.name] = "invalid_image"
            continue
        except DetectorUnavailable:
            outcome[path.name] = "detector_unavailable"
            continue
        except DefectsUnavailable:
            outcome[path.name] = "defects_unavailable"
            continue
        outcome[path.name] = "ok" if analysis.road_mask_used else "road_unknown"
        (out / f"{path.name}.record.json").write_text(
            analysis.record.model_dump_json(indent=2), encoding="utf-8"
        )
        with Image.open(path) as image:
            draw(image, analysis.record).save(out / f"{path.name}.overlay.png")
    counts = Counter(outcome.values())
    summary: dict[str, Any] = {
        "images": len(images),
        "valid": counts["ok"] + counts["road_unknown"],
        "road_unknown": counts["road_unknown"],
        "detector_unavailable": counts["detector_unavailable"],
        "defects_unavailable": counts["defects_unavailable"],
        "invalid_image": counts["invalid_image"],
        "failures": len(images) - counts["ok"] - counts["road_unknown"],
        "per_image": outcome,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if get_config()["pipeline"]["use_stubs"]:
        print("USE_STUBS is on: set USE_STUBS=0 to test the real models", file=sys.stderr)
        return 2
    images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"no images found in {args.folder}", file=sys.stderr)
        return 2
    summary = run_smoke(Pipeline(), images, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))
    return 0 if summary["failures"] == 0 and summary["road_unknown"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
