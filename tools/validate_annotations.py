"""Export structurally validated training samples and a JSON exclusion report. No ML runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from defects.dataset import build_samples
from defects.voc import find_image


def _image_for(xml: Path) -> Path:
    image = find_image(xml)
    if image is None:
        raise FileNotFoundError(f"No image for {xml}")
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error("root must be an existing dataset directory")
    if args.report.resolve() == args.sample_index.resolve():
        parser.error("report and sample-index must be different files")
    for output in (args.report, args.sample_index):
        if output.suffix.lower() != ".json":
            parser.error("output files must have a .json extension")
    samples, report = build_samples(args.root, _image_for)
    for output in (args.report, args.sample_index):
        output.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(args.report)
    rows = [
        {
            "id": s.sample_id,
            "image_path": Path(os.path.relpath(s.image_path, args.sample_index.parent)).as_posix(),
            "labels": list(s.labels),
        }
        for s in samples
    ]
    args.sample_index.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
