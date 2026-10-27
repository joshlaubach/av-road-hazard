"""Build eval/results.json from the evaluation reports and refresh the README results table.

Usage: python -m tools.build_results --classifier R.json --non-road R.json --detector-gap R.json
           --fine-tune R.json --fusion R.json [--real-photo-check R.json] [--chosen NAME]
           [--out eval/results.json] [--readme README.md]
Every input is a report written by one of the evaluation tools (tools/eval_classifier.py,
tools/eval_nonroad.py, tools/eval_detector.py, tools/eval_finetune.py,
tools/eval_fusion_synthetic.py). Only counts, rates, ranges and hashes go into the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.provenance import provenance
from eval.results import (
    build_results,
    keep_measured_retrieval,
    render_readme_table,
    retrieval_section,
    update_readme,
    validate_results,
)

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("classifier", "non-road", "detector-gap", "fine-tune", "fusion"):
        parser.add_argument(f"--{name}", type=Path, default=None)
    parser.add_argument("--retrieval", type=Path, default=None)
    parser.add_argument(
        "--add-retrieval",
        action="store_true",
        help="only replace the retrieval section of an existing results file",
    )
    parser.add_argument("--real-photo-check", type=Path, default=None)
    parser.add_argument("--chosen", default="hn2-repeat10")
    parser.add_argument("--out", type=Path, default=REPO / "eval" / "results.json")
    parser.add_argument("--readme", type=Path, default=REPO / "README.md")
    args = parser.parse_args(argv)

    if args.add_retrieval:
        if args.retrieval is None:
            parser.error("--add-retrieval needs --retrieval")
        results = _load(args.out)
        results["retrieval"] = retrieval_section(_load(args.retrieval))
        return _write(results, args)
    needed = ("classifier", "non_road", "detector_gap", "fine_tune", "fusion")
    if any(getattr(args, n) is None for n in needed):
        parser.error(
            "the full build needs --classifier --non-road --detector-gap --fine-tune --fusion"
        )
    here = provenance({}, thresholds={}, repo=REPO)
    results = build_results(
        classifier=_load(args.classifier),
        non_road=_load(args.non_road),
        detector_gap=_load(args.detector_gap),
        fine_tune=_load(args.fine_tune),
        fusion=_load(args.fusion),
        chosen=args.chosen,
        date_utc=here["date_utc"],
        git_commit=here["git_commit"],
        real_photo_check=_load(args.real_photo_check) if args.real_photo_check else None,
    )
    if args.retrieval is not None:
        results["retrieval"] = retrieval_section(_load(args.retrieval))
    elif args.out.exists():
        results = keep_measured_retrieval(results, _load(args.out))
    return _write(results, args)


def _write(results: dict[str, Any], args: argparse.Namespace) -> int:
    validate_results(results)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    readme = args.readme.read_text(encoding="utf-8")
    args.readme.write_text(update_readme(readme, render_readme_table(results)), encoding="utf-8")
    print(f"wrote {args.out} and the results table in {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
