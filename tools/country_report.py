"""Per-country, per-class report from a predictions artifact, with a country-prior baseline.

Usage: python -m tools.country_report --predictions MODEL.npz --out REPORT.json
         [--baseline-sample-index INDEX.json --splits SPLITS.json --fold K] [--threshold T]
The artifact is written by tools/predict_images.py (whole-image models) or by tile evaluation.
With the baseline arguments, every country's training prevalence (images of fold K's training
part) is evaluated on the same ids as the model, so the two columns are directly comparable. The
threshold defaults to the one saved in the artifact, else 0.5. Prints a table and writes JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from defects.evaluation import (
    Predictions,
    build_report,
    country_prior_probs,
    read_predictions,
)
from defects.labels import CODES
from defects.splits import read_splits

DEFAULT_THRESHOLD = 0.5


def baseline_predictions(
    model: Predictions, sample_index: Path, splits_path: Path, fold: int
) -> Predictions:
    """Country-prior probabilities for the model's ids, from fold K's training-part labels."""
    train_ids = read_splits(splits_path).folds[fold][0]
    rows = {r["id"]: r for r in json.loads(sample_index.read_text())}
    labels = np.asarray([rows[i]["labels"] for i in train_ids])
    probs = country_prior_probs(train_ids, labels, model.ids)
    meta = {"model": "country-prior", "trained_on": f"fold {fold} training images"}
    return Predictions(ids=list(model.ids), labels=model.labels, probs=probs, meta=meta)


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.3f}"


def format_table(report: dict[str, Any]) -> str:
    """One block per class: country, counts, prevalence, then model/baseline AUROC, AP and F1."""
    lines = []
    for code in CODES:
        lines.append(f"\n{code}   n  prev | AUROC model/prior | AP model/prior | F1 model/prior")
        for country, by_class in report["countries"].items():
            cell = by_class[code]
            model, prior = cell["model"], cell.get("baseline", {})
            lines.append(
                f"  {country:14s}{model['n']:7d} {model['prevalence']:5.1%} | "
                f"{_fmt(model['auroc'])} / {_fmt(prior.get('auroc'))}   | "
                f"{_fmt(model['average_precision'])} / {_fmt(prior.get('average_precision'))} | "
                f"{_fmt(model['f1'])} / {_fmt(prior.get('f1'))}"
                + (f"   ({'; '.join(model['notes'])})" if model["notes"] else "")
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-sample-index", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args(argv)
    if bool(args.baseline_sample_index) != bool(args.splits):
        parser.error("--baseline-sample-index and --splits go together")
    model = read_predictions(args.predictions)
    threshold = args.threshold
    if threshold is None:
        threshold = float(model.meta.get("threshold", DEFAULT_THRESHOLD))
    baseline = None
    if args.baseline_sample_index:
        baseline = baseline_predictions(model, args.baseline_sample_index, args.splits, args.fold)
    report = build_report(model, baseline, threshold=threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
