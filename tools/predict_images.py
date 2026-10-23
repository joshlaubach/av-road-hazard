"""Score a checkpoint on one part of one fold and save a predictions artifact (one row per image).

Usage: python -m tools.predict_images --config CFG.yaml --checkpoint best.pt
         --sample-index INDEX.json --splits SPLITS.json --fold K --part {train,val,heldout}
         --out PREDICTIONS.npz
The final test set cannot be chosen: it exists only to be scored once, at the end. The artifact
holds ids, labels, probabilities and run metadata (checkpoint hash, git state, threshold) and is
read by tools/country_report.py. A tile checkpoint is scored tile by tile and an image's score is
its highest tile probability (defects/tile_predict.py); the meta records which. Needs the ml extra
(torch, torchvision, timm).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from defects.dataset import Sample, file_sha256
from defects.evaluation import Predictions, write_predictions
from defects.provenance import _git_dirty, _git_sha
from defects.splits import read_splits
from defects.tile_predict import predict_tiled
from defects.train import load_checkpoint, predict, select_device

PARTS = {"train": 0, "val": 1, "heldout": 2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--part", choices=sorted(PARTS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    splits = read_splits(args.splits)
    if not 0 <= args.fold < len(splits.folds):
        parser.error(f"fold must be in [0, {len(splits.folds)})")
    ids = splits.folds[args.fold][PARTS[args.part]]
    rows = {r["id"]: r for r in json.loads(args.sample_index.read_text())}
    samples = [
        Sample(
            i,
            (args.sample_index.parent / rows[i]["image_path"]).resolve(),
            tuple(rows[i]["labels"]),
        )
        for i in ids
    ]
    device = select_device()
    model = load_checkpoint(args.checkpoint, cfg, device=device)
    sidecar = json.loads(args.checkpoint.with_suffix(".json").read_text())
    if sidecar.get("input") == "tiles":
        # The tile geometry the model was trained with is authoritative, not the config.
        tiling = sidecar["tile_params"]
        probs = np.asarray(
            predict_tiled(
                model,
                samples,
                cfg,
                tile_size=int(tiling["tile_size"]),
                overlap=float(tiling["overlap"]),
                device=device,
            )
        )
        how = {
            "aggregation": "max-over-tiles",
            "tile_size": tiling["tile_size"],
            "tile_overlap": tiling["overlap"],
        }
    else:
        probs = np.asarray(predict(model, samples, cfg, device=device))
        how = {"aggregation": "whole-image"}
    meta = {
        "model": cfg["model_name"],
        "unit": "image",
        **how,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "sample_index_sha256": file_sha256(args.sample_index),
        "splits_sha256": file_sha256(args.splits),
        "fold": args.fold,
        "part": args.part,
        "threshold": cfg["threshold"],
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    labels = np.asarray([s.labels for s in samples])
    write_predictions(Predictions(ids=list(ids), labels=labels, probs=probs, meta=meta), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
