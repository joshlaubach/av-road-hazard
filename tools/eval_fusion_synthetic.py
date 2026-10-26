"""Synthetic fusion evaluation: paste RDD2022 defects onto nuScenes road and score the pipeline.

Usage: USE_STUBS=0 DEFECT_CHECKPOINT=PATH python -m tools.eval_fusion_synthetic
           --outlines data/nuscenes/derived/road_outlines.json --out report.json
           [--nuscenes-root data/nuscenes/raw] [--rdd-root data/rdd2022/raw]
           [--splits data/rdd2022/derived/splits.json] [--config configs/eval_fusion.yaml]
           [--limit-frames N]
Needs the real models (detector, road model, classifier checkpoint) and the nuscenes-devkit. For
each outlined frame it pastes patches at known places on and off the hand-drawn road, runs the real
pipeline on each pasted frame, and scores detection, in_road and near_objects (the scoring and
its limits are explained in eval/fusion_synthetic.py). Patches come only from the held-out test
split of RDD2022, so the classifier has never trained on them. Frames without a patch are also
run, to report how many defects the pipeline raises on its own. The images stay on this machine;
the report holds only numbers.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from api.pipeline import Pipeline
from common.config import get_config
from defects.voc import EXPECTED_CODES, find_image, parse_annotation
from eval.detector import GtBox, evaluable
from eval.fusion_synthetic import (
    Box,
    NearFn,
    Reported,
    Trial,
    paste_patch,
    polygon_mask,
    sample_box,
    score_trial,
    summarize,
    verdict,
)
from eval.provenance import provenance
from fusion.config import FusionConfig, load_fusion_config
from fusion.fuse import normalized_distance
from fusion.schema import HazardRecord
from perception.labels import COCO_VOCABULARY, to_vocabulary

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "eval_fusion.yaml"
Analyze = Callable[[np.ndarray], list[Reported]]


# ------------------------------------------------------------------------------ patches


@dataclass(frozen=True)
class PatchCrop:
    """A defect crop with some road around it, and where the defect sits inside the crop."""

    image: np.ndarray
    inner: Box


def _looks_like_road(
    shape: tuple[int, ...], defect: tuple[int, int, int, int], cfg: dict[str, Any]
) -> bool:
    """Whether a defect box is a plausible road patch: low in its photo, not huge, not a sliver."""
    height, width = shape[:2]
    x1, y1, x2, y2 = defect
    box_w, box_h = x2 - x1, y2 - y1
    low, high = cfg["aspect_range"]
    return bool(
        (y1 + y2) / 2 >= cfg["min_center_y_frac"] * height
        and box_w * box_h <= cfg["max_area_frac"] * width * height
        and low <= box_w / box_h <= high
    )


def build_patch_bank(
    raw_root: Path,
    ids: Sequence[str],
    cfg: dict[str, Any],
    count: int,
    rng: random.Random,
) -> list[PatchCrop]:
    """Up to count defect crops from the given annotation ids (RDD2022 test split only)."""
    countries = set(cfg["countries"])
    pool = [i for i in ids if i.split("/")[0] in countries]
    rng.shuffle(pool)
    bank: list[PatchCrop] = []
    for item in pool:
        if len(bank) >= count:
            break
        xml = raw_root / f"{item}.xml"
        image_path = find_image(xml) if xml.is_file() else None
        if image_path is None:
            continue
        _, boxes = parse_annotation(xml)
        usable = [
            b
            for b in boxes
            if b[0] in EXPECTED_CODES and min(b[3] - b[1], b[4] - b[2]) >= cfg["min_defect_side_px"]
        ]
        if not usable:
            continue
        _, x1, y1, x2, y2 = rng.choice(usable)
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        if not _looks_like_road(rgb.shape, (x1, y1, x2, y2), cfg):
            continue
        pad_x, pad_y = (x2 - x1) * cfg["context"], (y2 - y1) * cfg["context"]
        cx1, cy1 = max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y))
        cx2, cy2 = (
            min(rgb.shape[1], math.ceil(x2 + pad_x)),
            min(rgb.shape[0], math.ceil(y2 + pad_y)),
        )
        bank.append(
            PatchCrop(rgb[cy1:cy2, cx1:cx2].copy(), (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1))
        )
    return bank


# ----------------------------------------------------------------------------- scoring


def make_expected_near(ground_truth: Sequence[GtBox], cfg: FusionConfig) -> NearFn:
    """Labels of ground-truth objects that fusion's own rule would list as near a defect box.

    Only objects a COCO detector could fairly be expected to name (visible enough, big enough,
    in its vocabulary) count, so the score reflects fusion and the detector, not the label set.
    """
    objects = [
        (label, g.box)
        for g in ground_truth
        if evaluable(g) and (label := to_vocabulary(g.label, COCO_VOCABULARY)) is not None
    ]

    def near(defect_box: Box) -> set[str]:
        return {
            label
            for label, box in objects
            if normalized_distance(box, defect_box, cfg) <= cfg.near_radius
        }

    return near


def reported_from_record(record: HazardRecord) -> list[Reported]:
    """The defects of a pipeline record, in the form the scorer takes."""
    return [
        Reported(d.box, d.confidence, d.in_road, tuple(n.label for n in d.near_objects))
        for d in record.defects
    ]


def _png(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def pipeline_analyzer(pipeline: Pipeline) -> Analyze:
    """The real pipeline as a frame -> reported-defects function (frames travel as lossless PNG)."""
    return lambda frame: reported_from_record(pipeline.analyze(_png(frame)))


def _pick_patch(
    bank: Sequence[PatchCrop], width: float, max_upscale: float, rng: random.Random
) -> PatchCrop:
    """A random patch that needs no more than max_upscale enlargement to reach the width.

    If none is big enough, the largest one is used and is enlarged as little as possible.
    """
    fits = [p for p in bank if p.image.shape[1] * max_upscale >= width]
    return rng.choice(fits) if fits else max(bank, key=lambda p: p.image.shape[1])


def run_frame(
    analyze: Analyze,
    frame: np.ndarray,
    road_truth: np.ndarray,
    blocked: Sequence[Box],
    bank: Sequence[PatchCrop],
    expected_near: NearFn,
    cfg: dict[str, Any],
    road_threshold: float,
    rng: random.Random,
) -> tuple[list[Trial], int]:
    """Paste patches on and off the road of one frame, one per pipeline run; (trials, skipped)."""
    height = frame.shape[0]
    patch_cfg, place = cfg["patch"], cfg["placement"]
    trials: list[Trial] = []
    skipped = 0
    for on_road, count in (
        (True, place["on_road_per_frame"]),
        (False, place["off_road_per_frame"]),
    ):
        for _ in range(count):
            width = math.exp(
                rng.uniform(
                    math.log(patch_cfg["min_width_px"]), math.log(patch_cfg["max_width_px"])
                )
            )
            patch = _pick_patch(bank, width, patch_cfg["max_upscale"], rng)
            scale = width / patch.image.shape[1]
            size = (round(width), max(1, round(patch.image.shape[0] * scale)))
            box = sample_box(
                road_truth,
                size,
                on_road=on_road,
                blocked=blocked,
                margin=place["margin_px"],
                min_y=place["min_y_frac"] * height,
                rng=rng,
            )
            if box is None:
                skipped += 1
                continue
            pasted = paste_patch(frame, patch.image, box, patch_cfg["feather"])
            sx, sy = (
                (box[2] - box[0]) / patch.image.shape[1],
                (box[3] - box[1]) / patch.image.shape[0],
            )
            truth = (
                box[0] + patch.inner[0] * sx,
                box[1] + patch.inner[1] * sy,
                box[0] + patch.inner[2] * sx,
                box[1] + patch.inner[3] * sy,
            )
            trials.append(
                score_trial(
                    truth,
                    on_road,
                    analyze(pasted),
                    road_truth=road_truth,
                    road_threshold=road_threshold,
                    expected_near=expected_near,
                )
            )
    return trials, skipped


# ------------------------------------------------------------------------------ nuScenes


def _ground_truth_by_file(root: Path, version: str, wanted: set[str]) -> dict[str, list[GtBox]]:
    from nuscenes.nuscenes import NuScenes

    from tools.eval_detector import _project

    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    out: dict[str, list[GtBox]] = {}
    for sample in nusc.sample:
        rel = nusc.get("sample_data", sample["data"]["CAM_FRONT"])["filename"]
        if rel in wanted:
            out[rel] = _project(nusc, sample)[1]
    return out


def _paste_sources(splits: Path) -> list[str]:
    return list(json.loads(splits.read_text(encoding="utf-8"))["test"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outlines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, default=Path("data/nuscenes/raw"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--rdd-root", type=Path, default=Path("data/rdd2022/raw"))
    parser.add_argument("--splits", type=Path, default=Path("data/rdd2022/derived/splits.json"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit-frames", type=int, default=None)
    args = parser.parse_args(argv)
    if get_config()["pipeline"]["use_stubs"]:
        print("USE_STUBS is on: set USE_STUBS=0 to test the real models", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fusion_cfg = load_fusion_config()
    outlines = json.loads(args.outlines.read_text(encoding="utf-8"))
    entries = outlines["frames"][: args.limit_frames]
    size = tuple(outlines["image_size"])
    truth_boxes = _ground_truth_by_file(
        args.nuscenes_root, args.version, {e["frame"] for e in entries}
    )
    rng = random.Random(cfg["seed"])
    bank = build_patch_bank(args.rdd_root, _paste_sources(args.splits), cfg["patch"], 200, rng)
    if not bank:
        print("no usable defect patches found in the RDD2022 test split", file=sys.stderr)
        return 2

    analyze = pipeline_analyzer(Pipeline())
    trials: list[Trial] = []
    skipped = clean_defects = 0
    for entry in entries:
        with Image.open(args.nuscenes_root / entry["frame"]) as image:
            frame = np.asarray(image.convert("RGB"))
        road = polygon_mask([tuple(p) for p in entry["road_polygon"]], size)
        gt = truth_boxes[entry["frame"]]
        found, missed = run_frame(
            analyze,
            frame,
            road,
            [g.box for g in gt],
            bank,
            make_expected_near(gt, fusion_cfg),
            cfg,
            fusion_cfg.road_overlap_threshold,
            rng,
        )
        trials += found
        skipped += missed
        clean_defects += len(analyze(frame))
        print(f"{entry['frame']}: {len(found)} trials", file=sys.stderr)

    bands = cfg["bands"]
    report = summarize(
        trials,
        size_edges_px=bands["size_edges_px"],
        y_edges_frac=bands["y_edges_frac"],
        image_height=size[1],
    )
    config = get_config()
    report["verdict"] = verdict(report, cfg["criteria"])
    report["criteria"] = cfg["criteria"]
    report["clean_frames"] = {
        "frames": len(entries),
        "defects": clean_defects,
        "per_frame": clean_defects / max(1, len(entries)),
    }
    report["skipped_placements"] = skipped
    report["patches_in_bank"] = len(bank)
    report["provenance"] = provenance(
        {
            "outlines": args.outlines,
            "config": args.config,
            "classifier": Path(config["defects"]["classifier_checkpoint"]),
        },
        thresholds=config["defects"]["thresholds"],
        repo=Path(__file__).resolve().parent.parent,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("verdict", "on_road", "off_road", "in_road", "near_objects")},
            indent=2,
        )
    )
    return 0 if report["verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
