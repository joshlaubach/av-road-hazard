"""Measure the pinned COCO detectors on nuScenes CAM_FRONT: accuracy, slices and CPU speed.

Usage: python -m tools.eval_detector --nuscenes-root data/nuscenes/raw [--version v1.0-mini]
           [--models NAME ...] [--timing-frames 100] [--threads N] [--thread-ladder N ...]
           [--out report.json]
--timing-frames 0 skips the speed test. --thread-ladder times the CPU at each thread count and
marks each rung against the 2 s p95 target (eval/detector.py).
Needs the nuscenes-devkit and the weights from tools/fetch_detector.py. Ground truth is each
annotation's 3D box projected into the image and clamped to it, with labels mapped by
perception/labels.py. Two ground-truth sets are scored: all projected boxes, and the evaluable ones
(at least 40% visible and at least 32 x 32 px; these two cut-offs are this tool's own choice, not
the source repo's). Scenes are split 80/20 by sorted token, as docs/EVALUATION.md specifies; the
score threshold is chosen on the training scenes only (maximum F1 on evaluable ground truth) and
the headline numbers are those at that threshold on the held-out scenes. Accuracy runs on the
default device; speed runs on the CPU (--threads limits its threads, to imitate a smaller host),
one frame at a time after warm-up, and reports the median and 95th percentile. The report also
carries a provenance block (weight hashes, git commit, date; eval/provenance.py). The nuScenes
images never leave this machine and nothing is written but the report. Matching, F1 and AP
are eval/models.py (same label, IoU >= 0.5, greedy by score).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from common.config import get_config
from eval.detector import (
    COLLECT_THRESHOLD,
    FrameData,
    GtBox,
    is_night,
    latency_summary,
    score_model,
    speed_verdict,
    split_scenes,
)
from eval.models import Detection
from eval.provenance import provenance
from perception.detector import DETECTOR_SUBDIR, DETECTORS, detect_objects, load_detector_model
from perception.infer import detector_settings
from perception.labels import FUSION_LABELS, nuscenes_label

# (scene token, is night, image path, ground truth)
LoadedFrame = tuple[str, bool, Path, list[GtBox]]


# ------------------------------------------------------------------- nuScenes


def _project(nusc: Any, sample: dict[str, Any]) -> tuple[Path, list[GtBox]]:
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    from PIL import Image

    token = sample["data"]["CAM_FRONT"]
    path, boxes, intrinsic = nusc.get_sample_data(token, box_vis_level=BoxVisibility.ANY)
    with Image.open(path) as image:
        width, height = image.size
    out: list[GtBox] = []
    for box in boxes:
        label = nuscenes_label(box.name)
        if label is None:
            continue
        pts = view_points(box.corners(), np.array(intrinsic), normalize=True)[:2]
        x1, y1 = max(0.0, float(pts[0].min())), max(0.0, float(pts[1].min()))
        x2, y2 = min(float(width), float(pts[0].max())), min(float(height), float(pts[1].max()))
        if x2 > x1 and y2 > y1:
            visibility = int(nusc.get("sample_annotation", box.token)["visibility_token"])
            out.append(GtBox(label, (x1, y1, x2, y2), visibility))
    return Path(path), out


def _load_frames(nusc: Any) -> list[LoadedFrame]:
    scenes = {s["token"]: s for s in nusc.scene}
    frames: list[LoadedFrame] = []
    for sample in nusc.sample:
        scene = scenes[sample["scene_token"]]
        path, boxes = _project(nusc, sample)
        frames.append((scene["token"], is_night(scene["description"]), path, boxes))
    return frames


def _rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def run_model(
    name: str,
    frames: list[LoadedFrame],
    checkpoint_dir: Path,
    timing_frames: int,
    threads: int | None = None,
    ladder: list[int] | None = None,
) -> dict[str, Any]:
    spec = DETECTORS[name]
    weights = checkpoint_dir / DETECTOR_SUBDIR / spec.filename
    model = load_detector_model(weights, spec)
    data = [
        FrameData(
            scene,
            night,
            [
                Detection(o.label, o.score, o.box)
                for o in detect_objects(model, _rgb(path), score_threshold=COLLECT_THRESHOLD)
            ],
            boxes,
        )
        for scene, night, path, boxes in frames
    ]
    train, _ = split_scenes(sorted({f[0] for f in frames}))
    result = score_model(data, set(train), FUSION_LABELS)
    if timing_frames > 0:
        if ladder:
            rungs = [
                _time_on_cpu(name, weights, frames, timing_frames, t) for t in ladder_rungs(ladder)
            ]
            result["cpu_latency_ladder"] = speed_verdict(rungs)
        else:
            result["cpu_latency"] = _time_on_cpu(name, weights, frames, timing_frames, threads)
    return result


@contextlib.contextmanager
def thread_limit(count: int | None) -> Iterator[None]:
    """Run with `count` torch threads (None: leave as is), restoring the previous count after."""
    import torch

    before = torch.get_num_threads()
    if count is not None:
        torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def ladder_rungs(counts: list[int]) -> list[int]:
    """The thread counts to time: unique, positive, highest first."""
    return sorted({c for c in counts if c > 0}, reverse=True)


def _time_on_cpu(
    name: str, weights: Path, frames: list[LoadedFrame], count: int, threads: int | None
) -> dict[str, Any]:
    import torch

    with thread_limit(threads):
        model = load_detector_model(weights, DETECTORS[name], device="cpu")
        step = max(1, len(frames) // count)
        paths = [f[2] for f in frames[::step]][:count]
        seconds: list[float] = []
        for i, path in enumerate(paths):
            frame = _rgb(path)
            start = time.perf_counter()
            detect_objects(model, frame, score_threshold=0.5, device="cpu")
            if i >= 3:  # the first few calls are warm-up
                seconds.append(time.perf_counter() - start)
        return {
            **latency_summary(seconds),
            "threads": torch.get_num_threads(),
            "cpu": platform.processor(),
        }


def _provenance(checkpoint_dir: Path, models: list[str]) -> dict[str, Any]:
    files = {m: checkpoint_dir / DETECTOR_SUBDIR / DETECTORS[m].filename for m in models}
    return provenance(
        files,
        thresholds={"score_threshold": detector_settings(get_config())[1]},
        repo=Path(__file__).resolve().parents[1],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--models", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    parser.add_argument("--timing-frames", type=int, default=100)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--thread-ladder", type=int, nargs="+", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from nuscenes.nuscenes import NuScenes

    checkpoint_dir = args.checkpoint_dir or Path(get_config()["paths"]["checkpoint_dir"])
    nusc = NuScenes(version=args.version, dataroot=str(args.nuscenes_root), verbose=False)
    frames = _load_frames(nusc)
    train, held_out = split_scenes(sorted({f[0] for f in frames}))
    report: dict[str, Any] = {
        "version": args.version,
        "frames": len(frames),
        "train_scenes": train,
        "held_out_scenes": held_out,
        "models": {
            name: run_model(
                name, frames, checkpoint_dir, args.timing_frames, args.threads, args.thread_ladder
            )
            for name in args.models
        },
        "provenance": _provenance(checkpoint_dir, args.models),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
