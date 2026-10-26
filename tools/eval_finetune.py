"""Score the fine-tuned detector against the zero-shot one, scene by scene (ADR 0015).

Usage: python -m tools.eval_finetune --nuscenes-root data/nuscenes/raw --run-dir RUNS/detector-ft-v1
           --out report.json [--timing-frames 60]
Each scene is scored only with the model that left it out (RUN_DIR/<scene>/model.safetensors, as
trained by tools/finetune_detector.py --folds all), and the pinned COCO detector is run on the same
frames. The report follows ADR 0015: new-class precision and recall per scene and pooled, with a
cutoff per scene chosen from the other scenes; old-class regression against the zero-shot
detector; CPU speed. It is built twice, for all six cameras (the verdict) and for the front camera
alone (comparable with the earlier numbers). The images stay on this machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.config import get_config
from common.hashing import file_sha256
from eval.detector import (
    COLLECT_THRESHOLD,
    FrameData,
    latency_summary,
    restrict_to_vocabulary,
    speed_verdict,
)
from eval.finetune import build_report
from eval.models import Detection
from eval.provenance import provenance
from perception.detector import DETECTOR_SUBDIR, DETECTORS, build_model, load_detector_model
from perception.finetune import FINETUNE_CLASSES, class_name, finetune_label, swap_head
from perception.labels import coco_label
from tools.eval_detector import ladder_rungs, thread_limit
from tools.finetune_detector import DEFAULT_CONFIG, SceneFrame, load_config, load_scene_frames

FRONT = "CAM_FRONT"
ADR_CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)  # fmt: skip
OLD_CLASSES = FINETUNE_CLASSES[:6]


def check_fold_meta(
    meta: dict[str, Any], scene: str, scenes: list[str], cfg: dict[str, Any]
) -> None:
    """Refuse a record that does not describe a full run that left exactly this scene out."""
    if meta.get("held_out_scene") != scene:
        raise ValueError(f"model for {scene} was held out of {meta.get('held_out_scene')!r}")
    if sorted(meta.get("train_scenes", [])) != sorted(s for s in scenes if s != scene):
        raise ValueError(f"model for {scene} was not trained on exactly the other scenes")
    if meta.get("smoke") is not False:
        raise ValueError(f"model for {scene} is not marked as a full (non-smoke) run")
    if meta.get("epochs") != cfg["train"]["epochs"]:
        raise ValueError(f"model for {scene} used {meta.get('epochs')} epochs, not the config's")
    if meta.get("classes") != list(FINETUNE_CLASSES):
        raise ValueError(f"model for {scene} has different classes")
    saved = meta.get("config", {})
    if saved.get("model") != cfg["model"]:
        raise ValueError(f"model for {scene} is not the configured model")
    if saved.get("train") != cfg["train"]:
        raise ValueError(f"model for {scene} was trained with other settings than the config")
    if saved.get("cameras") != cfg["cameras"]:
        raise ValueError(f"model for {scene} was trained on other cameras than the config")


def load_finetuned(
    folder: Path,
    device: str,
    *,
    scene: str | None = None,
    scenes: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> Any:
    """A fine-tuned fold model, after checking its file against the hash in its meta.json.

    With scene, scenes and cfg given, the record must also say the model left that scene out.
    """
    from safetensors.torch import load_file

    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if scene is not None and scenes is not None and cfg is not None:
        check_fold_meta(meta, scene, scenes, cfg)
    weights = folder / "model.safetensors"
    if file_sha256(weights) != meta["sha256"]:
        raise ValueError(f"{weights} does not match the hash in its meta.json")
    model = swap_head(build_model(meta["config"]["model"]))
    model.load_state_dict(load_file(str(weights)))
    return model.eval().to(device)


def predict(model: Any, frame: np.ndarray, device: str, *, finetuned: bool) -> list[Detection]:
    """Every detection above the collection threshold, labeled with fusion labels."""
    import torch

    image = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
    with torch.inference_mode():
        (output,) = model([image.to(device)])
    detections = []
    for box, category, score in zip(
        output["boxes"].cpu().tolist(),
        output["labels"].cpu().tolist(),
        output["scores"].cpu().tolist(),
        strict=True,
    ):
        label = class_name(int(category)) if finetuned else coco_label(int(category))
        if label is not None and score >= COLLECT_THRESHOLD:
            detections.append(Detection(label, float(score), (box[0], box[1], box[2], box[3])))
    return detections


def _rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def group_by_scene(
    frames: list[SceneFrame], detections: list[list[Detection]], *, camera: str | None = None
) -> dict[str, list[FrameData]]:
    """FrameData per scene (optionally one camera only), ground truth in the fine-tune labels."""
    grouped: dict[str, list[FrameData]] = {}
    for frame, found in zip(frames, detections, strict=True):
        if camera is not None and frame.camera != camera:
            continue
        truth, _ = restrict_to_vocabulary(frame.ground_truth, finetune_label)
        grouped.setdefault(frame.scene, []).append(
            FrameData(frame.scene, frame.night, found, truth)
        )
    return grouped


def time_on_cpu(model: Any, paths: list[Path], threads: int) -> dict[str, Any]:
    """Latency of one image at a time on the CPU with the given thread count."""
    import torch

    with thread_limit(threads):
        seconds = []
        for i, path in enumerate(paths):
            frame = _rgb(path)
            start = time.perf_counter()
            predict(model, frame, "cpu", finetuned=True)
            if i >= 3:  # warm-up
                seconds.append(time.perf_counter() - start)
        return {
            **latency_summary(seconds),
            "threads": torch.get_num_threads(),
            "cpu": platform.processor(),
        }


def check_cameras(cameras: list[str]) -> None:
    """The ADR 0015 verdict is for all six cameras; a config naming fewer is refused."""
    if sorted(cameras) != sorted(ADR_CAMERAS):
        raise ValueError(f"ADR 0015 needs the six cameras {list(ADR_CAMERAS)}, got {cameras}")


def exit_status(report: dict[str, Any]) -> int:
    """0 only if the six-camera detection checks pass and CPU speed was measured and passes."""
    ladder = report.get("cpu_latency_ladder", [])
    speeds_pass = bool(ladder) and all(r["passed"] for r in ladder)
    return 0 if report["all_cameras"]["passed"] and speeds_pass else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nuscenes-root", type=Path, default=Path("data/nuscenes/raw"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing-frames", type=int, default=60)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    from common.device import select_device

    cfg = load_config(args.config)
    check_cameras(cfg["cameras"])
    device = select_device()
    frames = load_scene_frames(args.nuscenes_root, args.version, cfg["cameras"])
    scenes = sorted({f.scene for f in frames})
    missing = [s for s in scenes if not (args.run_dir / s / "model.safetensors").is_file()]
    if missing:
        print(f"no trained model for {missing} in {args.run_dir}", file=sys.stderr)
        return 2

    checkpoint_dir = Path(get_config()["paths"]["checkpoint_dir"])
    spec = DETECTORS[cfg["model"]]
    zero_weights = checkpoint_dir / DETECTOR_SUBDIR / spec.filename
    zero_shot = load_detector_model(zero_weights, spec, device=device)
    tuned_detections: dict[int, list[Detection]] = {}
    zero_detections: list[list[Detection]] = []
    for scene in scenes:
        model = load_finetuned(args.run_dir / scene, device, scene=scene, scenes=scenes, cfg=cfg)
        for index, frame in enumerate(frames):
            if frame.scene == scene:
                tuned_detections[index] = predict(model, _rgb(frame.path), device, finetuned=True)
        print(f"scored {scene}", file=sys.stderr)
        del model
    for frame in frames:
        zero_detections.append(predict(zero_shot, _rgb(frame.path), device, finetuned=False))
    tuned = [tuned_detections[i] for i in range(len(frames))]

    report: dict[str, Any] = {}
    for name, camera in (("all_cameras", None), ("front_camera", FRONT)):
        report[name] = build_report(
            group_by_scene(frames, tuned, camera=camera),
            group_by_scene(frames, zero_detections, camera=camera),
            cfg["evaluation"],
            OLD_CLASSES,
        )
    if args.timing_frames > 0:
        cpu_model = load_finetuned(
            args.run_dir / scenes[0], "cpu", scene=scenes[0], scenes=scenes, cfg=cfg
        )
        front = [f.path for f in frames if f.camera == FRONT]
        step = max(1, len(front) // args.timing_frames)
        paths = front[::step][: args.timing_frames]
        import os

        rungs = [
            time_on_cpu(cpu_model, paths, t) for t in ladder_rungs([os.cpu_count() or 4, 4, 2])
        ]
        report["cpu_latency_ladder"] = speed_verdict(rungs)
    report["passed_all_checks_but_speed_and_sanity"] = report["all_cameras"]["passed"]
    report["provenance"] = provenance(
        {
            "zero_shot": zero_weights,
            "config": args.config,
            **{f"fold/{s}": args.run_dir / s / "model.safetensors" for s in scenes},
        },
        thresholds={"old_classes": cfg["evaluation"]["old_classes_cutoff"]},
        repo=Path(__file__).resolve().parents[1],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    verdicts = {
        view: {
            "passed": report[view]["passed"],
            "new": {k: v["verdict"]["passed"] for k, v in report[view]["new_classes"].items()},
            "old_failed": [k for k, v in report[view]["old_classes"].items() if not v["passed"]],
        }
        for view in ("all_cameras", "front_camera")
    }
    print(json.dumps(verdicts, indent=2))
    print(
        "detection checks and measured speeds only; the sanity check outside nuScenes is manual",
        file=sys.stderr,
    )
    return exit_status(report)


if __name__ == "__main__":
    sys.exit(main())
