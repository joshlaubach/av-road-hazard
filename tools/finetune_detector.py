"""Fine-tune the COCO detector on nuScenes for traffic cones and barriers (ADR 0015).

Usage: python -m tools.finetune_detector --nuscenes-root data/nuscenes/raw --out-dir DIR
           --run-id NAME [--folds all | --fold SCENE | --all-scenes] [--epochs N] [--smoke]
DIR must be outside the repository. Starts from the pinned COCO weights, swaps the box head for the
8-class one (the six COCO rows are kept), and trains on all six cameras. --folds all trains one
model per scene with that scene left out (the leave-one-scene-out evaluation); --all-scenes trains
the final model on every scene. --smoke trains briefly on a few images to check the loop runs.
Each model is saved to DIR/RUN_ID/<fold>/ as model.safetensors with a meta.json beside it. The
nuScenes images and these weights stay on this machine: the weights carry the nuScenes terms.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.config import get_config
from eval.detector import GtBox, is_night
from perception.detector import DETECTOR_SUBDIR, DETECTORS, load_detector_model
from perception.finetune import (
    FINETUNE_CLASSES,
    TrainItem,
    flip_boxes,
    freeze_batchnorm_statistics,
    scene_folds,
    swap_head,
    training_item,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "detector_finetune.yaml"
REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SceneFrame:
    """One nuScenes image with its scene, camera and ground truth."""

    scene: str
    night: bool
    camera: str
    path: Path
    ground_truth: list[GtBox]


def load_scene_frames(root: Path, version: str, cameras: list[str]) -> list[SceneFrame]:
    """Every image of the chosen cameras with its projected ground truth (needs the devkit)."""
    from nuscenes.nuscenes import NuScenes

    from tools.eval_detector import _project

    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    scenes = {s["token"]: s for s in nusc.scene}
    frames: list[SceneFrame] = []
    for sample in nusc.sample:
        scene = scenes[sample["scene_token"]]
        for camera in cameras:
            path, boxes = _project(nusc, sample, camera)
            frames.append(
                SceneFrame(scene["name"], is_night(scene["description"]), camera, path, boxes)
            )
    return frames


def items_for(frames: list[SceneFrame]) -> list[TrainItem]:
    """Training items (boxes and classes) for the frames."""
    return [
        training_item(str(f.path), [(g.label, g.box, g.visibility) for g in f.ground_truth])
        for f in frames
    ]


RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def check_run_id(run_id: str) -> str:
    """The run id, if it is one plain folder name (no separators, no '..'): it names a folder."""
    if not RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError(f"run id must be a plain folder name, got {run_id!r}")
    return run_id


def check_outside_repo(path: Path) -> Path:
    """The resolved output folder; refuses one inside the repository (weights never go in git)."""
    resolved = path.resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise ValueError(f"--out-dir must be outside the repository, got {resolved}")
    return resolved


# ------------------------------------------------------------------------------- data


class DetectionDataset:
    """Images with their boxes, loaded on demand, with the flip and brightness/contrast jitter."""

    def __init__(self, items: list[TrainItem], flip: float, jitter: float) -> None:
        self.items = items
        self.flip = flip
        self.jitter = jitter

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch
        from PIL import Image

        item = self.items[index]
        with Image.open(item.path) as image:
            array = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
        boxes = [list(b) for b in item.boxes]
        if random.random() < self.flip:
            tensor = tensor.flip(-1)
            boxes = [list(b) for b in flip_boxes(item.boxes, float(tensor.shape[-1]))]
        brightness = 1 + random.uniform(-self.jitter, self.jitter)
        contrast = 1 + random.uniform(-self.jitter, self.jitter)
        mean = tensor.mean()
        tensor = ((tensor * brightness - mean) * contrast + mean).clamp(0.0, 1.0)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(item.labels, dtype=torch.int64),
        }
        return tensor, target


def _collate(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict[str, Any]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


# ----------------------------------------------------------------------------- training


def learning_rate_factor(step: int, warmup: int, total: int) -> float:
    """1.0 is the base rate: a straight rise over the warm-up steps, then a cosine down to 0."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train(
    model: Any,
    items: list[TrainItem],
    cfg: dict[str, Any],
    *,
    device: str,
    epochs: int,
    progress: bool = True,
) -> list[float]:
    """Train the model on the items; returns the mean total loss of each epoch."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    data: Any = DetectionDataset(items, cfg["flip"], cfg["jitter"])
    loader = DataLoader(
        data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=_collate,
        drop_last=len(data) > cfg["batch_size"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"]
    )
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, cfg["warmup_iters"], total_steps)
    )
    amp = cfg.get("amp", "none") == "bf16" and device.startswith("cuda")
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        freeze_batchnorm_statistics(model)
        running = 0.0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", disable=not progress, leave=False)
        for images, targets in bar:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(
                device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp
            ):
                loss = sum(model(images, targets).values())
            if not math.isfinite(float(loss)):
                raise RuntimeError(f"loss is {float(loss)}: training diverged")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            running += float(loss)
            bar.set_postfix(loss=f"{float(loss):.3f}")
        losses.append(running / max(1, len(loader)))
    return losses


def build_start_model(device: str, model_name: str) -> Any:
    """The pinned COCO detector named by the run's config, with the 8-class head, ready to train."""
    cfg = get_config()
    spec = DETECTORS[model_name]
    path = Path(cfg["paths"]["checkpoint_dir"]) / DETECTOR_SUBDIR / spec.filename
    return swap_head(load_detector_model(path, spec, device=device), seed=0)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def save_model(model: Any, folder: Path, meta: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    from common.hashing import file_sha256

    folder.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(folder / "model.safetensors"))
    meta = {**meta, "sha256": file_sha256(folder / "model.safetensors")}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def class_box_counts(items: list[TrainItem]) -> dict[str, int]:
    counts = dict.fromkeys(FINETUNE_CLASSES, 0)
    for item in items:
        for label in item.labels:
            counts[FINETUNE_CLASSES[label - 1]] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nuscenes-root", type=Path, default=Path("data/nuscenes/raw"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folds", choices=["all"], help="one model per held-out scene")
    group.add_argument("--fold", help="one fold: the scene to hold out, e.g. scene-0061")
    group.add_argument("--all-scenes", action="store_true", help="the final model, all scenes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="a few images, one epoch")
    parser.add_argument("--images", type=int, default=32, help="images to use with --smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    from common.device import select_device

    cfg = load_config(args.config)
    out_root = check_outside_repo(check_outside_repo(args.out_dir) / check_run_id(args.run_id))
    device = select_device()
    frames = load_scene_frames(args.nuscenes_root, args.version, cfg["cameras"])
    scenes = sorted({f.scene for f in frames})
    jobs: list[tuple[str, list[str], str | None]]
    if args.all_scenes:
        jobs = [("all-scenes", scenes, None)]
    else:
        folds = scene_folds(scenes)
        wanted = [f for f in folds if args.folds == "all" or f[1] == args.fold]
        if not wanted:
            print(f"no scene named {args.fold!r}; scenes are {scenes}", file=sys.stderr)
            return 2
        jobs = [(held, train_scenes, held) for train_scenes, held in wanted]
    epochs = args.epochs or (1 if args.smoke else cfg["train"]["epochs"])
    for name, train_scenes, held_out in jobs:
        chosen = [f for f in frames if f.scene in set(train_scenes)]
        if args.smoke:
            chosen = random.Random(0).sample(chosen, min(args.images, len(chosen)))
        items = items_for(chosen)
        started = time.time()
        print(f"[{name}] {len(items)} images, {class_box_counts(items)}", file=sys.stderr)
        model = build_start_model(device, cfg["model"])
        losses = train(model, items, cfg["train"], device=device, epochs=epochs)
        save_model(
            model,
            out_root / name,
            {
                "run_id": args.run_id,
                "held_out_scene": held_out,
                "train_scenes": sorted(train_scenes),
                "images": len(items),
                "boxes": class_box_counts(items),
                "classes": list(FINETUNE_CLASSES),
                "epochs": epochs,
                "epoch_losses": losses,
                "config": cfg,
                "smoke": args.smoke,
                "seconds": round(time.time() - started),
            },
        )
        print(f"[{name}] losses {[round(x, 3) for x in losses]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
