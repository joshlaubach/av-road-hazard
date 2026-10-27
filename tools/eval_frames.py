"""Frame-level helpers for tools/eval_classifier.py: nuScenes frames, cached filters, alerts.

The road mask and detected objects do not depend on the classifier, so they are computed once per
frame (FrameContext) and reused for every checkpoint; only the classifier changes between runs.
Alerts come from the shipped defects.infer.find_tile_defects (one alert per flagged tile, before
neighbors are merged, so they match the tiles the reviewers saw). The nuScenes images stay on this
machine.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from defects.infer import find_tile_defects
from defects.patches import Tile
from defects.road_region import road_coverage
from eval.classifier import Alert, Box
from eval.detector import is_night
from fusion.schema import Object

Finder = Callable[..., list[Any]]
EXPORT_SEED = 11


def pack_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """A bool mask as packed bits plus its shape (an eighth of the memory)."""
    return np.packbits(mask), (int(mask.shape[0]), int(mask.shape[1]))


@dataclass(frozen=True)
class FrameContext:
    """One frame's classifier-independent inputs: objects and the road mask (packed)."""

    path: str  # relative to the nuScenes root
    night: bool
    objects: tuple[Object, ...]
    mask_bits: np.ndarray
    shape: tuple[int, int]

    def mask(self) -> np.ndarray:
        height, width = self.shape
        return (
            np.unpackbits(self.mask_bits, count=height * width).astype(bool).reshape(height, width)
        )


def load_frames(root: Path, version: str) -> list[tuple[str, bool]]:
    """(CAM_FRONT image path relative to root, whether its scene is a night scene)."""
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    scenes = {s["token"]: is_night(s["description"]) for s in nusc.scene}
    return [
        (nusc.get("sample_data", s["data"]["CAM_FRONT"])["filename"], scenes[s["scene_token"]])
        for s in nusc.sample
    ]


def prepare_frames(
    frames: Sequence[tuple[str, bool]],
    root: Path,
    segmenter: Callable[[np.ndarray], np.ndarray],
    detector: Callable[[np.ndarray], list[Object]],
) -> list[FrameContext]:
    """Run the road model and the object detector once per frame."""
    contexts = []
    for rel, night in frames:
        with Image.open(root / rel) as image:
            frame = np.asarray(image.convert("RGB"))
        contexts.append(
            FrameContext(rel, night, tuple(detector(frame)), *pack_mask(segmenter(frame)))
        )
    return contexts


def collect_alerts(
    contexts: Sequence[FrameContext],
    root: Path,
    *,
    finder: Finder = find_tile_defects,
    **settings: Any,
) -> list[Alert]:
    """Every per-tile alert on the frames (unmerged); classifier= and settings= pass through."""
    alerts: list[Alert] = []
    for context in contexts:
        mask = context.mask()
        with Image.open(root / context.path) as image:
            frame = np.asarray(image.convert("RGB"))
        for defect in finder(frame, objects=list(context.objects), road_mask=mask, **settings):
            box: Box = tuple(round(v) for v in defect.box)
            cover = road_coverage(Tile(defect.patch_row, defect.patch_col, box), mask)
            alerts.append(
                Alert(context.path, box, defect.name, defect.confidence, context.night, cover)
            )
    return alerts


def export_unreviewed(
    alerts: Sequence[Alert],
    root: Path,
    out_dir: Path,
    key_dir: Path,
    reviewed: Mapping[tuple[str, Box], str],
    *,
    prefix: str,
) -> int:
    """Write the alerts nobody has reviewed as an annotation batch; returns the tile count.

    One tile per (frame, box) in a seeded shuffled order: PNGs and a blank answer sheet in out_dir
    (annotator-facing), the answer key in key_dir. Both stay under the gitignored data/ folder.
    """
    tiles: dict[tuple[str, Box], dict[str, float]] = {}
    for alert in alerts:
        if (alert.frame, alert.box) not in reviewed:
            tiles.setdefault((alert.frame, alert.box), {})[alert.name] = round(alert.confidence, 4)
    order = sorted(tiles)
    random.Random(EXPORT_SEED).shuffle(order)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    key: dict[str, Any] = {}
    for n, (frame, box) in enumerate(order, 1):
        tile_id = f"{prefix}_{n:03d}"
        with Image.open(root / frame) as image:
            image.convert("RGB").crop(box).save(out_dir / f"{tile_id}.png")
        key[tile_id] = {"frame": frame, "box": list(box), "classes": tiles[(frame, box)]}
    (key_dir / "key.json").write_text(json.dumps(key, indent=1))
    with (out_dir / "annotations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile_id", "call", "certainty", "location", "type", "notes"])
        for tile_id in key:
            writer.writerow([tile_id, "", "", "", "", ""])
    return len(key)
