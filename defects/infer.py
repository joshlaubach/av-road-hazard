"""Defect detection entry point: the tile classifier behind find_defects, or a stub.

With USE_STUBS on, find_defects returns one fixed fake defect. With stubs off it loads the tile
classifier named by defects.classifier_checkpoint (env DEFECT_CHECKPOINT), tiles the frame, skips
tiles that objects cover or that are off the road, and applies the per-class thresholds from the
config (all under `defects:` in configs/default.yaml).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from common.config import get_config
from defects.labels import CODES
from fusion.fuse import RoadMask
from fusion.schema import Defect, Object


class Classifier(NamedTuple):
    """A loaded tile classifier and the training config its tiling and transform need."""

    model: Any
    cfg: dict[str, Any]


class DefectSettings(NamedTuple):
    """Thresholds in CODES order, plus the object and road cover limits for tile filtering."""

    thresholds: tuple[float, ...]
    max_object_cover: float
    min_road_cover: float


def defect_settings(cfg: dict[str, Any]) -> DefectSettings:
    """The `defects:` section of the shared config; thresholds must name exactly the classes."""
    section = cfg["defects"]
    by_code = section["thresholds"]
    if set(by_code) != set(CODES):
        raise ValueError(
            f"defects.thresholds must have exactly the keys {list(CODES)}, got {by_code}"
        )
    return DefectSettings(
        tuple(float(by_code[code]) for code in CODES),
        float(section["max_object_cover"]),
        float(section["min_road_cover"]),
    )


@lru_cache(maxsize=1)
def _load_classifier(checkpoint: Path) -> Classifier:
    import json

    from defects.train import load_checkpoint

    if not checkpoint.is_file():
        raise FileNotFoundError(f"classifier checkpoint not found: {checkpoint}")
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    # load_checkpoint checks the file hash and class order against the sidecar before loading.
    model = load_checkpoint(checkpoint, sidecar["config"]).eval()
    return Classifier(model, sidecar["config"])


def load_classifier() -> Classifier:
    """The configured classifier, loaded once per process."""
    path = get_config()["defects"]["classifier_checkpoint"]
    if not path:
        raise ValueError(
            "no classifier checkpoint set: put its path in defects.classifier_checkpoint "
            "or the DEFECT_CHECKPOINT environment variable"
        )
    return _load_classifier(Path(path).resolve())


def find_defects(
    frame: np.ndarray,
    *,
    objects: Sequence[Object] = (),
    road_mask: RoadMask | None = None,
    classifier: Classifier | None = None,
    settings: DefectSettings | None = None,
) -> list[Defect]:
    """Find road defects in an RGB uint8 frame (H, W, 3). Boxes are pixel (x1, y1, x2, y2).

    objects and road_mask are optional filters: tiles mostly covered by objects, or mostly off the
    road, are not scored. With no mask the road is unknown and no tile is dropped. classifier and
    settings default to the configured ones.
    """
    cfg = get_config()
    if cfg["pipeline"]["use_stubs"]:
        return _stub_defects(frame)
    from defects.tile_defects import find_frame_defects

    used = classifier or load_classifier()
    chosen = settings or defect_settings(cfg)
    return find_frame_defects(
        used.model,
        frame,
        objects,
        used.cfg,
        thresholds=chosen.thresholds,
        max_object_cover=chosen.max_object_cover,
        road_mask=road_mask,
        min_road_cover=chosen.min_road_cover,
    )


def _stub_defects(frame: np.ndarray) -> list[Defect]:
    h, w = frame.shape[:2]  # shape is (height, width, channels)
    # Stub only: class names/ids are placeholders, not the RDD2022 label map (defects/labels.py).
    return [
        Defect(
            cls=4,
            name="pothole",
            confidence=0.84,
            box=(0.42 * w, 0.78 * h, 0.52 * w, 0.88 * h),
            patch_row=6,
            patch_col=3,
        )
    ]
