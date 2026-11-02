"""Object detection entry point: the COCO detector behind detect(), or a stub.

With USE_STUBS on, detect returns two fixed fake objects. With stubs off it loads the model named
by perception.model in configs/default.yaml (env DETECTOR_MODEL) from
<paths.checkpoint_dir>/detector, and keeps detections at or above perception.score_threshold.
If perception.finetuned_checkpoint (env DETECTOR_CHECKPOINT) is set, the 8-class fine-tuned model
(ADR 0015) is loaded from that file instead, after its hash matches perception.finetuned_sha256,
and barrier and traffic_cone use their own cutoffs from perception.new_class_cutoffs.
If the model cannot be loaded, detect raises DetectorUnavailable instead of returning no objects:
an empty list would silently lower hazard scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from common.config import get_config
from fusion.schema import Object
from perception.detector import (
    DETECTOR_SUBDIR,
    DETECTORS,
    DetectorSpec,
    DetectorUnavailable,
    build_model,
    detect_objects,
    load_detector_model,
)
from perception.finetune import NEW_CLASSES, class_name, swap_head


def detector_settings(cfg: dict[str, Any]) -> tuple[str, float]:
    """The configured model name and score threshold, from the `perception:` config section."""
    section = cfg["perception"]
    threshold = section["score_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ValueError(f"perception.score_threshold must be a number, got {threshold!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"perception.score_threshold must be in [0, 1], got {threshold}")
    return str(section["model"]), float(threshold)


@dataclass(frozen=True)
class TunedSettings:
    """The fine-tuned detector: its file, the hash it must have, and the new-class cutoffs."""

    path: Path
    sha256: str
    cutoffs: dict[str, float]


def tuned_settings(cfg: dict[str, Any]) -> TunedSettings | None:
    """The fine-tuned detector's settings, or None when the zero-shot detector is configured."""
    section = cfg["perception"]
    checkpoint = section.get("finetuned_checkpoint")
    if not checkpoint:
        return None
    sha = section.get("finetuned_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("perception.finetuned_sha256 must be the 64-character hash of the file")
    cutoffs = section.get("new_class_cutoffs") or {}
    if set(cutoffs) != set(NEW_CLASSES):
        raise ValueError(f"perception.new_class_cutoffs needs exactly {sorted(NEW_CLASSES)}")
    for label, cutoff in cutoffs.items():
        if isinstance(cutoff, bool) or not isinstance(cutoff, int | float) or not 0 <= cutoff <= 1:
            raise ValueError(
                f"perception.new_class_cutoffs.{label} must be in [0, 1], got {cutoff!r}"
            )
    return TunedSettings(Path(checkpoint), sha, {k: float(v) for k, v in cutoffs.items()})


@lru_cache(maxsize=1)
def _load_tuned(path: Path, sha256: str, base: str) -> Any:
    spec = DetectorSpec(base, "", "", sha256)
    try:
        return load_detector_model(path, spec, factory=lambda name: swap_head(build_model(name)))
    except DetectorUnavailable:
        raise
    except (ImportError, OSError, RuntimeError) as exc:
        raise DetectorUnavailable(f"fine-tuned detector could not be loaded: {exc}") from exc


@lru_cache(maxsize=1)
def _load(weights: Path, name: str) -> Any:
    if name not in DETECTORS:
        raise DetectorUnavailable(f"unknown detector {name!r}; choose one of {sorted(DETECTORS)}")
    try:
        return load_detector_model(weights, DETECTORS[name])
    except DetectorUnavailable:
        raise
    except (ImportError, OSError, RuntimeError) as exc:
        raise DetectorUnavailable(f"detector {name} could not be loaded: {exc}") from exc


def detect(frame: np.ndarray, *, model: Any = None) -> list[Object]:
    """Detect objects in an RGB uint8 frame (H, W, 3). Boxes are pixel (x1, y1, x2, y2).

    model defaults to the configured detector, loaded once per process. Raises DetectorUnavailable
    when stubs are off and that model is missing, changed or cannot be loaded.
    """
    cfg = get_config()
    if cfg["pipeline"]["use_stubs"]:
        return _stub_objects(frame)
    name, threshold = detector_settings(cfg)
    tuned = tuned_settings(cfg)
    if tuned is not None:
        model = model or _load_tuned(tuned.path, tuned.sha256, name)
        return detect_objects(
            model, frame, score_threshold=threshold, labeler=class_name, cutoffs=tuned.cutoffs
        )
    if model is None:
        weights = Path(cfg["paths"]["checkpoint_dir"]) / DETECTOR_SUBDIR / f"{name}.safetensors"
        model = _load(weights, name)
    return detect_objects(model, frame, score_threshold=threshold)


def _stub_objects(frame: np.ndarray) -> list[Object]:
    # Stub values: fixed boxes placed as fractions of the frame so they scale to any image size.
    h, w = frame.shape[:2]
    return [
        Object(label="pedestrian", score=0.91, box=(0.40 * w, 0.45 * h, 0.45 * w, 0.70 * h)),
        Object(label="car", score=0.88, box=(0.55 * w, 0.50 * h, 0.75 * w, 0.68 * h)),
    ]
