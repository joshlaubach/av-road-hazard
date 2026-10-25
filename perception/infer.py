"""Object detection entry point: the COCO detector behind detect(), or a stub.

With USE_STUBS on, detect returns two fixed fake objects. With stubs off it loads the model named
by perception.model in configs/default.yaml (env DETECTOR_MODEL) from
<paths.checkpoint_dir>/detector, and keeps detections at or above perception.score_threshold.
If the model cannot be loaded, detect raises DetectorUnavailable instead of returning no objects:
an empty list would silently lower hazard scores.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from common.config import get_config
from fusion.schema import Object
from perception.detector import (
    DETECTOR_SUBDIR,
    DETECTORS,
    DetectorUnavailable,
    detect_objects,
    load_detector_model,
)


def detector_settings(cfg: dict[str, Any]) -> tuple[str, float]:
    """The configured model name and score threshold, from the `perception:` config section."""
    section = cfg["perception"]
    threshold = section["score_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ValueError(f"perception.score_threshold must be a number, got {threshold!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"perception.score_threshold must be in [0, 1], got {threshold}")
    return str(section["model"]), float(threshold)


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
