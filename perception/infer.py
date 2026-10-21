"""Object detection entry point. Stub until workstream A lands the real Faster R-CNN."""

from __future__ import annotations

import numpy as np

from common.config import get_config
from fusion.schema import Object


def detect(frame: np.ndarray) -> list[Object]:
    """Detect objects in an RGB uint8 frame (H, W, 3). Boxes are pixel (x1, y1, x2, y2)."""
    # Fail loudly rather than return fake results when stubs are off but no real model exists.
    if not get_config()["pipeline"]["use_stubs"]:
        raise NotImplementedError("Real detector arrives with workstream A (perception).")
    # Stub values: fixed boxes placed as fractions of the frame so they scale to any image size.
    h, w = frame.shape[:2]
    return [
        Object(label="pedestrian", score=0.91, box=(0.40 * w, 0.45 * h, 0.45 * w, 0.70 * h)),
        Object(label="car", score=0.88, box=(0.55 * w, 0.50 * h, 0.75 * w, 0.68 * h)),
    ]
