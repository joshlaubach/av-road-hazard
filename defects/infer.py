"""Defect detection entry point. Stub until workstream B lands road-masked tiling + EfficientNet."""

from __future__ import annotations

import numpy as np

from common.config import get_config
from fusion.schema import Defect


def find_defects(frame: np.ndarray) -> list[Defect]:
    """Find road defects in an RGB uint8 frame (H, W, 3). Boxes are pixel (x1, y1, x2, y2)."""
    # Fail loudly rather than return fake results when stubs are off but no real model exists.
    if not get_config()["pipeline"]["use_stubs"]:
        raise NotImplementedError("Real defect finder arrives with workstream B (defects).")
    h, w = frame.shape[:2]  # shape is (height, width, channels)
    # Stub only: class names/ids are placeholders. The real label map must be verified first
    # (docs/SOURCE_REPOS.md, S-11).
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
