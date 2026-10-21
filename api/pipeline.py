"""Service object that turns an image into a HazardRecord. Stub fusion until workstream C."""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from common.config import get_config
from common.debug import dump
from defects.infer import find_defects
from fusion.schema import DefectRecord, HazardRecord
from perception.infer import detect


class InvalidImage(ValueError):
    """The uploaded bytes are not a usable image (corrupt, unsupported, or too large).

    A ValueError subclass so callers can catch it specifically; the API maps it to HTTP 422.
    """


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB uint8 array of shape (height, width, 3).

    The pixel cap is checked from the file header before decoding, so a tiny file that expands to
    a huge image (a "decompression bomb") is rejected without allocating the memory first.
    """
    max_pixels = get_config()["api"]["max_image_pixels"]
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > max_pixels:  # header only; pixels not decoded yet
                raise InvalidImage("image dimensions too large")
            img.load()
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage("could not decode image") from exc


def frame_id_for(data: bytes) -> str:
    """Deterministic id from the image bytes, so uploading the same image twice gives the same
    record id (the second upload overwrites the first in the store). 16 hex chars is plenty here."""
    return "up-" + hashlib.sha256(data).hexdigest()[:16]


class Pipeline:
    """Loads models once (none yet) and analyzes frames."""

    def analyze(self, data: bytes, source: str = "upload") -> HazardRecord:
        """Run one image through detection and defect finding and return a HazardRecord.

        Currently: decode, call the two (stub) detectors, dump intermediates for debugging when
        DEBUG_DUMP_DIR is set, and wrap the defects in placeholder fusion fields.
        """
        frame = decode_image(data)
        h, w = frame.shape[:2]  # numpy arrays are (height, width, channels)
        objects = detect(frame)
        defects = find_defects(frame)
        dump("objects", [o.model_dump(mode="json") for o in objects])
        dump("defects_raw", [d.model_dump(mode="json") for d in defects])
        # Stub fusion: workstream C replaces this with real road-mask and proximity logic.
        record = HazardRecord(
            frame_id=frame_id_for(data),
            source=source,
            image_size=(w, h),
            objects=objects,
            defects=[
                DefectRecord(
                    name=d.name,
                    confidence=d.confidence,
                    box=d.box,
                    severity="unknown",
                    in_road=True,
                    near_objects=[],
                )
                for d in defects
            ],
            hazard_score=0.0,
        )
        dump("record", record)
        return record
