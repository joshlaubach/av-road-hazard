"""Service object that turns an image into a HazardRecord. Stub fusion until workstream C."""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from common.debug import dump
from defects.infer import find_defects
from fusion.schema import DefectRecord, HazardRecord
from perception.infer import detect


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage("could not decode image") from exc


def frame_id_for(data: bytes) -> str:
    return "up-" + hashlib.sha256(data).hexdigest()[:16]


class Pipeline:
    """Loads models once (none yet) and analyzes frames."""

    def analyze(self, data: bytes, source: str = "upload") -> HazardRecord:
        frame = decode_image(data)
        h, w = frame.shape[:2]
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
