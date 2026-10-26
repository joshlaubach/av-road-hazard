"""Service object: image in, HazardRecord out. With USE_STUBS=1 the detectors are fixed fakes."""

from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from common.config import get_config
from common.debug import dump
from defects.infer import find_defects
from defects.road_crop import (
    MODEL_SUBDIR,
    REQUIRED_FILES,
    ModelIntegrityError,
    RoadSegmenter,
)
from fusion.fuse import RoadMask, fuse
from fusion.schema import HazardRecord, Source
from perception.detector import DETECTOR_SUBDIR
from perception.infer import detect

log = logging.getLogger("api")


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


class Analysis(NamedTuple):
    """One analyzed frame: the record, and whether a road mask was used for it.

    road_mask_used False means every defect's in_road is unknown (null), not "off the road".
    """

    record: HazardRecord
    road_mask_used: bool


def model_status(cfg: Mapping[str, Any]) -> dict[str, str]:
    """What /health reports per component, from the config and the files on disk.

    Stubs on: the fixed stub answer. Stubs off: "available" when the files a model needs exist,
    else "unavailable". This checks files only; hashes are verified when a model first loads.
    """
    if cfg["pipeline"]["use_stubs"]:
        return {"perception": "stub", "defects": "stub", "fusion": "stub", "rag": "stub"}
    checkpoints = Path(cfg["paths"]["checkpoint_dir"])
    detector = checkpoints / DETECTOR_SUBDIR / f"{cfg['perception']['model']}.safetensors"
    classifier = cfg["defects"]["classifier_checkpoint"]
    classifier_ok = bool(classifier) and (
        Path(classifier).is_file() and Path(classifier).with_suffix(".json").is_file()
    )
    road_ok = all((checkpoints / MODEL_SUBDIR / name).is_file() for name in REQUIRED_FILES)

    def state(ok: bool) -> str:
        return "available" if ok else "unavailable"

    return {
        "perception": state(detector.is_file()),
        "defects": state(classifier_ok),
        "road_mask": state(road_ok),
        "fusion": "ready",
        "rag": "stub",
    }


class Pipeline:
    """Loads models once (the road model on first use) and analyzes frames."""

    def __init__(self, segmenter: Callable[[np.ndarray], RoadMask] | None = None) -> None:
        # Injected in tests; otherwise built on first use from the checkpoint directory.
        self._segmenter = segmenter

    def road_mask(self, frame: np.ndarray) -> RoadMask | None:
        """Boolean (height, width) road mask for the frame, or None when none is available.

        None means "unavailable or untrustworthy" (stubs on, or the road model is missing,
        unverified or failed): fusion then reports in_road as unknown (null) and still scores the
        defect. An all-False mask is a different, legitimate answer ("no road detected") and makes
        every defect in_road=False. Do not turn a failure into an empty mask. With stubs on no
        model is loaded; with stubs off a failure is logged and gives None.
        """
        if get_config()["pipeline"]["use_stubs"]:
            return None
        try:
            if self._segmenter is None:
                weights_dir = Path(get_config()["paths"]["checkpoint_dir"]) / MODEL_SUBDIR
                self._segmenter = RoadSegmenter(weights_dir)
            return self._segmenter(frame)
        except (ModelIntegrityError, OSError, ImportError, RuntimeError) as exc:
            log.error("road mask unavailable, in_road stays unknown: %s", exc)
            return None

    def analyze(self, data: bytes, source: Source = "upload") -> HazardRecord:
        """Run one image through detection, defect finding and fusion; return a HazardRecord."""
        return self.analyze_checked(data, source).record

    def analyze_checked(self, data: bytes, source: Source = "upload") -> Analysis:
        """Like analyze, but also says whether a road mask was used.

        Decode, call the two detectors, dump intermediates for debugging when DEBUG_DUMP_DIR is
        set, then fuse them with the road mask (if any) into the record. Severity stays "unknown"
        inside fuse() until severity is turned on (docs/ROADMAP.md, Next row 3).
        """
        frame = decode_image(data)
        h, w = frame.shape[:2]  # numpy arrays are (height, width, channels)
        objects = detect(frame)
        road_mask = self.road_mask(frame)
        defects = find_defects(frame, objects=objects, road_mask=road_mask)
        dump("objects", [o.model_dump(mode="json") for o in objects])
        dump("defects_raw", [d.model_dump(mode="json") for d in defects])
        record = fuse(
            frame_id=frame_id_for(data),
            source=source,
            image_size=(w, h),
            objects=objects,
            defects=defects,
            road_mask=road_mask,
        )
        dump("record", record)
        return Analysis(record, road_mask is not None)
