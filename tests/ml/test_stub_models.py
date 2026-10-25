import io

import numpy as np
import pytest
from PIL import Image

from defects.infer import find_defects
from fusion.schema import Defect, Object
from perception.detector import DetectorUnavailable
from perception.infer import detect

pytestmark = pytest.mark.ml


def _frame(frame_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes into the RGB array the detectors expect."""
    return np.asarray(Image.open(io.BytesIO(frame_bytes)).convert("RGB"))


def test_stub_outputs_fit_inside_frame(frame_bytes: bytes) -> None:
    """Stub boxes stay inside the image, so downstream code never sees out-of-range coordinates."""
    frame = _frame(frame_bytes)
    h, w = frame.shape[:2]
    items: list[Object | Defect] = [*detect(frame), *find_defects(frame)]
    for item in items:
        x1, y1, x2, y2 = item.box
        assert 0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h


def test_detector_without_weights_raises(monkeypatch, tmp_path, frame_bytes: bytes) -> None:  # type: ignore[no-untyped-def]
    """With stubs off and no weights, the detector raises instead of returning fake results.

    CHECKPOINT_DIR points at an empty folder so weights fetched on this machine are not used.
    """
    from common import config

    monkeypatch.setenv("USE_STUBS", "0")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    config.get_config.cache_clear()
    try:
        with pytest.raises(DetectorUnavailable):
            detect(_frame(frame_bytes))
    finally:
        monkeypatch.undo()
        config.get_config.cache_clear()
