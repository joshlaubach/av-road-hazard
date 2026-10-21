import io

import numpy as np
import pytest
from PIL import Image

from defects.infer import find_defects
from fusion.schema import Defect, Object
from perception.infer import detect

pytestmark = pytest.mark.ml


def _frame(frame_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(frame_bytes)).convert("RGB"))


def test_stub_outputs_fit_inside_frame(frame_bytes: bytes) -> None:
    frame = _frame(frame_bytes)
    h, w = frame.shape[:2]
    items: list[Object | Defect] = [*detect(frame), *find_defects(frame)]
    for item in items:
        x1, y1, x2, y2 = item.box
        assert 0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h


def test_real_models_not_yet_available(monkeypatch, frame_bytes: bytes) -> None:  # type: ignore[no-untyped-def]
    from common import config

    monkeypatch.setenv("USE_STUBS", "0")
    config.get_config.cache_clear()
    try:
        with pytest.raises(NotImplementedError):
            detect(_frame(frame_bytes))
    finally:
        monkeypatch.delenv("USE_STUBS")
        config.get_config.cache_clear()
