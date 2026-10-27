"""api/trace.py: the road-mask preview and the fusion steps that explain a hazard score."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from api.trace import fusion_steps, road_mask_preview
from fusion.config import load_fusion_config
from fusion.fuse import fuse
from fusion.schema import Defect, Object

CFG = load_fusion_config()


def test_no_road_mask_means_no_preview() -> None:
    assert road_mask_preview(None) is None


def test_the_preview_keeps_the_aspect_ratio_stays_small_and_marks_the_road() -> None:
    mask = np.zeros((900, 1600), dtype=bool)
    mask[450:, :] = True  # the lower half is road
    preview = road_mask_preview(mask, max_side=160)
    assert preview is not None
    assert (preview.width, preview.height) == (160, 90)
    image = Image.open(io.BytesIO(base64.b64decode(preview.png_base64))).convert("L")
    pixels = np.asarray(image) > 127
    assert image.size == (160, 90) and pixels[:40].sum() == 0 and pixels[50:].all()


def test_a_small_mask_is_not_enlarged() -> None:
    preview = road_mask_preview(np.ones((30, 40), dtype=bool), max_side=160)
    assert preview is not None and (preview.width, preview.height) == (40, 30)


def _defect(name: str, box: tuple[float, float, float, float], confidence: float) -> Defect:
    return Defect(cls=1, name=name, confidence=confidence, box=box, patch_row=0, patch_col=0)


@pytest.mark.parametrize("with_object", [False, True])
def test_the_fusion_steps_multiply_out_to_the_records_hazard_score(with_object: bool) -> None:
    objects = [Object(label="pedestrian", score=0.9, box=(110.0, 200.0, 130.0, 330.0))]
    defects = [
        _defect("D40", (100.0, 300.0, 140.0, 340.0), 0.8),
        _defect("D00", (300.0, 300.0, 340.0, 340.0), 0.95),
    ]
    used = objects if with_object else []
    road = np.ones((480, 640), dtype=bool)
    record = fuse(
        frame_id="f", source="upload", image_size=(640, 480), objects=used, defects=defects,
        road_mask=road, config=CFG,
    )  # fmt: skip
    steps = fusion_steps(record, used, [2, 1], CFG)
    assert [s.merged_from for s in steps] == [2, 1]
    assert [s.name for s in steps] == ["D40", "D00"]
    for step in steps:
        product = (
            step.road_multiplier
            * step.severity_weight
            * step.confidence
            * (0.5 + 0.5 * step.proximity)
        )
        assert step.score == pytest.approx(product)
    assert max(s.score for s in steps) == pytest.approx(record.hazard_score)
    assert (steps[0].proximity > 0) is with_object
