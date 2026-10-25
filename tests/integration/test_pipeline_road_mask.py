"""Pipeline.road_mask: the road model behind the stub switch, and its failure behaviour."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from api import pipeline as pipeline_module
from api.pipeline import Pipeline, decode_image
from common import config
from defects.road_crop import MODEL_SUBDIR, ModelIntegrityError

FRAME = np.zeros((20, 30, 3), dtype=np.uint8)


@pytest.fixture
def stubs_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("USE_STUBS", "0")
    config.get_config.cache_clear()
    yield
    config.get_config.cache_clear()


class Recorder:
    def __init__(self, mask: Any = None, error: Exception | None = None) -> None:
        self.mask, self.error, self.calls = mask, error, 0

    def __call__(self, frame: np.ndarray) -> Any:
        self.calls += 1
        if self.error:
            raise self.error
        return self.mask


def test_stubs_on_gives_no_mask_and_never_loads_the_model() -> None:
    segmenter = Recorder(np.ones((20, 30), dtype=bool))
    assert Pipeline(segmenter=segmenter).road_mask(FRAME) is None
    assert segmenter.calls == 0


@pytest.mark.usefixtures("stubs_off")
def test_stubs_off_returns_the_segmenters_mask() -> None:
    mask = np.ones((20, 30), dtype=bool)
    assert Pipeline(segmenter=Recorder(mask)).road_mask(FRAME) is mask


@pytest.mark.usefixtures("stubs_off")
@pytest.mark.parametrize(
    "error", [ModelIntegrityError("bad hash"), FileNotFoundError("weights"), ImportError("torch")]
)
def test_a_failing_road_model_gives_none_and_is_logged(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    # None means "unknown", never an empty mask (which would mean "no road anywhere").
    with caplog.at_level(logging.ERROR, logger="api"):
        assert Pipeline(segmenter=Recorder(error=error)).road_mask(FRAME) is None
    assert any(str(error) in record.getMessage() for record in caplog.records)


@pytest.mark.usefixtures("stubs_off")
def test_default_segmenter_reads_the_checkpoint_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def fake_segmenter(weights_dir: Path) -> Recorder:
        seen.append(weights_dir)
        return Recorder(np.ones((20, 30), dtype=bool))

    monkeypatch.setattr(pipeline_module, "RoadSegmenter", fake_segmenter)
    pipeline = Pipeline()
    pipeline.road_mask(FRAME)
    pipeline.road_mask(FRAME)
    assert seen == [Path(config.get_config()["paths"]["checkpoint_dir"]) / MODEL_SUBDIR]


def test_analyze_gives_defect_finder_the_same_mask(
    frame_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, w = decode_image(frame_bytes).shape[:2]
    mask = np.ones((h, w), dtype=bool)
    monkeypatch.setattr(Pipeline, "road_mask", lambda self, frame: mask)
    captured: dict[str, Any] = {}

    def fake_find_defects(frame: np.ndarray, **kwargs: Any) -> list[Any]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pipeline_module, "find_defects", fake_find_defects)
    Pipeline().analyze(frame_bytes)
    assert captured["road_mask"] is mask
    assert captured["objects"]
