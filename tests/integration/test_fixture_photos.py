"""The 20 Stellenbosch photos through the pipeline: stubs (CI) and real models (skips if absent)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from api.pipeline import Pipeline, model_status
from common import config

FIXTURES = sorted(Path(__file__).parents[1].joinpath("fixtures", "stellenbosch").glob("*.jpg"))


def test_every_fixture_photo_is_analyzed_with_stubs() -> None:
    assert len(FIXTURES) == 20
    ids = set()
    for path in FIXTURES:
        record = Pipeline().analyze(path.read_bytes())
        assert record.image_size[0] == 1280 and record.image_size[1] > 0, path.name
        ids.add(record.frame_id)
    assert len(ids) == len(FIXTURES)


@pytest.fixture
def stubs_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("USE_STUBS", "0")
    config.get_config.cache_clear()
    yield
    config.get_config.cache_clear()


@pytest.mark.ml
def test_every_fixture_photo_is_analyzed_with_the_real_models(stubs_off: None) -> None:
    status = model_status(config.get_config())
    if not all(status[k] == "available" for k in ("perception", "defects", "road_mask")):
        pytest.skip(f"real models are not all in place: {status}")
    pipeline = Pipeline()
    for path in FIXTURES:
        record = pipeline.analyze(path.read_bytes())
        width, height = record.image_size
        assert width == 1280, path.name
        for box in [*(o.box for o in record.objects), *(d.box for d in record.defects)]:
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, (path.name, box)
