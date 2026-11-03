"""/health says "ready" only for models that loaded at startup, not for files that exist."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main as main_module
from api import pipeline as pipeline_module
from api.pipeline import Pipeline
from common import config
from defects.infer import DefectsUnavailable
from fusion.fuse import RoadMask
from rag.embed import Embedder, HashEmbedder
from rag.service import RagService


@pytest.fixture
def stubs_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("USE_STUBS", "0")
    config.get_config.cache_clear()
    yield
    config.get_config.cache_clear()


def _mask(frame: np.ndarray) -> RoadMask:
    return np.ones(frame.shape[:2], dtype=bool)


def _fine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "detect", lambda frame: [])
    monkeypatch.setattr(pipeline_module, "find_defects", lambda frame, **kw: [])


def _fail_to_find_defects(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*args: Any, **kwargs: Any) -> Any:
        raise DefectsUnavailable("classifier checkpoint could not be loaded")

    monkeypatch.setattr(pipeline_module, "find_defects", down)


def _no_package(frame: np.ndarray) -> RoadMask:
    raise ImportError("No module named 'segmentation_models_pytorch'")


@pytest.mark.usefixtures("stubs_off")
def test_warm_up_marks_every_model_ready_when_each_one_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fine(monkeypatch)
    assert Pipeline(segmenter=_mask).warm_up() == {
        "perception": "ready",
        "defects": "ready",
        "road_mask": "ready",
    }


@pytest.mark.usefixtures("stubs_off")
def test_warm_up_marks_only_the_failing_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fine(monkeypatch)
    _fail_to_find_defects(monkeypatch)
    assert Pipeline(segmenter=_mask).warm_up() == {
        "perception": "ready",
        "defects": "unavailable",
        "road_mask": "ready",
    }


@pytest.mark.usefixtures("stubs_off")
def test_a_road_model_that_cannot_be_imported_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fine(monkeypatch)
    assert Pipeline(segmenter=_no_package).warm_up()["road_mask"] == "unavailable"


def test_warm_up_loads_nothing_with_stubs_on() -> None:
    assert Pipeline().warm_up() == {}


@pytest.mark.usefixtures("stubs_off")
def test_health_reports_what_the_startup_check_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Pipeline,
        "warm_up",
        lambda self: {"perception": "ready", "defects": "ready", "road_mask": "unavailable"},
    )
    monkeypatch.setattr(
        main_module,
        "model_status",
        lambda cfg: {
            "perception": "available",
            "defects": "available",
            "road_mask": "available",
            "fusion": "ready",
            "rag": "stub",
        },
    )
    with TestClient(main_module.create_app()) as client:
        body = client.get("/health").json()
    assert body["models"]["perception"] == "ready"
    assert body["models"]["road_mask"] == "unavailable"
    assert body["status"] == "degraded"


def _all_models_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    ready = {"perception": "ready", "defects": "ready", "road_mask": "ready"}
    monkeypatch.setattr(Pipeline, "warm_up", lambda self: ready)


@pytest.mark.usefixtures("stubs_off")
def test_health_says_search_is_ready_once_the_embedder_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_models_ready(monkeypatch)
    monkeypatch.setattr(RagService, "embedder", property(lambda self: HashEmbedder()))
    with TestClient(main_module.create_app()) as client:
        body = client.get("/health").json()
    assert body["models"]["rag"] == "ready"
    assert body["status"] == "ok"


@pytest.mark.usefixtures("stubs_off")
def test_health_says_search_is_unavailable_when_the_embedder_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_package(self: RagService) -> Embedder:
        raise ImportError("No module named 'open_clip'")

    _all_models_ready(monkeypatch)
    monkeypatch.setattr(RagService, "embedder", property(no_package))
    with TestClient(main_module.create_app()) as client:
        body = client.get("/health").json()
    assert body["models"]["rag"] == "unavailable"
    assert body["status"] == "degraded"
