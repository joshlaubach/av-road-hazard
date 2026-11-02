"""The stubs-off path: what /health reports, a missing classifier, and the road-mask flag."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main as main_module
from api import pipeline as pipeline_module
from api.main import app
from api.pipeline import Pipeline, model_status
from common import config
from defects.infer import DefectsUnavailable, load_classifier
from defects.road_crop import MODEL_SUBDIR
from perception.detector import DETECTOR_SUBDIR

client = TestClient(app)


def _cfg(tmp_path: Path, *, stubs: bool, checkpoint: str | None = None) -> dict[str, Any]:
    return {
        "pipeline": {"use_stubs": stubs},
        "paths": {"checkpoint_dir": str(tmp_path)},
        "perception": {"model": "fasterrcnn_mobilenet_v3_large_fpn"},
        "defects": {"classifier_checkpoint": checkpoint},
    }


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_status_with_stubs_on_is_the_old_stub_answer(tmp_path: Path) -> None:
    status = model_status(_cfg(tmp_path, stubs=True))
    assert status == {"perception": "stub", "defects": "stub", "fusion": "stub", "rag": "stub"}


def test_status_with_stubs_off_and_nothing_fetched_says_unavailable(tmp_path: Path) -> None:
    status = model_status(_cfg(tmp_path, stubs=False))
    assert status["perception"] == "unavailable"
    assert status["defects"] == "unavailable"
    assert status["road_mask"] == "unavailable"
    assert status["fusion"] == "ready"
    assert status["rag"] == "stub"


def test_status_with_stubs_off_and_files_in_place_says_available(tmp_path: Path) -> None:
    _touch(tmp_path / DETECTOR_SUBDIR / "fasterrcnn_mobilenet_v3_large_fpn.safetensors")
    for name in ("config.json", "model.safetensors"):
        _touch(tmp_path / MODEL_SUBDIR / name)
    best = tmp_path / "best.pt"
    _touch(best)
    _touch(best.with_suffix(".json"))
    status = model_status(_cfg(tmp_path, stubs=False, checkpoint=str(best)))
    assert (status["perception"], status["defects"], status["road_mask"]) == ("available",) * 3


def test_a_checkpoint_without_its_sidecar_is_unavailable(tmp_path: Path) -> None:
    best = tmp_path / "best.pt"
    _touch(best)
    assert model_status(_cfg(tmp_path, stubs=False, checkpoint=str(best)))["defects"] == (
        "unavailable"
    )


@pytest.fixture
def stubs_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("USE_STUBS", "0")
    config.get_config.cache_clear()
    yield
    config.get_config.cache_clear()


def test_health_is_degraded_while_a_model_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "model_status",
        lambda cfg: {"perception": "available", "defects": "unavailable", "fusion": "ready"},
    )
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["models"]["defects"] == "unavailable"


@pytest.mark.usefixtures("stubs_off")
def test_health_is_ok_when_every_model_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_module,
        "model_status",
        lambda cfg: {"perception": "available", "defects": "available", "rag": "stub"},
    )
    body = TestClient(main_module.create_app()).get("/health").json()
    assert body["status"] == "ok"
    assert body["stubs"] is False


@pytest.mark.usefixtures("stubs_off")
def test_a_missing_classifier_checkpoint_raises_defects_unavailable() -> None:
    with pytest.raises(DefectsUnavailable, match="no classifier checkpoint"):
        load_classifier()


def test_a_classifier_file_that_does_not_exist_raises_defects_unavailable(tmp_path: Path) -> None:
    from defects.infer import _load_classifier

    with pytest.raises(DefectsUnavailable, match="not found"):
        _load_classifier(tmp_path / "missing.pt")


def test_analyze_returns_503_when_the_classifier_is_unavailable(
    frame_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(*args: Any, **kwargs: Any) -> Any:
        raise DefectsUnavailable("/secret/path/best.pt not found")

    monkeypatch.setattr(pipeline_module, "find_defects", down)
    response = client.post("/analyze", files={"file": ("f.png", frame_bytes, "image/png")})
    assert response.status_code == 503
    assert response.json() == {"detail": "defect classifier unavailable"}
    assert "secret" not in response.text


@pytest.mark.parametrize(("mask_ok", "expected"), [(True, True), (False, False)])
def test_analysis_says_whether_a_road_mask_was_used(
    frame_bytes: bytes, monkeypatch: pytest.MonkeyPatch, mask_ok: bool, expected: bool
) -> None:
    mask = np.ones((360, 640), dtype=bool) if mask_ok else None
    pipeline = Pipeline()
    monkeypatch.setattr(pipeline, "road_mask", lambda frame: mask)
    result = pipeline.analyze_checked(frame_bytes)
    assert result.road_mask_used is expected
    assert result.record.image_size == (640, 360)
    assert pipeline.analyze(frame_bytes).frame_id == result.record.frame_id


def test_status_looks_for_the_tuned_detector_file_when_one_is_configured(tmp_path: Path) -> None:
    tuned = tmp_path / "tuned.safetensors"
    cfg = _cfg(tmp_path, stubs=False)
    cfg["perception"]["finetuned_checkpoint"] = str(tuned)
    assert model_status(cfg)["perception"] == "unavailable"
    _touch(tuned)
    assert model_status(cfg)["perception"] == "available"
