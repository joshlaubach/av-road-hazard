"""perception.detect: config-driven settings, the real path with an injected model, and the stub."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from common import config
from common.config import load_config
from perception import infer
from perception.detector import DETECTORS, DetectorUnavailable
from perception.infer import detect, detector_settings, tuned_settings

FRAME = np.full((100, 200, 3), 255, dtype=np.uint8)  # (height, width, channels)


@pytest.fixture
def stubs_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Real path on, with an empty checkpoint dir so no weights on this machine are picked up."""
    monkeypatch.setenv("USE_STUBS", "0")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    config.get_config.cache_clear()
    yield tmp_path
    config.get_config.cache_clear()


def test_shipped_settings_name_a_pinned_model() -> None:
    model, threshold = detector_settings(load_config())
    assert model in DETECTORS and 0.0 < threshold < 1.0


def test_env_picks_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTOR_MODEL", "fasterrcnn_resnet50_fpn_v2")
    assert detector_settings(load_config())[0] == "fasterrcnn_resnet50_fpn_v2"


@pytest.mark.parametrize("bad", [-0.1, 1.1, "high"])
def test_the_score_threshold_must_be_a_probability(bad: Any) -> None:
    cfg = load_config()
    cfg["perception"]["score_threshold"] = bad
    with pytest.raises(ValueError, match="score_threshold"):
        detector_settings(cfg)


def test_stubs_on_still_returns_the_two_stub_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_STUBS", "1")
    config.get_config.cache_clear()
    try:
        assert [o.label for o in detect(FRAME)] == ["pedestrian", "car"]
    finally:
        config.get_config.cache_clear()


@pytest.mark.usefixtures("stubs_off")
def test_missing_weights_say_how_to_fetch_them() -> None:
    with pytest.raises(DetectorUnavailable, match="fetch_detector"):
        detect(FRAME)


@pytest.mark.usefixtures("stubs_off")
def test_an_unknown_model_name_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTOR_MODEL", "nope")
    config.get_config.cache_clear()
    with pytest.raises(DetectorUnavailable, match="nope"):
        detect(FRAME)


@pytest.mark.usefixtures("stubs_off")
@pytest.mark.parametrize("error", [ImportError("torch"), OSError("disk"), RuntimeError("bad")])
def test_load_failures_become_detector_unavailable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(infer, "load_detector_model", boom)
    with pytest.raises(DetectorUnavailable, match=str(error)):
        detect(FRAME)


class FakeModel:
    """Returns one car and one low-score pedestrian in torchvision's output format."""

    def __call__(self, images: list[Any]) -> list[dict[str, Any]]:
        import torch

        return [
            {
                "boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0], [60.0, 10.0, 90.0, 40.0]]),
                "labels": torch.tensor([3, 1]),
                "scores": torch.tensor([0.9, 0.2]),
            }
        ]

    def parameters(self) -> Any:
        import torch

        return iter([torch.zeros(1)])


@pytest.mark.usefixtures("stubs_off")
def test_real_path_uses_the_model_and_the_configured_threshold() -> None:
    pytest.importorskip("torch")
    objects = detect(FRAME, model=FakeModel())
    assert [(o.label, o.box) for o in objects] == [("car", (10.0, 20.0, 50.0, 80.0))]


def _tuned_cfg(**changes: Any) -> dict[str, Any]:
    cfg = load_config()
    cfg["perception"].update(
        {"finetuned_checkpoint": "x.safetensors", "finetuned_sha256": "a" * 64}
    )
    cfg["perception"].update(changes)
    return cfg


def test_the_shipped_config_uses_the_zero_shot_detector() -> None:
    assert tuned_settings(load_config()) is None


def test_the_tuned_detector_needs_its_hash_and_both_cutoffs() -> None:
    assert tuned_settings(_tuned_cfg()).cutoffs == {"barrier": 0.55, "traffic_cone": 0.65}  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="finetuned_sha256"):
        tuned_settings(_tuned_cfg(finetuned_sha256=None))
    with pytest.raises(ValueError, match="new_class_cutoffs"):
        tuned_settings(_tuned_cfg(new_class_cutoffs={"barrier": 0.5}))
    with pytest.raises(ValueError, match="traffic_cone"):
        tuned_settings(_tuned_cfg(new_class_cutoffs={"barrier": 0.5, "traffic_cone": 1.5}))


@pytest.mark.usefixtures("stubs_off")
def test_a_missing_tuned_file_is_unavailable_not_silently_zero_shot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DETECTOR_CHECKPOINT", str(tmp_path / "gone.safetensors"))
    monkeypatch.setattr(
        infer,
        "tuned_settings",
        lambda cfg: infer.TunedSettings(tmp_path / "gone.safetensors", "a" * 64, {}),
    )
    with pytest.raises(DetectorUnavailable):
        detect(FRAME)
