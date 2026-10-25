"""find_defects: config-driven settings, the real path with an injected classifier, and the stub."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from common import config
from common.config import load_config
from fusion.schema import Object

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from defects.infer import (  # noqa: E402
    Classifier,
    DefectSettings,
    defect_settings,
    find_defects,
)

SIDE = 16
FRAME_W, FRAME_H = 200, 100


def red_detector() -> Any:
    """Class 0 (D00) fires on a bright red channel; the other classes sit at 0.5."""
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
    )
    with torch.no_grad():
        model[2].weight.zero_()
        model[2].weight[0, 0] = 10.0
        model[2].bias.zero_()
    return model


@pytest.fixture
def classifier() -> Classifier:
    cfg = {
        "input_size": SIDE,
        "batch_size": 4,
        "num_workers": 0,
        "amp": "none",
        "channels_last": False,
        "tile_size": 100,
        "tile_overlap": 0.0,
    }
    return Classifier(red_detector(), cfg)


@pytest.fixture
def real_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("USE_STUBS", "0")
    config.get_config.cache_clear()
    yield
    config.get_config.cache_clear()


def white_frame() -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)  # (height, width, channels)
    frame[:, :100] = 255  # the left tile is white, the right tile black
    return frame


def test_shipped_settings_match_the_tuned_thresholds() -> None:
    settings = defect_settings(load_config())
    assert settings == DefectSettings((0.90, 0.75, 0.65, 0.75), 0.30, 0.4)


@pytest.mark.parametrize("keys", [["D00", "D10", "D20"], ["D00", "D10", "D20", "D40", "D50"]])
def test_thresholds_need_exactly_the_four_class_codes(keys: list[str]) -> None:
    cfg = load_config()
    cfg["defects"]["thresholds"] = {k: 0.5 for k in keys}
    with pytest.raises(ValueError, match="thresholds"):
        defect_settings(cfg)


def test_env_sets_the_classifier_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_config()["defects"]["classifier_checkpoint"] is None
    monkeypatch.setenv("DEFECT_CHECKPOINT", "somewhere/best.pt")
    assert load_config()["defects"]["classifier_checkpoint"] == "somewhere/best.pt"


def test_stubs_on_still_returns_the_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_STUBS", "1")
    config.get_config.cache_clear()
    try:
        assert find_defects(white_frame())[0].name == "pothole"
    finally:
        config.get_config.cache_clear()


@pytest.mark.usefixtures("real_path")
def test_stubs_off_without_a_checkpoint_fails_clearly() -> None:
    with pytest.raises(ValueError, match="DEFECT_CHECKPOINT"):
        find_defects(white_frame())


@pytest.mark.usefixtures("real_path")
def test_stubs_off_missing_checkpoint_file_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("DEFECT_CHECKPOINT", str(tmp_path / "nope.pt"))
    config.get_config.cache_clear()
    with pytest.raises(FileNotFoundError, match=r"nope.pt"):
        find_defects(white_frame())


@pytest.mark.usefixtures("real_path")
def test_real_path_reports_the_white_tile(classifier: Classifier) -> None:
    defects = find_defects(white_frame(), classifier=classifier)
    assert [(d.name, d.patch_col) for d in defects] == [("D00", 0)]


@pytest.mark.usefixtures("real_path")
def test_objects_covering_a_tile_remove_it(classifier: Classifier) -> None:
    car = Object(label="car", score=0.9, box=(0.0, 0.0, 100.0, 100.0))
    assert find_defects(white_frame(), objects=[car], classifier=classifier) == []


@pytest.mark.usefixtures("real_path")
def test_road_mask_removes_off_road_tiles(classifier: Classifier) -> None:
    on_right_only = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    on_right_only[:, 100:] = True
    assert find_defects(white_frame(), road_mask=on_right_only, classifier=classifier) == []


@pytest.mark.usefixtures("real_path")
def test_cover_limit_comes_from_settings(classifier: Classifier) -> None:
    # The car hides about 60% of the white tile: dropped at the shipped 0.30, kept at 0.90.
    car = Object(label="car", score=0.9, box=(0.0, 0.0, 100.0, 60.0))
    assert find_defects(white_frame(), objects=[car], classifier=classifier) == []
    cfg = load_config()
    cfg["defects"]["max_object_cover"] = 0.9
    kept = find_defects(
        white_frame(), objects=[car], classifier=classifier, settings=defect_settings(cfg)
    )
    assert [d.name for d in kept] == ["D00"]


@pytest.mark.usefixtures("real_path")
def test_region_limits_scoring_to_part_of_the_frame(classifier: Classifier) -> None:
    left, right = (0, 0, 100, 100), (100, 0, 200, 100)
    (defect,) = find_defects(white_frame(), classifier=classifier, region=left)
    assert (defect.name, defect.box) == ("D00", (0.0, 0.0, 100.0, 100.0))
    assert find_defects(white_frame(), classifier=classifier, region=right) == []


@pytest.mark.usefixtures("real_path")
def test_region_boxes_stay_in_frame_pixels(classifier: Classifier) -> None:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, 100:] = 255  # white right tile only
    (defect,) = find_defects(frame, classifier=classifier, region=(100, 0, 200, 100))
    assert defect.box == (100.0, 0.0, 200.0, 100.0)
