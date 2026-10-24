"""Road mask model: pinned files, hash-checked loading, and prediction (defects/road_crop.py).

The pure parts run anywhere. The model tests build a tiny random U-Net in tmp_path, so no real
weights are needed; one opt-in test runs the pinned model when ROAD_MODEL_DIR is set.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from defects.road_crop import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MODEL_FILES,
    ModelIntegrityError,
    RoadSegmenter,
    load_road_model,
    predict_road_mask,
    road_mask_from_class_map,
    verify_model_files,
)

# ---------------------------------------------------------------- pure parts


def test_classes_0_and_1_are_road_and_2_3_4_are_not() -> None:
    class_map = np.array([[0, 1, 2, 3, 4]])
    mask = road_mask_from_class_map(class_map, 1, 5)
    assert mask.tolist() == [[True, True, False, False, False]]


def test_upsampling_keeps_height_width_order_and_dtype() -> None:
    class_map = np.array([[0, 2], [2, 0]])
    mask = road_mask_from_class_map(class_map, 6, 10)
    assert mask.shape == (6, 10) and mask.dtype == np.bool_
    assert mask[:3, :5].all() and mask[3:, 5:].all()
    assert not mask[:3, 5:].any() and not mask[3:, :5].any()


def test_downsampling_picks_nearest_cells() -> None:
    class_map = np.zeros((4, 4), dtype=np.int64)
    class_map[:, 2:] = 2
    mask = road_mask_from_class_map(class_map, 2, 2)
    assert mask.tolist() == [[True, False], [True, False]]


def test_upsampling_samples_cell_centres_not_cell_corners() -> None:
    class_map = np.array([[0, 2], [2, 0]])
    mask = road_mask_from_class_map(class_map, 3, 3)
    # Each output pixel looks at the model cell under its centre: rows and columns map to 0, 1, 1.
    assert mask.tolist() == [
        [True, False, False],
        [False, True, True],
        [False, True, True],
    ]


@pytest.mark.parametrize("bad", [(0, 4), (4, 0)])
def test_a_non_positive_output_size_raises(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        road_mask_from_class_map(np.zeros((2, 2), dtype=np.int64), *bad)


def test_a_class_map_that_is_not_2d_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        road_mask_from_class_map(np.zeros((2, 2, 1), dtype=np.int64), 4, 4)


def _write(directory: Path, files: dict[str, bytes]) -> dict[str, str]:
    for name, data in files.items():
        (directory / name).write_bytes(data)
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def test_matching_files_verify(tmp_path: Path) -> None:
    expected = _write(tmp_path, {"a.bin": b"alpha", "b.bin": b"beta"})
    verify_model_files(tmp_path, expected)


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    expected = _write(tmp_path, {"a.bin": b"alpha"})
    (tmp_path / "a.bin").unlink()
    with pytest.raises(ModelIntegrityError, match=r"a\.bin.*missing"):
        verify_model_files(tmp_path, expected)


def test_a_wrong_hash_reports_expected_and_actual(tmp_path: Path) -> None:
    _write(tmp_path, {"a.bin": b"alpha"})
    wrong = "0" * 64
    actual = hashlib.sha256(b"alpha").hexdigest()
    with pytest.raises(ModelIntegrityError, match=rf"a\.bin.*{wrong}.*{actual}"):
        verify_model_files(tmp_path, {"a.bin": wrong})


def test_a_single_flipped_byte_is_caught(tmp_path: Path) -> None:
    data = bytes(range(256)) * 10
    expected = _write(tmp_path, {"a.bin": data})
    flipped = bytearray(data)
    flipped[1000] ^= 0x01
    (tmp_path / "a.bin").write_bytes(bytes(flipped))
    with pytest.raises(ModelIntegrityError, match=r"a\.bin"):
        verify_model_files(tmp_path, expected)


def test_integrity_error_is_a_value_error() -> None:
    assert issubclass(ModelIntegrityError, ValueError)


def test_loading_refuses_a_file_that_has_no_expected_hash(tmp_path: Path) -> None:
    expected = _write(tmp_path, {"config.json": b"{}", "model.safetensors": b"x"})
    del expected["model.safetensors"]
    with pytest.raises(ModelIntegrityError, match=r"model\.safetensors.*unverified"):
        load_road_model(tmp_path, device="cpu", expected=expected)


def test_the_pinned_files_are_the_two_the_code_needs() -> None:
    assert set(MODEL_FILES) == {"config.json", "model.safetensors"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in MODEL_FILES.values())


def test_an_invalid_frame_is_rejected_before_the_model_is_touched() -> None:
    with pytest.raises(ValueError, match="uint8"):
        predict_road_mask(object(), np.zeros((8, 8, 3), dtype=np.float32))


# ---------------------------------------------------------------- tiny model


def _tiny_config() -> dict[str, Any]:
    return {
        "_model_class": "Unet",
        "encoder_name": "resnet18",
        "encoder_weights": None,
        "encoder_depth": 3,
        "decoder_channels": [16, 8, 8],
        "in_channels": 3,
        "classes": 5,
    }


def _write_tiny_model(directory: Path, config: dict[str, Any] | None = None) -> dict[str, str]:
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    config = config or _tiny_config()
    torch.manual_seed(0)
    constructor = {k: v for k, v in config.items() if k != "_model_class"}
    model = smp.Unet(**constructor)
    (directory / "config.json").write_text(json.dumps(config))
    safetensors_torch.save_file(model.state_dict(), str(directory / "model.safetensors"))
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("config.json", "model.safetensors")
    }


@pytest.fixture
def tiny(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    return tmp_path, _write_tiny_model(tmp_path)


def _frame(height: int = 90, width: int = 130) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.mark.ml
def test_a_tiny_model_loads_and_predicts_a_bool_mask_of_the_frame_shape(
    tiny: tuple[Path, dict[str, str]],
) -> None:
    directory, expected = tiny
    model = load_road_model(directory, device="cpu", expected=expected)
    mask = predict_road_mask(model, _frame(90, 130), device="cpu")
    assert mask.shape == (90, 130) and mask.dtype == np.bool_
    assert not model.training


@pytest.mark.ml
def test_two_calls_give_identical_masks(tiny: tuple[Path, dict[str, str]]) -> None:
    directory, expected = tiny
    model = load_road_model(directory, device="cpu", expected=expected)
    frame = _frame()
    first = predict_road_mask(model, frame, device="cpu")
    second = predict_road_mask(model, frame, device="cpu")
    np.testing.assert_array_equal(first, second)


@pytest.mark.ml
@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((0, 8, 3), dtype=np.uint8),
    ],
)
def test_a_frame_that_is_not_rgb_uint8_is_rejected(
    tiny: tuple[Path, dict[str, str]], frame: np.ndarray
) -> None:
    directory, expected = tiny
    model = load_road_model(directory, device="cpu", expected=expected)
    with pytest.raises(ValueError, match="shape"):
        predict_road_mask(model, frame, device="cpu")


@pytest.mark.ml
def test_a_wrong_model_class_is_rejected(tmp_path: Path) -> None:
    config = {**_tiny_config(), "_model_class": "Linknet"}
    expected = _write_tiny_model(tmp_path, config)
    with pytest.raises(ModelIntegrityError, match="Unet"):
        load_road_model(tmp_path, device="cpu", expected=expected)


@pytest.mark.ml
def test_a_config_that_asks_for_pretrained_encoder_weights_is_rejected(tmp_path: Path) -> None:
    expected = _write_tiny_model(tmp_path)
    config = {**_tiny_config(), "encoder_weights": "imagenet"}  # would trigger a download
    (tmp_path / "config.json").write_text(json.dumps(config))
    expected["config.json"] = hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    with pytest.raises(ModelIntegrityError, match="encoder_weights"):
        load_road_model(tmp_path, device="cpu", expected=expected)


@pytest.mark.ml
def test_a_tampered_weights_file_is_rejected(tiny: tuple[Path, dict[str, str]]) -> None:
    directory, expected = tiny
    path = directory / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ModelIntegrityError, match=r"model\.safetensors"):
        load_road_model(directory, device="cpu", expected=expected)


def test_the_segmenter_checks_for_files_only_when_called_and_needs_no_ml_extra(
    tmp_path: Path,
) -> None:
    segmenter = RoadSegmenter(tmp_path, device="cpu")  # no model files exist yet: must not raise
    with pytest.raises(ModelIntegrityError, match="missing"):
        segmenter(_frame())


@pytest.mark.ml
def test_the_segmenter_predicts_once_files_exist(tiny: tuple[Path, dict[str, str]]) -> None:
    directory, expected = tiny
    segmenter = RoadSegmenter(directory, device="cpu", expected=expected)
    mask = segmenter(_frame(70, 50))
    assert mask.shape == (70, 50) and mask.dtype == np.bool_


@pytest.mark.ml
def test_the_input_size_constants_are_width_then_height() -> None:
    assert (INPUT_WIDTH, INPUT_HEIGHT) == (512, 384)


# ---------------------------------------------------------------- stub models


def _stub(logits_fn: Any) -> Any:
    """A parameter-free module whose output is logits_fn(input) with shape (n, 5, h, w)."""
    pytest.importorskip("torch")
    import torch

    class Stub(torch.nn.Module):  # type: ignore[misc, unused-ignore]
        def forward(self, x: Any) -> Any:
            return logits_fn(x, torch)

    return Stub()


def _constant_class(cls: int) -> Any:
    def logits(x: Any, torch: Any) -> Any:
        out = torch.zeros(x.shape[0], 5, x.shape[2], x.shape[3])
        out[:, cls] = 1.0
        return out

    return _stub(logits)


@pytest.mark.ml
@pytest.mark.parametrize(
    ("cls", "is_road"), [(0, True), (1, True), (2, False), (3, False), (4, False)]
)
def test_each_class_id_maps_to_road_or_not_road(cls: int, is_road: bool) -> None:
    mask = predict_road_mask(_constant_class(cls), _frame(90, 130), device="cpu")
    assert mask.shape == (90, 130)
    assert bool(mask.all()) is is_road and bool(mask.any()) is is_road


@pytest.mark.ml
def test_a_road_on_the_left_half_stays_on_the_left_of_the_frame() -> None:
    def logits(x: Any, torch: Any) -> Any:
        out = torch.zeros(x.shape[0], 5, x.shape[2], x.shape[3])
        out[:, 3] = 1.0
        out[:, 0, :, : x.shape[3] // 2] = 2.0  # left half of the model input is road
        return out

    mask = predict_road_mask(_stub(logits), _frame(60, 100), device="cpu")
    assert mask[:, :50].all() and not mask[:, 50:].any()


@pytest.mark.ml
def test_a_road_in_the_top_half_stays_at_the_top_of_the_frame() -> None:
    def logits(x: Any, torch: Any) -> Any:
        out = torch.zeros(x.shape[0], 5, x.shape[2], x.shape[3])
        out[:, 3] = 1.0
        out[:, 0, : x.shape[2] // 2, :] = 2.0  # top half of the model input is road
        return out

    mask = predict_road_mask(_stub(logits), _frame(60, 100), device="cpu")
    assert mask[:30].all() and not mask[30:].any()


@pytest.mark.ml
def test_the_model_sees_raw_0_to_255_pixels_not_normalised_ones() -> None:
    def logits(x: Any, torch: Any) -> Any:
        out = torch.full((x.shape[0], 5, x.shape[2], x.shape[3]), -1e9)
        out[:, 3] = 0.0
        out[:, 0] = x.mean(dim=1) - 100.0  # road only where the raw mean pixel exceeds 100
        return out

    model = _stub(logits)
    white = np.full((40, 60, 3), 255, dtype=np.uint8)
    black = np.zeros((40, 60, 3), dtype=np.uint8)
    assert predict_road_mask(model, white, device="cpu").all()
    assert not predict_road_mask(model, black, device="cpu").any()


@pytest.mark.ml
def test_predicting_without_a_device_uses_the_device_the_model_is_on(
    tiny: tuple[Path, dict[str, str]],
) -> None:
    directory, expected = tiny
    model = load_road_model(directory, device="cpu", expected=expected)
    mask = predict_road_mask(model, _frame(60, 80))  # no device given: must follow the model
    assert mask.shape == (60, 80)


# ---------------------------------------------------------------- real weights


@pytest.mark.ml
def test_the_pinned_real_model_predicts_a_mask() -> None:
    folder = os.environ.get("ROAD_MODEL_DIR")
    if not folder:
        pytest.skip("set ROAD_MODEL_DIR to the folder with the real model files")
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("safetensors")
    model = load_road_model(Path(folder), device="cpu")
    mask = predict_road_mask(model, _frame(300, 520), device="cpu")
    assert mask.shape == (300, 520) and mask.dtype == np.bool_
