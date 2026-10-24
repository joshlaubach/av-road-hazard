"""Road mask from a segmentation model: the comma10k U-Net (ADR 0013).

Purpose: the tile classifier fires on grass, snow banks and scenery, so tiles are kept only when
enough of them lies on the road (defects/road_region.py). This module holds the pinned model
description and, below it, the code that turns a frame into a boolean road mask.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from defects.runtime import select_device
from fusion.fuse import RoadMask

# The model is pinned to one Hugging Face revision and to the SHA-256 of each file, so a changed or
# tampered download is rejected before any weights are loaded. Only the two files the code needs
# are listed. Weights are never committed; tools/fetch_road_model.py downloads them.
MODEL_REPO = "commaai/comma10k-segnet"
MODEL_REVISION = "b642c614eb0737fe7befa229f47fbd0245033038"
MODEL_FILES: dict[str, str] = {
    "config.json": "2b8f16dbad9bd85386609386a9cb5dedc6e0c518253a9af484e0a128d9463c88",
    "model.safetensors": "8208672861ad1b111dc98f3a7c54196d29875b709c7353e2dd1b7614343fb3a8",
}
MODEL_SAFETENSORS_BYTES = 38_502_740
# Folder under the checkpoint directory (paths.checkpoint_dir) that holds the model files.
MODEL_SUBDIR = "road_mask/comma10k_unet"

# The model's shipped eval transform is a bilinear resize to 512 x 384 (width x height) and nothing
# else: no division by 255 and no ImageNet normalisation (normalising lowered road-vs-not-road
# AUC from 0.98 to 0.81 on 148 annotated tiles).
INPUT_WIDTH = 512
INPUT_HEIGHT = 384
# comma10k classes: 0 road, 1 lane markings, 2 undrivable, 3 movable, 4 own car. Road is 0 and 1.
ROAD_CLASSES = (0, 1)

# verify_model_files hashes in chunks so a large file is never held in memory just to be checked.
_HASH_CHUNK_BYTES = 1024 * 1024
# load_road_model needs both of these, and each must come with an expected hash.
_REQUIRED_FILES = ("config.json", "model.safetensors")


class ModelIntegrityError(ValueError):
    """The model files are missing, changed, or not the model this code expects."""


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_files(weights_dir: Path, expected: Mapping[str, str] = MODEL_FILES) -> None:
    """Raise ModelIntegrityError unless every expected file exists with the expected SHA-256."""
    for name, want in expected.items():
        path = weights_dir / name
        if not path.is_file():
            raise ModelIntegrityError(f"{name}: expected sha256 {want}, got missing")
        got = _sha256_of(path)
        if got != want:
            raise ModelIntegrityError(f"{name}: expected sha256 {want}, got {got}")


def _read_verified(path: Path, name: str, want: str) -> bytes:
    """The file's bytes, read once and checked against want, so parsing sees what was hashed."""
    if not path.is_file():
        raise ModelIntegrityError(f"{name}: expected sha256 {want}, got missing")
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ModelIntegrityError(f"{name}: expected sha256 {want}, got {got}")
    return data


def _build_unet(config_bytes: bytes) -> Any:
    import segmentation_models_pytorch as smp

    config = json.loads(config_bytes)
    model_class = config.pop("_model_class", None)
    if model_class != "Unet":
        raise ModelIntegrityError(f"config.json: _model_class must be 'Unet', got {model_class!r}")
    # Building the network must never download anything: a pretrained encoder would fetch weights.
    if config.get("encoder_weights") is not None:
        raise ModelIntegrityError("config.json: encoder_weights must be null (no downloads)")
    return smp.Unet(**config)


def load_road_model(
    weights_dir: Path,
    *,
    device: str | None = None,
    expected: Mapping[str, str] = MODEL_FILES,
) -> Any:
    """The pinned U-Net in eval mode on device (default: the best available).

    Both files must come with an expected hash. Each is read once, hashed, and only then parsed, so
    nothing is interpreted before it is verified and the bytes cannot change in between. Weights
    come from safetensors, which holds plain tensors and cannot run code (unlike torch.load).
    """
    for name in _REQUIRED_FILES:
        if name not in expected:
            raise ModelIntegrityError(
                f"{name}: no expected sha256 given, refusing to load it unverified"
            )
    config_bytes = _read_verified(
        weights_dir / "config.json", "config.json", expected["config.json"]
    )
    weights_bytes = _read_verified(
        weights_dir / "model.safetensors", "model.safetensors", expected["model.safetensors"]
    )
    verify_model_files(weights_dir, {k: v for k, v in expected.items() if k not in _REQUIRED_FILES})
    from safetensors.torch import load as load_safetensors

    model = _build_unet(config_bytes)
    model.load_state_dict(load_safetensors(weights_bytes))
    return model.eval().to(device or select_device())


def road_mask_from_class_map(class_map: np.ndarray, height: int, width: int) -> RoadMask:
    """Nearest-neighbour upsample a (h, w) class map to (height, width); True on road classes."""
    if class_map.ndim != 2:
        raise ValueError(f"class map must be 2-D (h, w), got shape {class_map.shape}")
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height} and {width}")
    # Each output pixel takes the model cell under its centre, so mask edges are not shifted by up
    # to half a cell (about 4 px on a 1600 x 900 frame). Integer maths keeps this exact.
    rows = (2 * np.arange(height) + 1) * class_map.shape[0] // (2 * height)
    cols = (2 * np.arange(width) + 1) * class_map.shape[1] // (2 * width)
    return np.isin(class_map[np.ix_(rows, cols)], ROAD_CLASSES)


def _check_frame(frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or 0 in frame.shape:
        raise ValueError(
            f"frame must be a non-empty RGB uint8 array (H, W, 3), got shape {frame.shape} "
            f"and dtype {frame.dtype}"
        )


def _class_map(model: Any, frame: np.ndarray, device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    # Raw 0-255 values go straight into the bilinear resize, exactly as the model's shipped eval
    # transform does. Dividing by 255 or applying ImageNet normalisation dropped road-vs-not-road
    # AUC from 0.98 to 0.81, so neither is done here.
    image = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    image = functional.interpolate(
        image,
        size=(INPUT_HEIGHT, INPUT_WIDTH),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    with torch.inference_mode():
        logits = model(image.to(device))
    return np.asarray(logits.argmax(dim=1)[0].cpu().numpy())


def _model_device(model: Any) -> str:
    """Where the model's weights are; the best available device for a model with no parameters."""
    for parameter in model.parameters():
        return str(parameter.device)
    return select_device()


def predict_road_mask(model: Any, frame: np.ndarray, *, device: str | None = None) -> RoadMask:
    """Bool mask (H, W), True on road, for an RGB uint8 frame; same size as the frame.

    The input goes to `device`, which defaults to the device the model is already on.
    """
    _check_frame(frame)
    class_map = _class_map(model, frame, device or _model_device(model))
    return road_mask_from_class_map(class_map, frame.shape[0], frame.shape[1])


class RoadSegmenter:
    """Frame in, road mask out. The model loads on the first call, not at construction."""

    def __init__(
        self,
        weights_dir: Path,
        *,
        device: str | None = None,
        expected: Mapping[str, str] = MODEL_FILES,
    ) -> None:
        self._weights_dir = weights_dir
        self._device = device
        self._expected = expected
        self._model: Any = None

    def __call__(self, frame: np.ndarray) -> RoadMask:
        if self._model is None:
            self._model = load_road_model(
                self._weights_dir, device=self._device, expected=self._expected
            )
        return predict_road_mask(self._model, frame, device=self._device)
