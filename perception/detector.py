"""COCO object detector: pinned torchvision weights, hash-checked loading, and prediction.

Zero-shot: torchvision's COCO-trained Faster R-CNN models are used as published, with no
fine-tuning (ADR 0014). Two candidates are pinned, a heavy and a light one, so phase 3 can measure
accuracy against CPU speed. Weights are never committed and never downloaded implicitly: the
published .pth files are fetched by tools/fetch_detector.py, which checks their SHA-256, loads them
with torch's weights-only loader (plain tensors, no code) and saves them as safetensors. This
module loads only that safetensors file, and only after hashing it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.device import select_device
from fusion.schema import Object
from perception.labels import coco_label

# Folder under the checkpoint directory (paths.checkpoint_dir) that holds the converted weights.
DETECTOR_SUBDIR = "detector"


class DetectorUnavailable(RuntimeError):
    """The detector weights are missing, changed, or cannot be loaded."""


@dataclass(frozen=True)
class DetectorSpec:
    """One pinned model: where the published weights come from and the hash of each form.

    source_sha256 is the published .pth file; sha256 is the safetensors file the fetch tool makes
    from it (the one this module loads).
    """

    name: str
    url: str
    source_sha256: str
    sha256: str

    @property
    def filename(self) -> str:
        return f"{self.name}.safetensors"


DETECTORS: dict[str, DetectorSpec] = {
    spec.name: spec
    for spec in (
        DetectorSpec(
            "fasterrcnn_resnet50_fpn_v2",
            "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
            "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf",
            "2a67bf22150dd96d91775f895b561f186b12b6d655bad795cc115cd002a63e4e",
        ),
        DetectorSpec(
            "fasterrcnn_mobilenet_v3_large_fpn",
            "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
            "fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533",
            "f7731b02ff6365453ad4fb85c008f4dc86dea63a2d4d77377f986fa54b797e21",
        ),
    )
}


def build_model(name: str) -> Any:
    """The untrained network for a spec name: no pretrained weights, so nothing is downloaded."""
    from torchvision.models import detection

    builders = {
        "fasterrcnn_resnet50_fpn_v2": detection.fasterrcnn_resnet50_fpn_v2,
        "fasterrcnn_mobilenet_v3_large_fpn": detection.fasterrcnn_mobilenet_v3_large_fpn,
    }
    if name not in builders:
        raise DetectorUnavailable(f"unknown detector {name!r}; choose one of {sorted(builders)}")
    return builders[name](weights=None, weights_backbone=None)


def load_detector_model(
    path: Path,
    spec: DetectorSpec,
    *,
    device: str | None = None,
    factory: Callable[[str], Any] = build_model,
) -> Any:
    """The pinned detector in eval mode on device (default: the best available).

    The file is read once and hashed before anything parses it, so the bytes that were checked
    are the bytes that are loaded. safetensors holds plain tensors and cannot run code.
    """
    if not path.is_file():
        raise DetectorUnavailable(
            f"{path} is missing; run: python -m tools.fetch_detector --model {spec.name}"
        )
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != spec.sha256:
        raise DetectorUnavailable(f"{path.name}: expected sha256 {spec.sha256}, got {got}")
    from safetensors.torch import load as load_safetensors

    model = factory(spec.name)
    model.load_state_dict(load_safetensors(data))
    return model.eval().to(device or select_device())


def _model_device(model: Any) -> str:
    for parameter in model.parameters():
        return str(parameter.device)
    return select_device()


def detect_objects(
    model: Any,
    frame: np.ndarray,
    *,
    score_threshold: float,
    device: str | None = None,
    labeler: Callable[[int], str | None] = coco_label,
    cutoffs: Mapping[str, float] | None = None,
) -> list[Object]:
    """Objects in an RGB uint8 frame (H, W, 3), with fusion labels and boxes in frame pixels.

    Detections below score_threshold, and classes fusion does not weight (see
    perception/labels.py), are dropped; a score equal to the threshold is kept. labeler turns a
    class index into a fusion label (the 8-class model passes its own), and cutoffs gives a class
    its own threshold in place of score_threshold. Boxes are clamped to the frame, and any that
    end up empty are dropped.
    """
    import torch

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or 0 in frame.shape:
        raise ValueError(
            f"frame must be a non-empty RGB uint8 array (H, W, 3), got shape {frame.shape} "
            f"and dtype {frame.dtype}"
        )
    height, width = frame.shape[:2]
    # torchvision detection models take float images in 0-1, channels first, and do their own
    # resizing and normalisation.
    image = (
        torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
    )  # copy: may be read-only
    with torch.inference_mode():
        (output,) = model([image.to(device or _model_device(model))])
    objects: list[Object] = []
    for box, category, score in zip(
        output["boxes"].cpu().tolist(),
        output["labels"].cpu().tolist(),
        output["scores"].cpu().tolist(),
        strict=True,
    ):
        label = labeler(int(category))
        if label is None or score < (cutoffs or {}).get(label, score_threshold):
            continue
        x1, y1 = max(0.0, box[0]), max(0.0, box[1])
        x2, y2 = min(float(width), box[2]), min(float(height), box[3])
        if x2 > x1 and y2 > y1:
            objects.append(Object(label=label, score=float(score), box=(x1, y1, x2, y2)))
    return objects
