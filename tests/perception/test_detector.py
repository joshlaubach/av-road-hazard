"""Detector module: pinned weight specs, hash-checked loading, and turning outputs into Objects.

No real weights are needed: the model tests use a tiny stand-in that returns torchvision-style
output dicts, and the loader tests save a tiny state dict to tmp_path.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from perception.detector import (
    DETECTORS,
    DetectorSpec,
    DetectorUnavailable,
    detect_objects,
    load_detector_model,
)
from perception.finetune import class_name

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
safetensors_torch = pytest.importorskip("safetensors.torch")

HEX64 = re.compile(r"[0-9a-f]{64}")


def test_specs_pin_both_candidate_models() -> None:
    assert set(DETECTORS) == {"fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn"}


@pytest.mark.parametrize("spec", DETECTORS.values(), ids=list(DETECTORS))
def test_spec_hashes_are_well_formed_and_match_the_url(spec: DetectorSpec) -> None:
    assert HEX64.fullmatch(spec.source_sha256) and HEX64.fullmatch(spec.sha256)
    assert spec.url.startswith("https://download.pytorch.org/models/")
    # torchvision names each file after the first 8 hex digits of its own SHA-256.
    suffix = re.search(r"-([0-9a-f]{8})\.pth$", spec.url)
    assert suffix and spec.source_sha256.startswith(suffix.group(1))


# ------------------------------------------------------------------- loading


def Tiny() -> Any:
    return torch.nn.Sequential(torch.nn.Linear(2, 2))


def tiny_spec(path: Path, *, sha: str | None = None) -> DetectorSpec:
    digest = sha or hashlib.sha256(path.read_bytes()).hexdigest()
    return DetectorSpec(
        "tiny", "https://download.pytorch.org/models/x-00000000.pth", "0" * 64, digest
    )


def saved_tiny(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.safetensors"
    safetensors_torch.save_file(Tiny().state_dict(), str(path))
    return path


def test_loading_a_verified_file_gives_an_eval_model(tmp_path: Path) -> None:
    path = saved_tiny(tmp_path)
    model = load_detector_model(path, tiny_spec(path), device="cpu", factory=lambda name: Tiny())
    assert not model.training


def test_a_changed_file_is_refused_before_it_is_parsed(tmp_path: Path) -> None:
    path = saved_tiny(tmp_path)
    spec = tiny_spec(path)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(DetectorUnavailable, match="sha256"):
        load_detector_model(path, spec, device="cpu", factory=lambda name: Tiny())


def test_a_missing_file_says_to_fetch_it(tmp_path: Path) -> None:
    spec = tiny_spec(saved_tiny(tmp_path))
    with pytest.raises(DetectorUnavailable, match="fetch_detector"):
        load_detector_model(tmp_path / "gone.safetensors", spec, factory=lambda name: Tiny())


# ------------------------------------------------------------------ predicting


class FakeDetector:
    """Returns fixed torchvision-style output and remembers what it was given."""

    def __init__(self, boxes: list[list[float]], labels: list[int], scores: list[float]) -> None:
        self.out = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "scores": torch.tensor(scores, dtype=torch.float32),
        }
        self.seen: list[Any] = []

    def __call__(self, images: list[Any]) -> list[dict[str, Any]]:
        self.seen = images
        return [self.out]

    def parameters(self) -> Any:
        return iter([torch.zeros(1)])


FRAME = np.full((100, 200, 3), 255, dtype=np.uint8)  # (height, width, channels)


def test_detections_become_objects_with_fusion_labels() -> None:
    model = FakeDetector([[10, 20, 50, 80], [60, 10, 90, 40]], [1, 3], [0.9, 0.8])
    objects = detect_objects(model, FRAME, score_threshold=0.5)
    assert [(o.label, round(o.score, 2), o.box) for o in objects] == [
        ("pedestrian", 0.9, (10.0, 20.0, 50.0, 80.0)),
        ("car", 0.8, (60.0, 10.0, 90.0, 40.0)),
    ]


def test_model_gets_a_float_01_channel_first_tensor() -> None:
    model = FakeDetector([], [], [])
    detect_objects(model, FRAME, score_threshold=0.5)
    (image,) = model.seen
    assert image.shape == (3, 100, 200) and image.dtype == torch.float32
    assert float(image.max()) == 1.0 and float(image.min()) == 1.0


def test_low_scores_and_unmapped_classes_are_dropped() -> None:
    # 0.4 is under the threshold; class 10 is a traffic light, which fusion does not weight.
    model = FakeDetector([[0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5]], [1, 10, 8], [0.4, 0.99, 0.5])
    objects = detect_objects(model, FRAME, score_threshold=0.5)
    assert [o.label for o in objects] == ["truck"]  # a score equal to the threshold is kept


def test_boxes_are_clamped_to_the_frame_and_empty_ones_dropped() -> None:
    model = FakeDetector([[-5, -5, 300, 150], [250, 10, 260, 40]], [3, 3], [0.9, 0.9])
    objects = detect_objects(model, FRAME, score_threshold=0.5)
    assert [o.box for o in objects] == [(0.0, 0.0, 200.0, 100.0)]  # the second is off the frame


@pytest.mark.parametrize(
    "frame",
    [np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32)],
)
def test_frames_must_be_rgb_uint8(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="RGB uint8"):
        detect_objects(FakeDetector([], [], []), frame, score_threshold=0.5)


def test_a_labeler_and_per_class_cutoffs_decide_what_is_kept() -> None:
    # Class 7 is "barrier" and 8 "traffic_cone" in the 8-class model; barrier needs 0.55, cone 0.65.
    model = FakeDetector([[0, 0, 5, 5]] * 4, [7, 7, 8, 4], [0.52, 0.6, 0.6, 0.5])
    objects = detect_objects(
        model,
        FRAME,
        score_threshold=0.5,
        labeler=class_name,
        cutoffs={"barrier": 0.55, "traffic_cone": 0.65},
    )
    assert [o.label for o in objects] == ["barrier", "car"]
