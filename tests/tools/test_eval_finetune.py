"""tools/eval_finetune.py: loading a fold model safely, predicting with fusion labels, grouping."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval.detector import GtBox
from eval.models import Detection
from perception.detector import build_model
from perception.finetune import swap_head
from tools.eval_finetune import group_by_scene, load_finetuned, predict
from tools.finetune_detector import SceneFrame, save_model

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torchvision") is None, reason="torch and torchvision not installed"
)

META = {"config": {"model": "fasterrcnn_mobilenet_v3_large_fpn"}}


def _save(folder: Path) -> None:
    save_model(swap_head(build_model(META["config"]["model"]), seed=0), folder, META)


@needs_torch
def test_a_saved_fold_model_loads_and_predicts_with_fusion_labels(tmp_path: Path) -> None:
    _save(tmp_path / "scene-1")
    model = load_finetuned(tmp_path / "scene-1", "cpu")
    frame = np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    found = predict(model, frame, "cpu", finetuned=True)
    allowed = {
        "pedestrian",
        "cyclist",
        "motorcycle",
        "car",
        "truck",
        "bus",
        "barrier",
        "traffic_cone",
    }
    assert all(d.label in allowed and 0.05 <= d.score <= 1.0 for d in found)


@needs_torch
def test_a_model_file_that_changed_is_refused(tmp_path: Path) -> None:
    _save(tmp_path / "scene-1")
    weights = tmp_path / "scene-1" / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match"):
        load_finetuned(tmp_path / "scene-1", "cpu")


@needs_torch
def test_the_meta_records_the_hash_of_the_saved_file(tmp_path: Path) -> None:
    _save(tmp_path / "scene-1")
    meta = json.loads((tmp_path / "scene-1" / "meta.json").read_text())
    assert len(meta["sha256"]) == 64


def _frame(scene: str, camera: str, truth: list[GtBox]) -> SceneFrame:
    return SceneFrame(scene, False, camera, Path(f"{scene}-{camera}.jpg"), truth)


def test_grouping_puts_frames_in_their_scene_and_maps_ground_truth_labels() -> None:
    box = (0.0, 0.0, 50.0, 50.0)
    frames = [
        _frame("a", "CAM_FRONT", [GtBox("construction_worker", box, 4), GtBox("debris", box, 4)]),
        _frame("a", "CAM_BACK", []),
        _frame("b", "CAM_FRONT", [GtBox("barrier", box, 4)]),
    ]
    found: list[list[Detection]] = [[], [], []]
    grouped = group_by_scene(frames, found)
    assert set(grouped) == {"a", "b"} and len(grouped["a"]) == 2
    assert [g.label for g in grouped["a"][0].ground_truth] == ["pedestrian"]  # debris dropped
    assert [g.label for g in grouped["b"][0].ground_truth] == ["barrier"]


def test_grouping_can_keep_one_camera_only() -> None:
    frames = [_frame("a", "CAM_FRONT", []), _frame("a", "CAM_BACK", [])]
    assert len(group_by_scene(frames, [[], []], camera="CAM_FRONT")["a"]) == 1


def test_detections_stay_aligned_with_their_frames() -> None:
    frames = [_frame("a", "CAM_FRONT", []), _frame("a", "CAM_BACK", [])]
    first = [Detection("car", 0.9, (0.0, 0.0, 5.0, 5.0))]
    grouped: Any = group_by_scene(frames, [first, []])
    assert grouped["a"][0].detections == first and grouped["a"][1].detections == []
