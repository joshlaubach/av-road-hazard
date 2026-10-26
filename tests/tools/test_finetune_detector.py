"""tools/finetune_detector.py: output guard, data loading, learning-rate schedule and a tiny run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from perception.finetune import FINETUNE_CLASSES, TrainItem, class_index, swap_head
from tools.finetune_detector import (
    REPO,
    DetectionDataset,
    check_outside_repo,
    class_box_counts,
    learning_rate_factor,
    load_config,
    save_model,
    train,
)

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torchvision") is None, reason="torch and torchvision not installed"
)


def _image(folder: Path, name: str, size: tuple[int, int] = (96, 64)) -> str:
    path = folder / name
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    ).save(path)
    return str(path)


def test_the_output_folder_must_be_outside_the_repository(tmp_path: Path) -> None:
    assert check_outside_repo(tmp_path) == tmp_path.resolve()
    with pytest.raises(ValueError, match="outside the repository"):
        check_outside_repo(REPO / "data" / "weights")
    with pytest.raises(ValueError):
        check_outside_repo(REPO)


def test_the_shipped_config_names_the_eight_classes_and_the_adr_limits() -> None:
    cfg = load_config()
    assert cfg["model"] == "fasterrcnn_mobilenet_v3_large_fpn" and len(cfg["cameras"]) == 6
    limits = cfg["evaluation"]["limits"]
    assert (limits["min_precision"], limits["min_recall"]) == (0.70, 0.40)
    assert limits["min_passing_scenes"] == {"barrier": 2, "traffic_cone": 1}
    assert cfg["evaluation"]["new_classes"] == ["barrier", "traffic_cone"]


def test_the_learning_rate_warms_up_then_follows_a_cosine_to_zero() -> None:
    assert learning_rate_factor(0, 10, 100) == pytest.approx(0.1)
    assert learning_rate_factor(9, 10, 100) == pytest.approx(1.0)
    assert learning_rate_factor(10, 10, 100) == pytest.approx(1.0)
    assert learning_rate_factor(55, 10, 100) == pytest.approx(0.5, abs=0.01)
    assert learning_rate_factor(100, 10, 100) == pytest.approx(0.0, abs=1e-9)
    assert learning_rate_factor(500, 10, 100) == pytest.approx(0.0, abs=1e-9)  # never negative


@needs_torch
def test_the_dataset_returns_a_float_image_and_matching_targets(tmp_path: Path) -> None:
    item = TrainItem(_image(tmp_path, "a.png"), ((10.0, 10.0, 40.0, 50.0),), (class_index("car"),))
    image, target = DetectionDataset([item], flip=0.0, jitter=0.0)[0]
    assert (
        tuple(image.shape) == (3, 64, 96) and 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    )
    assert target["boxes"].tolist() == [[10.0, 10.0, 40.0, 50.0]]
    assert target["labels"].tolist() == [class_index("car")]


@needs_torch
def test_a_flip_mirrors_the_boxes_with_the_image(tmp_path: Path) -> None:
    path = _image(tmp_path, "a.png")
    item = TrainItem(path, ((10.0, 10.0, 40.0, 50.0),), (class_index("car"),))
    flipped, target = DetectionDataset([item], flip=1.0, jitter=0.0)[0]
    plain, _ = DetectionDataset([item], flip=0.0, jitter=0.0)[0]
    assert target["boxes"].tolist() == [[56.0, 10.0, 86.0, 50.0]]
    assert flipped.flip(-1).sub(plain).abs().max() < 1e-5


@needs_torch
def test_an_image_with_no_boxes_gives_an_empty_target(tmp_path: Path) -> None:
    item = TrainItem(_image(tmp_path, "a.png"), (), ())
    _, target = DetectionDataset([item], flip=0.0, jitter=0.0)[0]
    assert tuple(target["boxes"].shape) == (0, 4) and tuple(target["labels"].shape) == (0,)


def test_box_counts_are_per_class() -> None:
    items = [
        TrainItem("a", (), ()),
        TrainItem(
            "b", ((0.0, 0.0, 1.0, 1.0),) * 3, (class_index("car"),) * 2 + (class_index("barrier"),)
        ),
    ]
    counts = class_box_counts(items)
    assert counts["car"] == 2 and counts["barrier"] == 1 and counts["bus"] == 0
    assert set(counts) == set(FINETUNE_CLASSES)


def _tiny_model() -> Any:
    from perception.detector import build_model

    model = build_model("fasterrcnn_mobilenet_v3_large_fpn")
    model.transform.min_size = (64,)
    model.transform.max_size = 96
    return swap_head(model, seed=0)


@needs_torch
def test_a_short_training_run_gives_finite_losses_and_changes_the_weights(tmp_path: Path) -> None:
    items = [
        TrainItem(
            _image(tmp_path, f"{i}.png"),
            ((10.0, 10.0, 50.0, 50.0),),
            (class_index("traffic_cone"),),
        )
        for i in range(4)
    ]
    cfg = {
        "seed": 1, "epochs": 2, "batch_size": 2, "lr": 0.01, "momentum": 0.9,
        "weight_decay": 0.0, "warmup_iters": 1, "grad_clip": 10.0, "num_workers": 0,
        "flip": 0.5, "jitter": 0.1,
    }  # fmt: skip
    model = _tiny_model()
    before = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    losses = train(model, items, cfg, device="cpu", epochs=2, progress=False)
    assert len(losses) == 2 and all(np.isfinite(losses))
    assert not before.equal(model.roi_heads.box_predictor.cls_score.weight.detach())


@needs_torch
def test_a_saved_model_comes_with_its_hash_and_notes(tmp_path: Path) -> None:
    save_model(_tiny_model(), tmp_path / "fold", {"held_out_scene": "scene-1"})
    meta = json.loads((tmp_path / "fold" / "meta.json").read_text())
    assert meta["held_out_scene"] == "scene-1" and len(meta["sha256"]) == 64
    assert (tmp_path / "fold" / "model.safetensors").is_file()
