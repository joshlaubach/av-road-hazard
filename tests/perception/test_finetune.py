"""perception/finetune.py: the 8-class set, training items, scene folds, head swap, augmentation."""

from __future__ import annotations

from typing import Any

import pytest

from perception.finetune import (
    FINETUNE_CLASSES,
    LABEL_TO_COCO_ID,
    NEW_CLASSES,
    class_index,
    class_name,
    finetune_label,
    flip_boxes,
    freeze_batchnorm_statistics,
    scene_folds,
    swap_head,
    training_item,
)
from perception.labels import COCO_VOCABULARY, FUSION_LABELS

BOX = (10.0, 10.0, 60.0, 80.0)


def test_the_class_set_is_the_six_coco_labels_plus_the_two_new_ones() -> None:
    assert set(FINETUNE_CLASSES[:6]) == COCO_VOCABULARY
    assert FINETUNE_CLASSES[6:] == NEW_CLASSES == ("barrier", "traffic_cone")
    assert set(FINETUNE_CLASSES) <= set(FUSION_LABELS)


def test_class_indices_start_at_one_because_zero_is_background() -> None:
    assert [class_index(c) for c in FINETUNE_CLASSES] == list(range(1, 9))
    assert all(class_name(class_index(c)) == c for c in FINETUNE_CLASSES)


def test_finer_labels_train_as_their_family_and_debris_is_left_out() -> None:
    assert finetune_label("construction_worker") == "pedestrian"
    assert finetune_label("child") == "pedestrian"
    assert finetune_label("construction_vehicle") == "truck"
    assert finetune_label("barrier") == "barrier"
    assert finetune_label("traffic_cone") == "traffic_cone"
    assert finetune_label("debris") is None


def test_a_training_item_keeps_visible_big_enough_boxes_with_a_class() -> None:
    item = training_item(
        "a.jpg",
        [
            ("car", BOX, 4),
            ("traffic_cone", BOX, 2),
            ("barrier", (0, 0, 10, 10), 4),  # 100 px: too small
            ("pedestrian", BOX, 1),  # under 40% visible
            ("debris", BOX, 4),  # no class
        ],
    )
    assert item.path == "a.jpg"
    assert item.labels == (class_index("car"), class_index("traffic_cone"))
    assert item.boxes == (BOX, BOX)


def test_an_image_with_nothing_to_learn_gives_an_empty_item() -> None:
    item = training_item("b.jpg", [])
    assert item.boxes == () and item.labels == ()


def test_folds_hold_out_each_scene_once_and_never_train_on_it() -> None:
    scenes = ["s3", "s1", "s2", "s4"]
    folds = scene_folds(scenes)
    assert [held for _, held in folds] == ["s1", "s2", "s3", "s4"]
    for train, held in folds:
        assert held not in train and sorted([*train, held]) == ["s1", "s2", "s3", "s4"]


def test_folds_need_at_least_two_scenes() -> None:
    with pytest.raises(ValueError):
        scene_folds(["only"])


def test_flipping_boxes_mirrors_them_inside_the_image() -> None:
    assert flip_boxes([(10.0, 5.0, 40.0, 20.0)], 100.0) == [(60.0, 5.0, 90.0, 20.0)]
    assert flip_boxes(flip_boxes([BOX], 200.0), 200.0) == [BOX]


# ---------------------------------------------------------------------------- head swap


def _small_model() -> Any:
    from perception.detector import build_model

    model = build_model("fasterrcnn_mobilenet_v3_large_fpn")
    model.transform.min_size = (64,)
    model.transform.max_size = 96
    return model


def test_the_swapped_head_has_nine_outputs_and_keeps_the_coco_rows_of_the_old_classes() -> None:
    import torch

    model = _small_model()
    old = model.roi_heads.box_predictor
    old_cls, old_box = old.cls_score.weight.clone(), old.bbox_pred.weight.clone()
    swap_head(model, seed=1)
    head = model.roi_heads.box_predictor
    assert head.cls_score.out_features == 9 and head.bbox_pred.out_features == 36
    assert torch.equal(head.cls_score.weight[0], old_cls[0])  # background
    for label, coco_id in LABEL_TO_COCO_ID.items():
        row = class_index(label)
        assert torch.equal(head.cls_score.weight[row], old_cls[coco_id]), label
        assert torch.equal(
            head.bbox_pred.weight[4 * row : 4 * row + 4], old_box[4 * coco_id : 4 * coco_id + 4]
        ), label


def test_the_two_new_classes_start_small_and_random_not_copied() -> None:
    model = _small_model()
    old_rows = model.roi_heads.box_predictor.cls_score.weight.clone()
    swap_head(model, seed=1)
    weights = model.roi_heads.box_predictor.cls_score.weight
    for label in NEW_CLASSES:
        row = weights[class_index(label)]
        assert row.abs().max() < 0.1
        assert not any(row.equal(old_rows[i]) for i in range(old_rows.shape[0]))


def test_the_same_seed_gives_the_same_new_rows() -> None:
    import torch

    first, second = _small_model(), _small_model()
    swap_head(first, seed=3)
    swap_head(second, seed=3)
    a = first.roi_heads.box_predictor.cls_score.weight[class_index("barrier")]
    b = second.roi_heads.box_predictor.cls_score.weight[class_index("barrier")]
    assert torch.equal(a, b)


def test_a_swapped_model_still_runs_and_predicts_with_eight_classes() -> None:
    import torch

    model = swap_head(_small_model(), seed=0).eval()
    with torch.inference_mode():
        (out,) = model([torch.rand(3, 64, 80)])
    assert set(out) == {"boxes", "labels", "scores"}
    assert all(0 < int(v) <= 8 for v in out["labels"])


def test_freezing_batchnorm_keeps_it_in_eval_mode_after_train() -> None:
    from torch import nn

    model = _small_model()
    model.train()
    freeze_batchnorm_statistics(model)
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert norms and not any(m.training for m in norms)
    assert model.training  # the rest of the model is still training
