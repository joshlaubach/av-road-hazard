"""Fine-tuning the COCO detector on nuScenes for traffic cones and barriers (ADR 0015).

Pure pieces that need no GPU or dataset: the 8-class label set, turning ground-truth boxes into
training items, the leave-one-scene-out folds, the head swap that keeps the six COCO rows the
detector already knows, and the augmentations. tools/finetune_detector.py does the training.

Class index 0 is background, 1..8 follow FINETUNE_CLASSES. Construction workers, children and the
like are trained as "pedestrian" and construction vehicles as "truck" (the families in
perception/labels.py).
Debris is dropped: 13 examples are too few to learn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from perception.labels import COCO_TO_LABEL, to_vocabulary

FINETUNE_CLASSES = (
    "pedestrian",
    "cyclist",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "barrier",
    "traffic_cone",
)
NEW_CLASSES = ("barrier", "traffic_cone")
FINETUNE_VOCABULARY = frozenset(FINETUNE_CLASSES)
LABEL_TO_COCO_ID = {label: coco_id for coco_id, label in COCO_TO_LABEL.items()}

# A training box must be at least this visible (nuScenes visibility level 2 = 40-60% of the object)
# and at least 16 x 16 px. Smaller or more hidden boxes are neither positives nor negatives for
# the evaluation either (eval/detector.py), so the model is not rewarded or punished for them.
MIN_VISIBILITY = 2
MIN_TRAIN_AREA_PX = 16 * 16

Box = tuple[float, float, float, float]


def class_index(label: str) -> int:
    """1-based class index of a fine-tune label (0 is background)."""
    return FINETUNE_CLASSES.index(label) + 1


def class_name(index: int) -> str:
    """The label of a 1-based class index; background (0) and out-of-range indices have none."""
    if not 1 <= index <= len(FINETUNE_CLASSES):
        raise ValueError(f"no class has index {index}")
    return FINETUNE_CLASSES[index - 1]


def finetune_label(ground_truth_label: str) -> str | None:
    """The fine-tune class a ground-truth fusion label is trained as, or None to leave it out."""
    return to_vocabulary(ground_truth_label, FINETUNE_VOCABULARY)


@dataclass(frozen=True)
class TrainItem:
    """One training image: its path, its boxes and their class indices (1..8)."""

    path: str
    boxes: tuple[Box, ...]
    labels: tuple[int, ...]


def training_item(path: str, ground_truth: Sequence[tuple[str, Box, int]]) -> TrainItem:
    """Training boxes for one image from (label, box, visibility) ground truth.

    Keeps boxes that are visible enough and large enough, whose label has a fine-tune class.
    """
    boxes: list[Box] = []
    labels: list[int] = []
    for label, box, visibility in ground_truth:
        target = finetune_label(label)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if target is None or visibility < MIN_VISIBILITY or area < MIN_TRAIN_AREA_PX:
            continue
        boxes.append(box)
        labels.append(class_index(target))
    return TrainItem(path, tuple(boxes), tuple(labels))


def scene_folds(scenes: Sequence[str]) -> list[tuple[list[str], str]]:
    """Leave-one-scene-out: (training scenes, held-out scene), one fold per scene, in sorted order.

    Scenes are the unit, never frames: neighboring frames are near copies, so splitting by frame
    would leak the held-out scene into training.
    """
    ordered = sorted(set(scenes))
    if len(ordered) < 2:
        raise ValueError("need at least two scenes to leave one out")
    return [([s for s in ordered if s != held_out], held_out) for held_out in ordered]


def deal_into_folds(ranked_scenes: Sequence[str], k: int) -> dict[str, int]:
    """Deal scenes to k folds in turn: the scene at rank i goes to fold i % k.

    The caller ranks the scenes by a rule fixed before any result is seen, so the scenes with the
    most barriers and cones end up in different folds instead of all in one.
    """
    if k < 2:
        raise ValueError("need at least two folds")
    if len(set(ranked_scenes)) != len(ranked_scenes):
        raise ValueError("a scene is listed twice")
    if len(ranked_scenes) < k:
        raise ValueError(f"{len(ranked_scenes)} scenes cannot fill {k} folds")
    return {scene: rank % k for rank, scene in enumerate(ranked_scenes)}


def grouped_folds(plan: Mapping[str, int]) -> list[tuple[list[str], list[str]]]:
    """Grouped cross-validation: (training scenes, held-out scenes) per fold, in fold order.

    plan maps each scene to a fold number; the numbers must be 0..k-1 with k >= 2. Every scene is
    held out once, by a model that never trained on it.
    """
    numbers = sorted(set(plan.values()))
    if len(numbers) < 2 or numbers != list(range(len(numbers))):
        raise ValueError(f"fold numbers must be 0..k-1 with k >= 2, got {numbers}")
    return [
        (sorted(s for s, f in plan.items() if f != n), sorted(s for s, f in plan.items() if f == n))
        for n in numbers
    ]


def swap_head(model: Any, *, seed: int = 0) -> Any:
    """Replace the 91-class COCO box head with the 8-class one, keeping the six rows COCO knew.

    Background and the six old classes start from their COCO rows (class scores and box
    regressors), so the model begins by knowing what it knew. The two new classes start from the
    usual small random values. Changes the model in place and returns it.
    """
    import torch
    from torch import nn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    old = model.roi_heads.box_predictor
    in_features = old.cls_score.in_features
    count = len(FINETUNE_CLASSES) + 1
    generator = torch.Generator().manual_seed(seed)
    new = FastRCNNPredictor(in_features, count)
    with torch.no_grad():
        nn.init.normal_(new.cls_score.weight, std=0.01, generator=generator)
        nn.init.zeros_(new.cls_score.bias)
        nn.init.normal_(new.bbox_pred.weight, std=0.001, generator=generator)
        nn.init.zeros_(new.bbox_pred.bias)
        # (row in the new head, row in the COCO head): background, then each class COCO knew.
        copies = [(0, 0)] + [
            (index, LABEL_TO_COCO_ID[label])
            for index, label in enumerate(FINETUNE_CLASSES, start=1)
            if label in LABEL_TO_COCO_ID
        ]
        for new_index, old_index in copies:
            new.cls_score.weight[new_index] = old.cls_score.weight[old_index]
            new.cls_score.bias[new_index] = old.cls_score.bias[old_index]
            new_rows = slice(4 * new_index, 4 * new_index + 4)
            old_rows = slice(4 * old_index, 4 * old_index + 4)
            new.bbox_pred.weight[new_rows] = old.bbox_pred.weight[old_rows]
            new.bbox_pred.bias[new_rows] = old.bbox_pred.bias[old_rows]
    new.to(old.cls_score.weight.device)
    model.roi_heads.box_predictor = new
    return model


def freeze_batchnorm_statistics(model: Any) -> None:
    """Keep every BatchNorm layer in eval mode, so small batches cannot disturb its statistics.

    Call after model.train(), which would switch them back.
    """
    from torch import nn

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def flip_boxes(boxes: Sequence[Box], width: float) -> list[Box]:
    """The boxes after a left-right flip of an image of the given width."""
    return [(width - x2, y1, width - x1, y2) for x1, y1, x2, y2 in boxes]
