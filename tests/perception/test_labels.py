"""The label maps that connect the detector's classes to the labels fusion weights by."""

from __future__ import annotations

import pytest

from fusion.config import load_fusion_config
from perception.labels import COCO_TO_LABEL, FUSION_LABELS, coco_label, nuscenes_label


def test_fusion_labels_are_the_ones_fusion_weights() -> None:
    assert set(FUSION_LABELS) == set(load_fusion_config().vulnerability_weights)


def test_every_mapped_label_is_a_fusion_label() -> None:
    assert set(COCO_TO_LABEL.values()) <= set(FUSION_LABELS)


@pytest.mark.parametrize(
    ("coco_id", "label"),
    [(1, "pedestrian"), (2, "cyclist"), (3, "car"), (4, "motorcycle"), (6, "bus"), (8, "truck")],
)
def test_coco_ids_map_to_fusion_labels(coco_id: int, label: str) -> None:
    assert coco_label(coco_id) == label


@pytest.mark.parametrize("coco_id", [0, 5, 7, 9, 10, 91, -1])
def test_other_coco_classes_are_dropped(coco_id: int) -> None:
    # 5 is airplane, 7 is train, 9 is boat, 10 is traffic light: none are road users we weight.
    assert coco_label(coco_id) is None


def test_coco_ids_name_the_expected_classes_in_torchvision() -> None:
    """The ids above are torchvision's 91-slot COCO indexing; check against its own metadata."""
    pytest.importorskip("torchvision")
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as Weights

    names = Weights.COCO_V1.meta["categories"]
    expected = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        6: "bus",
        8: "truck",
    }
    assert {i: names[i] for i in COCO_TO_LABEL} == expected


@pytest.mark.parametrize(
    ("category", "label"),
    [
        ("human.pedestrian.adult", "pedestrian"),
        ("human.pedestrian.child", "pedestrian"),
        ("human.pedestrian.construction_worker", "pedestrian"),
        ("vehicle.bicycle", "cyclist"),
        ("vehicle.motorcycle", "motorcycle"),
        ("vehicle.bus.bendy", "bus"),
        ("vehicle.bus.rigid", "bus"),
        ("vehicle.truck", "truck"),
        ("vehicle.trailer", "truck"),
        ("vehicle.construction", "truck"),
        ("vehicle.car", "car"),
        ("vehicle.emergency.police", "car"),
        ("vehicle.emergency.ambulance", "car"),
    ],
)
def test_nuscenes_categories_map_to_fusion_labels(category: str, label: str) -> None:
    assert nuscenes_label(category) == label


@pytest.mark.parametrize(
    "category",
    ["movable_object.barrier", "movable_object.trafficcone", "static_object.bicycle_rack", ""],
)
def test_other_nuscenes_categories_are_dropped(category: str) -> None:
    assert nuscenes_label(category) is None
