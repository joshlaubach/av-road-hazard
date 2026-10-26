"""The label maps that connect the detector's classes to the labels fusion weights by."""

from __future__ import annotations

import pytest

from fusion.config import load_fusion_config
from perception.labels import (
    COCO_TO_LABEL,
    COCO_VOCABULARY,
    FUSION_LABELS,
    coco_label,
    nuscenes_label,
    to_vocabulary,
)


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
        ("human.pedestrian.wheelchair", "pedestrian"),
        ("human.pedestrian.stroller", "pedestrian"),
        ("human.pedestrian.child", "child"),
        ("human.pedestrian.construction_worker", "construction_worker"),
        ("human.pedestrian.police_officer", "police_officer"),
        ("human.pedestrian.personal_mobility", "personal_mobility"),
        ("vehicle.bicycle", "cyclist"),
        ("vehicle.motorcycle", "motorcycle"),
        ("vehicle.bus.bendy", "bus"),
        ("vehicle.bus.rigid", "bus"),
        ("vehicle.truck", "truck"),
        ("vehicle.trailer", "truck"),
        ("vehicle.construction", "construction_vehicle"),
        ("vehicle.car", "car"),
        ("vehicle.emergency.police", "car"),
        ("vehicle.emergency.ambulance", "car"),
        ("movable_object.barrier", "barrier"),
        ("movable_object.trafficcone", "traffic_cone"),
        ("movable_object.debris", "debris"),
    ],
)
def test_nuscenes_categories_map_to_fusion_labels(category: str, label: str) -> None:
    assert nuscenes_label(category) == label


@pytest.mark.parametrize(
    "category", ["movable_object.pushable_pullable", "static_object.bicycle_rack", "animal", ""]
)
def test_other_nuscenes_categories_are_dropped(category: str) -> None:
    assert nuscenes_label(category) is None


def test_the_coco_vocabulary_is_what_coco_can_emit() -> None:
    assert set(COCO_TO_LABEL.values()) == COCO_VOCABULARY


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("pedestrian", "pedestrian"),
        ("child", "pedestrian"),
        ("construction_worker", "pedestrian"),
        ("police_officer", "pedestrian"),
        ("personal_mobility", "pedestrian"),
        ("construction_vehicle", "truck"),
        ("car", "car"),
        ("barrier", None),
        ("traffic_cone", None),
        ("debris", None),
    ],
)
def test_a_label_the_detector_cannot_emit_falls_back_to_its_family_or_none(
    label: str, expected: str | None
) -> None:
    assert to_vocabulary(label, COCO_VOCABULARY) == expected


def test_a_detector_that_emits_the_fine_label_keeps_it() -> None:
    vocabulary = COCO_VOCABULARY | {"child", "barrier"}
    assert to_vocabulary("child", vocabulary) == "child"
    assert to_vocabulary("barrier", vocabulary) == "barrier"
    assert to_vocabulary("construction_worker", vocabulary) == "pedestrian"
