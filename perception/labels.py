"""Label maps from the detector's classes and from nuScenes to the labels fusion weights by.

The fusion labels are the keys of vulnerability_weights in configs/fusion.yaml. COCO has no
"cyclist" class, so a COCO bicycle becomes "cyclist" (a rider is not detected separately), and a
COCO person becomes "pedestrian". Both weights are 1.0 in fusion, so this choice costs nothing
there, but it is pinned by tests/perception/test_labels.py so it cannot drift unnoticed.
"""

from __future__ import annotations

FUSION_LABELS = (
    "pedestrian",
    "child",
    "construction_worker",
    "police_officer",
    "personal_mobility",
    "cyclist",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "construction_vehicle",
    "barrier",
    "traffic_cone",
    "debris",
)

# torchvision's detection models index COCO in 91 slots (some unused), not the 80 contiguous
# classes; these ids are checked against the weights' own category names in the test.
COCO_TO_LABEL: dict[int, str] = {
    1: "pedestrian",  # person
    2: "cyclist",  # bicycle
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

# The labels the COCO models can emit. A detector trained on more classes has a larger vocabulary.
COCO_VOCABULARY = frozenset(COCO_TO_LABEL.values())

# A finer label and the coarser label a detector without it would call the same object. Labels
# not listed here (barrier, traffic_cone, debris) have no coarser label: such a detector is blind.
_FAMILY: dict[str, str] = {
    "child": "pedestrian",
    "construction_worker": "pedestrian",
    "police_officer": "pedestrian",
    "personal_mobility": "pedestrian",
    "construction_vehicle": "truck",
}

# nuScenes category-name prefix -> fusion label, first match wins. Wheelchair and stroller users
# count as pedestrians, trailers as trucks, emergency vehicles as cars. Pushable carts and bicycle
# racks are not hazards we weight and are dropped.
_NUSCENES_PREFIXES: tuple[tuple[str, str], ...] = (
    ("human.pedestrian.child", "child"),
    ("human.pedestrian.construction_worker", "construction_worker"),
    ("human.pedestrian.police_officer", "police_officer"),
    ("human.pedestrian.personal_mobility", "personal_mobility"),
    ("human.pedestrian", "pedestrian"),
    ("vehicle.bicycle", "cyclist"),
    ("vehicle.motorcycle", "motorcycle"),
    ("vehicle.bus", "bus"),
    ("vehicle.truck", "truck"),
    ("vehicle.trailer", "truck"),
    ("vehicle.construction", "construction_vehicle"),
    ("vehicle.car", "car"),
    ("vehicle.emergency", "car"),
    ("movable_object.barrier", "barrier"),
    ("movable_object.trafficcone", "traffic_cone"),
    ("movable_object.debris", "debris"),
)


def coco_label(category_id: int) -> str | None:
    """The fusion label for a torchvision COCO category id, or None to drop the detection."""
    return COCO_TO_LABEL.get(category_id)


def nuscenes_label(category_name: str) -> str | None:
    """The fusion label for a nuScenes category name such as "vehicle.car", or None to drop it."""
    return next(
        (label for prefix, label in _NUSCENES_PREFIXES if category_name.startswith(prefix)), None
    )


def to_vocabulary(label: str, vocabulary: frozenset[str]) -> str | None:
    """The label a detector with this vocabulary would use for an object, or None if it is blind.

    A label in the vocabulary stays as it is. Otherwise the coarser family label is used if the
    vocabulary has it (a COCO model calls a child "pedestrian"). With neither, None: the object is
    outside what the detector can say, and is reported separately instead of counting as a miss.
    """
    if label in vocabulary:
        return label
    family = _FAMILY.get(label)
    return family if family in vocabulary else None
