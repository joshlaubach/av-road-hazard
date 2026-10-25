"""Label maps from the detector's classes and from nuScenes to the labels fusion weights by.

The fusion labels are the keys of vulnerability_weights in configs/fusion.yaml. COCO has no
"cyclist" class, so a COCO bicycle becomes "cyclist" (a rider is not detected separately), and a
COCO person becomes "pedestrian". Both weights are 1.0 in fusion, so this choice costs nothing
there, but it is pinned by tests/perception/test_labels.py so it cannot drift unnoticed.
"""

from __future__ import annotations

FUSION_LABELS = ("pedestrian", "cyclist", "motorcycle", "car", "truck", "bus")

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

# nuScenes category-name prefix -> fusion label, first match wins. Trailers and construction
# vehicles count as trucks, emergency vehicles as cars; everything else (barriers, cones,
# bicycle racks) is not a road user and is dropped.
_NUSCENES_PREFIXES: tuple[tuple[str, str], ...] = (
    ("human.pedestrian", "pedestrian"),
    ("vehicle.bicycle", "cyclist"),
    ("vehicle.motorcycle", "motorcycle"),
    ("vehicle.bus", "bus"),
    ("vehicle.truck", "truck"),
    ("vehicle.trailer", "truck"),
    ("vehicle.construction", "truck"),
    ("vehicle.car", "car"),
    ("vehicle.emergency", "car"),
)


def coco_label(category_id: int) -> str | None:
    """The fusion label for a torchvision COCO category id, or None to drop the detection."""
    return COCO_TO_LABEL.get(category_id)


def nuscenes_label(category_name: str) -> str | None:
    """The fusion label for a nuScenes category name such as "vehicle.car", or None to drop it."""
    return next(
        (label for prefix, label in _NUSCENES_PREFIXES if category_name.startswith(prefix)), None
    )
