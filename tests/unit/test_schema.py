import pytest
from pydantic import ValidationError

from fusion.schema import Defect, DefectRecord, HazardRecord, Object


def _record() -> HazardRecord:
    """A fully populated HazardRecord used by several tests."""
    return HazardRecord(
        frame_id="f1",
        source="upload",
        image_size=(1600, 900),
        objects=[Object(label="pedestrian", score=0.9, box=(10, 20, 30, 40))],
        defects=[
            DefectRecord(
                name="pothole",
                confidence=0.8,
                box=(1, 2, 3, 4),
                severity="high",
                in_road=True,
                near_objects=[{"label": "pedestrian", "iou": 0.0, "dist_px": 42}],
            )
        ],
        hazard_score=0.0,
    )


def test_in_road_may_be_unknown() -> None:
    """in_road accepts True, False and None (unknown); other types are rejected."""
    base = {"name": "p", "confidence": 0.5, "box": (0, 0, 1, 1), "severity": "low"}
    for value in (True, False, None):
        assert DefectRecord(**base, in_road=value).in_road is value
    with pytest.raises(ValidationError):
        DefectRecord(**base, in_road=[1])


def test_round_trip() -> None:
    """A record serializes to JSON and back without changing."""
    rec = _record()
    assert HazardRecord.model_validate_json(rec.model_dump_json()) == rec


def test_image_size_is_width_then_height() -> None:
    """image_size is (width, height) and a non-positive size is rejected."""
    assert _record().image_size == (1600, 900)
    with pytest.raises(ValidationError):
        HazardRecord(frame_id="f", source="upload", image_size=(0, 900), hazard_score=0.0)


def test_box_ordering_enforced() -> None:
    """A box with min greater than max (flipped corners) is rejected."""
    with pytest.raises(ValidationError):
        Object(label="car", score=0.5, box=(10, 10, 5, 20))


def test_unknown_fields_rejected() -> None:
    """Extra fields are an error, not silently ignored."""
    with pytest.raises(ValidationError):
        Object.model_validate({"label": "car", "score": 0.5, "box": [0, 0, 1, 1], "extra": 1})


def test_enums_rejected() -> None:
    """Only the listed sources are accepted."""
    with pytest.raises(ValidationError):
        HazardRecord(frame_id="f", source="webcam", image_size=(1, 1), hazard_score=0.0)


def test_defect_requires_one_indexed_class() -> None:
    """Defect classes start at 1; 0 is not a valid class id."""
    with pytest.raises(ValidationError):
        Defect(cls=0, name="x", confidence=0.5, box=(0, 0, 1, 1), patch_row=0, patch_col=0)
