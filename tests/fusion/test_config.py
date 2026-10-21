from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any

import pytest
import yaml

from fusion.config import DEFAULT_PATH, SEVERITIES, load_fusion_config, parse_fusion_config


def _raw() -> dict[str, Any]:
    with DEFAULT_PATH.open() as f:
        loaded: dict[str, Any] = yaml.safe_load(f)
    return loaded


def test_shipped_config_loads_and_is_in_range() -> None:
    """The committed starting values load and every weight is a valid probability-like number."""
    cfg = load_fusion_config()
    assert set(cfg.severity_weights) == set(SEVERITIES)
    assert cfg.severity_weights["low"] < cfg.severity_weights["medium"] < 1.0
    assert cfg.severity_weights["high"] == 1.0
    assert cfg.severity_weights["unknown"] == cfg.severity_weights["medium"]
    assert cfg.vulnerability("pedestrian") > cfg.vulnerability("car")
    assert cfg.vulnerability("no-such-label") == cfg.default_vulnerability


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("severity_weights", "high"), 1.5),
        (("severity_weights", "low"), -0.1),
        (("vulnerability_weights", "pedestrian"), 1.01),
        (("default_vulnerability",), -1),
        (("road_overlap_threshold",), 1.1),
        (("road_overlap_threshold",), -0.1),
        (("road_overlap_threshold",), math.nan),
        (("road_overlap_threshold",), math.inf),
        (("near_radius",), -1.0),
        (("near_radius",), math.inf),
        (("distance_scale",), 0),
        (("distance_scale",), -2.0),
        (("distance_scale",), math.inf),
        (("distance_scale",), math.nan),
        (("min_object_height_px",), 0),
        (("min_object_height_px",), -5),
        (("min_object_height_px",), math.inf),
        (("distance_scale",), "1.0"),
        (("distance_scale",), True),
    ],
)
def test_bad_settings_are_rejected(path: tuple[str, ...], value: object) -> None:
    """Out-of-range, non-finite, non-positive and non-numeric settings fail at load time."""
    raw = copy.deepcopy(_raw())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        parse_fusion_config(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_scale", 0.0),
        ("distance_scale", math.inf),
        ("road_overlap_threshold", 1.1),
        ("near_radius", -1.0),
        ("min_object_height_px", 0.0),
        ("default_vulnerability", 2.0),
    ],
)
def test_replace_and_direct_construction_are_validated_too(field: str, value: float) -> None:
    """dataclasses.replace validates too, not only the YAML loader."""
    changes: dict[str, Any] = {field: value}
    with pytest.raises(ValueError):
        dataclasses.replace(load_fusion_config(), **changes)


def test_weight_mappings_are_read_only_and_copied() -> None:
    """A shared config cannot be changed by one caller, and the source dict is not aliased."""
    raw = copy.deepcopy(_raw())
    cfg = parse_fusion_config(raw)
    with pytest.raises(TypeError):
        cfg.severity_weights["unknown"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        cfg.vulnerability_weights["pedestrian"] = 0.0  # type: ignore[index]
    raw["severity_weights"]["high"] = 0.0
    assert cfg.severity_weights["high"] == 1.0


@pytest.mark.parametrize("bad", [{"low": 0.3, "medium": 0.6, "high": 1.0}, {"extreme": 1.0}])
def test_severity_keys_must_be_exactly_the_contract_values(bad: dict[str, float]) -> None:
    """A missing or unknown severity key is rejected, so weights cannot drift from the schema."""
    raw = copy.deepcopy(_raw())
    raw["severity_weights"] = bad
    with pytest.raises(ValueError, match="severity_weights"):
        parse_fusion_config(raw)
