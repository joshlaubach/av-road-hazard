"""Load and validate the fusion settings in configs/fusion.yaml.

Validation happens at load time so a bad value fails loudly instead of changing results. Example:
a road overlap threshold of 1.1 would mark every defect off-road and silently score it zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "fusion.yaml"
SEVERITIES = ("low", "medium", "high", "unknown")


@dataclass(frozen=True)
class FusionConfig:
    """Validated settings. All weights are in [0, 1]; scale and minimum are strictly positive.

    Validation runs in __post_init__, so it also covers direct construction and
    dataclasses.replace, not only loading from YAML. The weight mappings are copied into read-only
    views, so a shared config cannot be changed after the fact by one caller.
    """

    severity_weights: Mapping[str, float]
    vulnerability_weights: Mapping[str, float]
    default_vulnerability: float
    distance_scale: float
    min_object_height_px: float
    near_radius: float
    road_overlap_threshold: float

    def __post_init__(self) -> None:
        """Check every setting and replace the fields with clean, read-only values."""
        if set(self.severity_weights) != set(SEVERITIES):
            raise ValueError(
                f"severity_weights keys must be exactly {SEVERITIES}, "
                f"got {sorted(self.severity_weights)}"
            )
        near_radius = _number(self.near_radius, "near_radius")
        if near_radius < 0.0:
            raise ValueError(f"near_radius must be >= 0, got {near_radius}")
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(
            self,
            "severity_weights",
            MappingProxyType(
                {k: _unit(v, f"severity_weights.{k}") for k, v in self.severity_weights.items()}
            ),
        )
        object.__setattr__(
            self,
            "vulnerability_weights",
            MappingProxyType(
                {
                    k: _unit(v, f"vulnerability_weights.{k}")
                    for k, v in self.vulnerability_weights.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "default_vulnerability",
            _unit(self.default_vulnerability, "default_vulnerability"),
        )
        object.__setattr__(self, "distance_scale", _positive(self.distance_scale, "distance_scale"))
        object.__setattr__(
            self,
            "min_object_height_px",
            _positive(self.min_object_height_px, "min_object_height_px"),
        )
        object.__setattr__(self, "near_radius", near_radius)
        object.__setattr__(
            self,
            "road_overlap_threshold",
            _unit(self.road_overlap_threshold, "road_overlap_threshold"),
        )

    def vulnerability(self, label: str) -> float:
        """Weight for an object label, falling back to the default for unlisted labels."""
        return self.vulnerability_weights.get(label, self.default_vulnerability)


def _number(value: Any, name: str) -> float:
    """A finite real number (bool is rejected: True would silently mean 1.0)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _unit(value: Any, name: str) -> float:
    number = _number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {number}")
    return number


def _positive(value: Any, name: str) -> float:
    number = _number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0, got {number}")
    return number


def parse_fusion_config(raw: dict[str, Any]) -> FusionConfig:
    """Build a FusionConfig from a settings mapping (already loaded from YAML).

    Validation is done by FusionConfig itself; a missing key raises KeyError.
    """
    return FusionConfig(
        severity_weights=raw["severity_weights"],
        vulnerability_weights=raw["vulnerability_weights"],
        default_vulnerability=raw["default_vulnerability"],
        distance_scale=raw["distance_scale"],
        min_object_height_px=raw["min_object_height_px"],
        near_radius=raw["near_radius"],
        road_overlap_threshold=raw["road_overlap_threshold"],
    )


def load_fusion_config(path: Path | None = None) -> FusionConfig:
    """Read configs/fusion.yaml (or `path`) and return validated settings."""
    with (path or DEFAULT_PATH).open() as f:
        return parse_fusion_config(yaml.safe_load(f))
