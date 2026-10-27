"""Defect detection entry point: the tile classifier behind find_defects, or a stub.

With USE_STUBS on, find_defects returns one fixed fake defect. With stubs off it loads the tile
classifier named by defects.classifier_checkpoint (env DEFECT_CHECKPOINT), tiles the frame, skips
tiles that objects cover or that are off the road, applies the per-class thresholds from the
config (all under `defects:` in configs/default.yaml), and merges neighboring flagged tiles of one
class into one defect (defects/merge.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from common.config import get_config
from defects.labels import CODES
from defects.merge import combine, merge_defects, merge_groups
from defects.patches import IntBox
from fusion.fuse import RoadMask
from fusion.schema import Defect, Object

if TYPE_CHECKING:
    from defects.tile_defects import ScoredTile


class DefectsUnavailable(RuntimeError):
    """The tile classifier cannot be used: no checkpoint set, the file is missing, or it is invalid.

    The API answers 503 instead of returning no defects, which would look like a clean road.
    """


class Classifier(NamedTuple):
    """A loaded tile classifier and the training config its tiling and transform need."""

    model: Any
    cfg: dict[str, Any]


class DefectSettings(NamedTuple):
    """Thresholds in CODES order, plus the object and road cover limits for tile filtering."""

    thresholds: tuple[float, ...]
    max_object_cover: float
    min_road_cover: float


def defect_settings(cfg: dict[str, Any]) -> DefectSettings:
    """The `defects:` section of the shared config; thresholds must name exactly the classes."""
    section = cfg["defects"]
    by_code = section["thresholds"]
    if set(by_code) != set(CODES):
        raise ValueError(
            f"defects.thresholds must have exactly the keys {list(CODES)}, got {by_code}"
        )
    return DefectSettings(
        tuple(float(by_code[code]) for code in CODES),
        float(section["max_object_cover"]),
        float(section["min_road_cover"]),
    )


@lru_cache(maxsize=1)
def _load_classifier(checkpoint: Path) -> Classifier:
    import json

    from defects.model import load_checkpoint

    if not checkpoint.is_file():
        raise DefectsUnavailable(f"classifier checkpoint not found: {checkpoint}")
    try:
        sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
        # load_checkpoint checks the file hash and class order against the sidecar before loading.
        model = load_checkpoint(checkpoint, sidecar["config"]).eval()
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        raise DefectsUnavailable(f"classifier checkpoint could not be loaded: {exc}") from exc
    return Classifier(model, sidecar["config"])


def load_classifier() -> Classifier:
    """The configured classifier, loaded once per process."""
    path = get_config()["defects"]["classifier_checkpoint"]
    if not path:
        raise DefectsUnavailable(
            "no classifier checkpoint set: put its path in defects.classifier_checkpoint "
            "or the DEFECT_CHECKPOINT environment variable"
        )
    return _load_classifier(Path(path).resolve())


def find_defects(
    frame: np.ndarray,
    **settings: Any,
) -> list[Defect]:
    """find_tile_defects with neighboring flagged tiles of one class merged into one defect.

    Same arguments as find_tile_defects. This is what the pipeline reports (defects/merge.py).
    """
    return merge_defects(find_tile_defects(frame, **settings))


def find_tile_defects(
    frame: np.ndarray,
    *,
    objects: Sequence[Object] = (),
    road_mask: RoadMask | None = None,
    classifier: Classifier | None = None,
    settings: DefectSettings | None = None,
    region: IntBox | None = None,
) -> list[Defect]:
    """Find road defects in an RGB uint8 frame (H, W, 3). Boxes are pixel (x1, y1, x2, y2).

    objects and road_mask are optional filters: tiles mostly covered by objects, or mostly off the
    road, are not scored. With no mask the road is unknown and no tile is dropped. region is
    (x1, y1, x2, y2) in frame pixels and limits scoring to that part of the frame (default: all of
    it); tiles are laid out from its top-left corner, so a sub-region can score different crops than
    the whole frame does. Boxes stay in frame pixels. classifier and settings default to the
    configured ones. One Defect per flagged tile and class, so tile boxes stay exact (the
    evaluation tools match reviewed tiles by box); find_defects merges neighbors.
    """
    cfg = get_config()
    if cfg["pipeline"]["use_stubs"]:
        return _stub_defects(frame)
    from defects.tile_defects import find_frame_defects

    used = classifier or load_classifier()
    chosen = settings or defect_settings(cfg)
    return find_frame_defects(
        used.model,
        frame,
        objects,
        used.cfg,
        thresholds=chosen.thresholds,
        max_object_cover=chosen.max_object_cover,
        road_mask=road_mask,
        min_road_cover=chosen.min_road_cover,
        region=region,
    )


@dataclass(frozen=True)
class TracedDefects:
    """Everything find_defects did on a frame: every tile with its fate, and the merged groups.

    groups holds the flagged tiles of each merged defect, so merged() is exactly what find_defects
    returns. tile_size and tile_overlap describe the grid; thresholds are in CODES order.
    """

    tiles: list[ScoredTile]
    tile_size: int
    tile_overlap: float
    thresholds: tuple[float, ...]
    max_object_cover: float
    min_road_cover: float
    groups: list[list[Defect]]

    def merged(self) -> list[Defect]:
        return [combine(group) for group in self.groups]


def trace_defects(
    frame: np.ndarray,
    *,
    objects: Sequence[Object] = (),
    road_mask: RoadMask | None = None,
    classifier: Classifier | None = None,
    settings: DefectSettings | None = None,
    region: IntBox | None = None,
) -> TracedDefects:
    """find_defects, keeping the per-tile story: which tiles were skipped and why, each scored
    tile's class probabilities, and which flagged tiles merged into which defect."""
    cfg = get_config()
    chosen = settings or defect_settings(cfg)
    if cfg["pipeline"]["use_stubs"]:
        return _stub_trace(frame, chosen)
    from defects.tile_defects import score_frame_tiles, tiles_to_defects

    used = classifier or load_classifier()
    scored = score_frame_tiles(
        used.model,
        frame,
        objects,
        used.cfg,
        max_object_cover=chosen.max_object_cover,
        road_mask=road_mask,
        min_road_cover=chosen.min_road_cover,
        region=region,
    )
    flagged = [s for s in scored if s.probabilities is not None]
    raw = tiles_to_defects(
        [s.tile for s in flagged], [list(s.probabilities or ()) for s in flagged], chosen.thresholds
    )
    return TracedDefects(
        scored,
        int(used.cfg["tile_size"]),
        float(used.cfg["tile_overlap"]),
        chosen.thresholds,
        chosen.max_object_cover,
        chosen.min_road_cover,
        merge_groups(raw),
    )


def _stub_trace(frame: np.ndarray, chosen: DefectSettings) -> TracedDefects:
    """A fixed fake trace for the stub pipeline: a 2-tile-wide grid, one tile area flagged."""
    from defects.patches import tile_grid
    from defects.tile_defects import ScoredTile

    h, w = frame.shape[:2]
    stub = _stub_defects(frame)[0]
    size, overlap = max(1, min(h, w) // 2), 0.5
    quiet = 0.04
    scored: list[ScoredTile] = []
    for tile in tile_grid((w, h), size, overlap):
        x1, y1, x2, y2 = tile.box
        hit = (
            x1 <= (stub.box[0] + stub.box[2]) / 2 <= x2
            and y1 <= (stub.box[1] + stub.box[3]) / 2 <= y2
        )
        row = (quiet, quiet, quiet, stub.confidence if hit else quiet)
        scored.append(ScoredTile(tile, 0.0, None, "scored", row))
    return TracedDefects(
        scored, size, overlap, chosen.thresholds, chosen.max_object_cover, chosen.min_road_cover,
        [[stub]],
    )  # fmt: skip


def _stub_defects(frame: np.ndarray) -> list[Defect]:
    h, w = frame.shape[:2]  # shape is (height, width, channels)
    # Stub only: class names/ids are placeholders, not the RDD2022 label map (defects/labels.py).
    return [
        Defect(
            cls=4,
            name="pothole",
            confidence=0.84,
            box=(0.42 * w, 0.78 * h, 0.52 * w, 0.88 * h),
            patch_row=6,
            patch_col=3,
        )
    ]
