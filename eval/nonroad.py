"""How often the tile classifier flags tiles that are not road (the "non-road false-positive rate").

The shipped pipeline never scores a tile with less than DEFAULT_MIN_ROAD of road under it, so this
measures what the road filter protects against: the classifier's own behavior on sky, trees,
buildings and shoulders. Pure functions over per-tile probabilities and road cover; the tool that
scores real frames is tools/eval_nonroad.py.

Two bands, by the share of the tile that is road:
  clearly_not_road      under 5% road. Treated as having no road damage, so every flag is a false
                        positive.
  partly_road_skipped   5% up to the pipeline's cut. May hold real damage at the road's edge, so
                        its rate is an upper bound on the false-positive rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from defects.road_region import DEFAULT_MIN_ROAD
from eval.classifier import wilson_interval

# Name -> (lowest cover included, first cover excluded).
BANDS: dict[str, tuple[float, float]] = {
    "clearly_not_road": (0.0, 0.05),
    "partly_road_skipped": (0.05, DEFAULT_MIN_ROAD),
}


def _rate(count: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(count, total)
    rate = count / total if total else None
    return {"count": count, "total": total, "rate": rate, "low90": low, "high90": high}


def non_road_report(
    probabilities: npt.NDArray[np.float64],
    cover: npt.NDArray[np.float64],
    thresholds: Sequence[float],
    codes: Sequence[str],
) -> dict[str, Any]:
    """Flag rates for the non-road bands.

    probabilities has one row per tile and one column per class in codes order, cover one value in
    [0, 1] per tile (the share of the tile on road). A tile is flagged when any class reaches its
    own threshold, the same rule the pipeline uses. Tiles at or above the pipeline's road cut are
    road tiles and are left out.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != len(codes):
        raise ValueError(f"need one column per class ({len(codes)}), got {probabilities.shape}")
    if len(cover) != len(probabilities):
        raise ValueError("need one coverage per tile")
    if len(thresholds) != len(codes):
        raise ValueError(f"need one threshold per class ({len(codes)}), got {len(thresholds)}")
    above = probabilities >= np.asarray(thresholds, dtype=float)
    report: dict[str, Any] = {}
    for name, (low, high) in BANDS.items():
        inside = (cover >= low) & (cover < high)
        total = int(inside.sum())
        report[name] = {
            "tiles": total,
            "cover_from": low,
            "cover_below": high,
            "flagged": _rate(int(above[inside].any(axis=1).sum()), total),
            "per_class": {
                c: _rate(int(above[inside, k].sum()), total) for k, c in enumerate(codes)
            },
        }
    return report
