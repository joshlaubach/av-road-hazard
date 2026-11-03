"""How much does the hazard score move when the detectors make one mistake?

Usage: python -m tools.noise_test [--records runs/answers-records.json] [--out DIR]
Takes the real hazard records of the 404 nuScenes mini frames and changes one thing at a time:
  false defect   one made-up defect is added to the frame (one tile wide, on the road, lower half)
  missed object  one detected object is removed from a frame that has a defect
The scores are recomputed with the real fusion functions (`fusion.fuse`), nothing is re-run on
images. A change counts as meaningful when the gauge (0 to 100) moves by at least 10 points, or
when the top-10 list of frames changes. Counts come with 90% Wilson ranges. Exploratory: one
dataset, made-up false defects, no motion or video.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from eval.classifier import wilson_interval
from fusion.config import FusionConfig, load_fusion_config
from fusion.fuse import defect_score, proximity
from fusion.schema import Box, DefectRecord, HazardRecord, Object

GAUGE_POINTS = 10.0  # a change of at least this many points on the 0-100 gauge is meaningful
TOP_N = 10
TILE_PX = 384  # one tile wide, the size of a single flagged tile
TOLERANCE = 1e-6


def frame_score(
    objects: Sequence[Object], defects: Sequence[DefectRecord], cfg: FusionConfig
) -> float:
    """The frame's hazard score, from stored defects: the max defect score, capped at 1."""
    scores = [
        defect_score(d.in_road, d.severity, d.confidence, proximity(d.box, objects, cfg), cfg)
        for d in defects
    ]
    return min(1.0, max(scores, default=0.0))


def top_n(scores: dict[str, float]) -> list[str]:
    """The n most urgent frames, ties broken by frame id so the list is stable."""
    return heapq.nsmallest(TOP_N, scores, key=lambda f: (-scores[f], f))


def false_defect(
    record: HazardRecord, name: str, confidence: float, rng: random.Random, cfg: FusionConfig
) -> DefectRecord:
    """A made-up defect, on the road and in the lower half of the frame, one tile wide."""
    width, height = record.image_size
    x1 = rng.uniform(0, max(width - TILE_PX, 0))
    y1 = rng.uniform(0.4 * height, max(height - TILE_PX, 0.4 * height))
    box: Box = (x1, y1, min(x1 + TILE_PX, width), min(y1 + TILE_PX, height))
    return DefectRecord(
        name=name, confidence=confidence, box=box, severity=cfg.severity_for(name), in_road=True
    )


def effect(base: dict[str, float], frame: str, new_score: float) -> tuple[float, bool, bool]:
    """(gauge points moved, top-10 members changed, top-10 order changed) for one changed frame."""
    before = top_n(base)
    after = top_n({**base, frame: new_score})
    return 100 * abs(new_score - base[frame]), set(before) != set(after), before != after


def summarize(cases: Sequence[tuple[float, bool, bool]]) -> dict[str, Any]:
    """Counts and 90% ranges: share that moved >= 10 points, changed top-10 members or order."""

    def share(count: int) -> dict[str, Any]:
        low, high = wilson_interval(count, len(cases))
        return {"count": count, "total": len(cases), "low90": low, "high90": high}

    moves = sorted(c[0] for c in cases)
    return {
        "cases": len(cases),
        "median_points": moves[len(moves) // 2] if moves else 0.0,
        "max_points": moves[-1] if moves else 0.0,
        "moved_10_points_or_more": share(sum(c[0] >= GAUGE_POINTS for c in cases)),
        "top10_members_changed": share(sum(c[1] for c in cases)),
        "top10_order_changed": share(sum(c[2] for c in cases)),
    }


def run(
    records: Sequence[HazardRecord],
    thresholds: dict[str, float],
    cfg: FusionConfig,
    *,
    positions: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Recompute every frame, check the stored scores, then apply each single mistake."""
    base = {r.frame_id: frame_score(r.objects, r.defects, cfg) for r in records}
    mismatched = sum(abs(base[r.frame_id] - min(1.0, r.hazard_score)) > TOLERANCE for r in records)
    rng = random.Random(seed)
    report: dict[str, Any] = {
        "frames": len(records),
        "frames_with_defects": sum(bool(r.defects) for r in records),
        "recomputed_scores_differing_from_stored": mismatched,
        "base_top10": [(f, round(100 * base[f])) for f in top_n(base)],
        "false_defect": {},
        "missed_object": {},
    }
    for label, pick in (
        ("at_threshold", lambda t: t),  # the weakest score that still raises a flag
        ("confident_0.95", lambda t: 0.95),
    ):
        report["false_defect"][label] = _false_defect_cases(
            records, base, thresholds, pick, positions, rng, cfg
        )
    report["missed_object"] = _missed_object_cases(records, base, cfg)
    return report


def _false_defect_cases(
    records: Sequence[HazardRecord],
    base: dict[str, float],
    thresholds: dict[str, float],
    confidence: Callable[[float], float],
    positions: int,
    rng: random.Random,
    cfg: FusionConfig,
) -> dict[str, Any]:
    by_group: dict[str, list[tuple[float, bool, bool]]] = {
        "clean_frames": [],
        "frames_with_defects": [],
    }
    per_class: dict[str, list[tuple[float, bool, bool]]] = {name: [] for name in thresholds}
    for r in records:
        group = "frames_with_defects" if r.defects else "clean_frames"
        for name, threshold in thresholds.items():
            for _ in range(positions):
                fake = false_defect(r, name, confidence(threshold), rng, cfg)
                score = max(base[r.frame_id], frame_score(r.objects, [fake], cfg))
                case = effect(base, r.frame_id, score)
                by_group[group].append(case)
                per_class[name].append(case)
    return {
        "by_frame_group": {k: summarize(v) for k, v in by_group.items()},
        "by_class": {k: summarize(v) for k, v in per_class.items()},
    }


def _missed_object_cases(
    records: Sequence[HazardRecord], base: dict[str, float], cfg: FusionConfig
) -> dict[str, Any]:
    cases: list[tuple[float, bool, bool]] = []
    for r in records:
        if not r.defects:
            continue
        for i in range(len(r.objects)):
            kept = [o for j, o in enumerate(r.objects) if j != i]
            cases.append(effect(base, r.frame_id, frame_score(kept, r.defects, cfg)))
    return summarize(cases)


def main(argv: list[str] | None = None) -> int:
    from common.config import get_config
    from defects.infer import defect_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("runs/answers-records.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/noise-test"))
    parser.add_argument(
        "--positions", type=int, default=5, help="random places per frame and class"
    )
    args = parser.parse_args(argv)
    data = json.loads(args.records.read_text())
    records = [HazardRecord.model_validate(r) for r in data["records"]]
    settings = defect_settings(get_config())
    thresholds = dict(zip(("D00", "D10", "D20", "D40"), settings.thresholds, strict=True))
    report = run(records, thresholds, load_fusion_config(), positions=args.positions)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
