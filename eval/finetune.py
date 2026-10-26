"""Scoring a fine-tuned detector scene by scene, and checking it against the ADR 0015 limits.

Pure helpers for tools/eval_finetune.py. Every scene is scored only with a model that never trained
on it (leave-one-scene-out). The score cutoff of a new class is chosen for each scene from the
other scenes' held-out detections, never from the scene being scored. Counting follows
eval/detector.py: a prediction matches an evaluable ground-truth box of the same label at
IoU 0.5 or more, greedily by score, and a prediction with no match is a false positive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eval.classifier import wilson_interval
from eval.detector import FrameData, evaluable
from eval.models import GroundTruth, match_detections

Counts = dict[str, int]  # tp, fp, fn


def class_counts(data: Sequence[FrameData], label: str, cutoff: float) -> Counts:
    """True positives, false positives and misses for one class at one score cutoff."""
    tp = fp = fn = 0
    for frame in data:
        preds = [d for d in frame.detections if d.label == label and d.score >= cutoff]
        truth = [
            GroundTruth(g.label, g.box)
            for g in frame.ground_truth
            if g.label == label and evaluable(g)
        ]
        matched = match_detections(preds, truth)
        tp, fp, fn = tp + len(matched.tp), fp + len(matched.fp), fn + len(matched.fn)
    return {"tp": tp, "fp": fp, "fn": fn}


def add_counts(items: Sequence[Counts]) -> Counts:
    return {key: sum(c[key] for c in items) for key in ("tp", "fp", "fn")}


def rates(counts: Counts) -> dict[str, Any]:
    """Precision and recall with 90% ranges; None where there is nothing to divide."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    out: dict[str, Any] = {**counts, "boxes": tp + fn, "predictions": tp + fp}
    out["precision"] = tp / (tp + fp) if tp + fp else None
    out["recall"] = tp / (tp + fn) if tp + fn else None
    out["precision_range90"] = wilson_interval(tp, tp + fp)
    out["recall_range90"] = wilson_interval(tp, tp + fn)
    return out


def scene_table(
    by_scene: Mapping[str, Sequence[FrameData]], label: str, grid: Sequence[float]
) -> dict[str, dict[float, Counts]]:
    """Counts for one class in every scene at every cutoff of the grid."""
    return {
        scene: {cutoff: class_counts(frames, label, cutoff) for cutoff in grid}
        for scene, frames in by_scene.items()
    }


def choose_cutoffs(
    table: Mapping[str, Mapping[float, Counts]], grid: Sequence[float], min_precision: float
) -> dict[str, dict[str, Any]]:
    """For each scene, the lowest cutoff whose precision on the OTHER scenes reaches the minimum.

    The lowest such cutoff keeps the most recall. If no cutoff on the grid reaches it, the highest
    is used and reached is False. The scene's own detections play no part in its cutoff.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for scene in table:
        others = [s for s in table if s != scene]
        pick, reached = grid[-1], False
        for cutoff in sorted(grid):
            pooled = add_counts([table[s][cutoff] for s in others])
            if (
                pooled["tp"] + pooled["fp"]
                and pooled["tp"] / (pooled["tp"] + pooled["fp"]) >= min_precision
            ):
                pick, reached = cutoff, True
                break
        chosen[scene] = {"cutoff": pick, "reached": reached}
    return chosen


def scene_rows(
    table: Mapping[str, Mapping[float, Counts]], cutoffs: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """One row per scene at its own cutoff: counts, precision and recall with ranges."""
    rows = []
    for scene in sorted(table):
        cutoff = cutoffs[scene]["cutoff"]
        rows.append({"scene": scene, "cutoff": cutoff, **rates(table[scene][cutoff])})
    return rows


def new_label_verdict(
    rows: Sequence[Mapping[str, Any]], label: str, limits: Mapping[str, Any]
) -> dict[str, Any]:
    """Whether a new class meets the precision and recall limits on enough scenes (ADR 0015).

    Only scenes with at least min_boxes clearly visible boxes count. The pooled counts over those
    scenes must reach both limits, and at least min_passing_scenes[label] scenes must reach them on
    their own. Passing with a 90% range that dips below a limit is a caveat, not a clean pass.
    """
    qualifying = [r for r in rows if r["boxes"] >= limits["min_boxes"]]
    pooled = rates(add_counts([{k: r[k] for k in ("tp", "fp", "fn")} for r in qualifying]))

    def reaches(r: Mapping[str, Any]) -> bool:
        return (
            r["precision"] is not None
            and r["recall"] is not None
            and r["precision"] >= limits["min_precision"]
            and r["recall"] >= limits["min_recall"]
        )

    passing = [r["scene"] for r in qualifying if reaches(r)]
    needed = limits["min_passing_scenes"][label]
    reasons: list[str] = []
    if len(qualifying) < needed:
        reasons.append(f"only {len(qualifying)} scene(s) have {limits['min_boxes']}+ boxes")
    if not reaches(pooled):
        reasons.append("pooled precision or recall is below the limit")
    if len(passing) < needed:
        reasons.append(f"{len(passing)} scene(s) pass on their own, {needed} needed")
    passed = not reasons
    caveat = passed and (
        pooled["precision_range90"][0] < limits["min_precision"]
        or pooled["recall_range90"][0] < limits["min_recall"]
    )
    return {
        "label": label,
        "qualifying_scenes": [r["scene"] for r in qualifying],
        "passing_scenes": passing,
        "pooled": pooled,
        "passed": passed,
        "caveat": caveat,
        "reasons": reasons,
    }


def regression_verdict(
    zero_shot: Counts, fine_tuned: Counts, limits: Mapping[str, Any]
) -> dict[str, Any]:
    """An old class may lose at most max_recall_drop of recall and max_precision_drop of it."""
    before, after = rates(zero_shot), rates(fine_tuned)
    reasons: list[str] = []
    for name, limit in (
        ("recall", limits["max_recall_drop"]),
        ("precision", limits["max_precision_drop"]),
    ):
        if before[name] is None or after[name] is None:
            reasons.append(f"no {name} to compare")
        elif before[name] - after[name] > limit + 1e-9:
            reasons.append(f"{name} fell {before[name] - after[name]:.3f} (limit {limit})")
    return {"zero_shot": before, "fine_tuned": after, "passed": not reasons, "reasons": reasons}


def build_report(
    fine_tuned: Mapping[str, Sequence[FrameData]],
    zero_shot: Mapping[str, Sequence[FrameData]],
    evaluation: Mapping[str, Any],
    old_classes: Sequence[str],
) -> dict[str, Any]:
    """Everything ADR 0015 asks of the scoring, for frames grouped by held-out scene.

    fine_tuned holds each scene's detections from the model that never trained on it; zero_shot
    holds the pinned COCO detector's detections on the same frames. Both use the same ground
    truth. New classes get a cutoff per scene chosen from the other scenes; old classes are
    compared with the zero-shot detector at the fixed old-class cutoff.
    """
    grid = list(evaluation["cutoff_grid"])
    limits = evaluation["limits"]
    new_report: dict[str, Any] = {}
    for label in evaluation["new_classes"]:
        table = scene_table(fine_tuned, label, grid)
        cutoffs = choose_cutoffs(table, grid, limits["min_precision"])
        rows = scene_rows(table, cutoffs)
        sweep = [{"cutoff": c, **rates(add_counts([table[s][c] for s in table]))} for c in grid]
        new_report[label] = {
            "cutoffs": cutoffs,
            "scenes": rows,
            "verdict": new_label_verdict(rows, label, limits),
            "sweep_all_scenes": sweep,
        }
    cutoff = evaluation["old_classes_cutoff"]
    old_report = {
        label: regression_verdict(
            add_counts([class_counts(f, label, cutoff) for f in zero_shot.values()]),
            add_counts([class_counts(f, label, cutoff) for f in fine_tuned.values()]),
            limits,
        )
        for label in old_classes
    }
    passed = all(v["verdict"]["passed"] for v in new_report.values()) and all(
        v["passed"] for v in old_report.values()
    )
    return {"new_classes": new_report, "old_classes": old_report, "passed": passed}
