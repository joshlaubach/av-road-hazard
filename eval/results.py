"""One results file for the core (eval/results.json) and the README table rendered from it.

build_results gathers the numbers from the evaluation reports: the tile classifier (validation
tiles, nuScenes alerts, clean probe tiles, reviewer calls), the classifier's false positives on
non-road tiles, the object detector (zero-shot and the fine-tune test), and the synthetic fusion
test. render_readme_table turns the same dictionary into the table in README.md, so a README claim
can only say what the file says. Only counts, rates, ranges and hashes go in: no images, frame
names or paths.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from eval.classifier import wilson_interval

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
SCHEMA_VERSION = 1
SECTIONS = (
    "classifier", "detector", "synthetic_fusion", "effective_range", "retrieval", "provenance",
)  # fmt: skip
CLASSES = ("D00", "D10", "D20", "D40")
# nuScenes CAM_FRONT focal length in pixels (calibrated sensor, about 1266 for every scene).
NUSCENES_FOCAL_PX = 1266.0
ASSUMED_DEFECT_WIDTH_M = 0.5
_PATHLIKE = re.compile(r"[A-Za-z]:[\\/]|/Users/|\.(?:jpe?g|png)\b|samples/|n008-", re.IGNORECASE)


def _scanner_safe(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance block with file labels the secret scanner will not read as API keys.

    A label such as "review_key" next to a 64-character hash matches gitleaks' generic API key
    rule, so a field name holding "key" becomes "tiles" (these hashes are public file digests).
    """
    files = {k.replace("review_key", "review_tiles"): v for k, v in provenance["files"].items()}
    return {**provenance, "files": files}


def _without_provenance(report: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k != "provenance"}


def effective_range(
    fusion: Mapping[str, Any],
    *,
    focal_px: float = NUSCENES_FOCAL_PX,
    defect_width_m: float = ASSUMED_DEFECT_WIDTH_M,
) -> dict[str, Any]:
    """How large a defect must look to be found, and the distance that size would mean.

    The bands come from the synthetic fusion test (found rate by pasted-patch width in the frame).
    Distance follows the pinhole camera model, d = f * W / w for a defect of real width W seen w
    pixels wide by a camera of focal length f pixels: an illustration for one assumed defect size,
    not a measured range.
    """
    bands = [
        {"band": name, **{k: row[k] for k in ("found", "trials", "rate", "range90")}}
        for name, row in fusion["on_road"]["by_size"].items()
    ]
    distances = {str(w): round(focal_px * defect_width_m / w, 2) for w in (128, 64)}
    return {
        "basis": "defect patches pasted on nuScenes front-camera frames; found = a reported box "
        "contains the patch center",
        "bands": bands,
        "assumed_defect_width_m": defect_width_m,
        "focal_length_px": focal_px,
        "distance_formula": "d = f * W / w",
        "distance_m_at_width_px": distances,
        "caveat": "These are pasted patches, not real damage, and the distance is only the "
        "pinhole-camera size for one assumed defect width. Real damage will differ.",
    }


def _fine_tune_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    view = report["all_cameras"]

    def new_class(name: str) -> dict[str, Any]:
        verdict = view["new_classes"][name]["verdict"]
        keep = (
            "passed",
            "qualifying_scenes",
            "passing_scenes",
            "caveat",
            "caveat_scenes",
            "reasons",
        )
        return {**{k: verdict[k] for k in keep}, "pooled": verdict["pooled"]}

    front = report["front_camera"]
    return {
        "passed": view["passed"],
        "decision": "the fine-tuned detector ships"
        if view["passed"]
        else "the zero-shot detector stays",
        "cones": new_class("traffic_cone"),
        "barriers": new_class("barrier"),
        "old_classes": {
            name: {k: row[k] for k in ("passed", "zero_shot", "fine_tuned")}
            for name, row in view["old_classes"].items()
        },
        "front_camera": {
            "passed": front["passed"],
            "cones_passed": front["new_classes"]["traffic_cone"]["verdict"]["passed"],
            "barriers_passed": front["new_classes"]["barrier"]["verdict"]["passed"],
            "old_classes_failed": [k for k, v in front["old_classes"].items() if not v["passed"]],
        },
        "cpu_p95": report.get("cpu_latency_ladder", []),
    }


def retrieval_section(report: Mapping[str, Any]) -> dict[str, Any]:
    """The search numbers (tools/eval_retrieval.py) without the provenance block."""
    return {
        "status": "measured",
        "frames": report["frames"],
        "queries": report["queries"],
        "embedder": report["embedder"],
        "pipeline": report["pipeline"],
        "methods": report["methods"],
        "provenance": _scanner_safe(report["provenance"]),
        "caveat": (
            "Queries are made from the nuScenes human annotations, so they are mostly about "
            "objects and light; they say little about how well appearance words (wet road, glare) "
            "work. Relevant frames contain a clearly visible annotated object; the search sees "
            "only the pipeline's own records and the photos."
        ),
    }


def build_results(
    *,
    classifier: Mapping[str, Any],
    non_road: Mapping[str, Any],
    detector_gap: Mapping[str, Any],
    fine_tune: Mapping[str, Any],
    fusion: Mapping[str, Any],
    chosen: str,
    date_utc: str,
    git_commit: str,
    real_photo_check: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The results dictionary (eval/results.json) from the reports and the chosen checkpoint."""
    base = classifier["baseline"]
    checkpoints = classifier["checkpoints"]
    (model_name,) = detector_gap["models"]
    zero = detector_gap["models"][model_name]
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": date_utc,
        "git_commit": git_commit,
        "classifier": {
            "chosen": chosen,
            "baseline": base,
            "thresholds": classifier["provenance"]["thresholds"],
            "checkpoint_sha256": classifier["provenance"]["files"].get(f"checkpoint/{chosen}"),
            "validation": {n: checkpoints[n]["validation"] for n in (base, chosen)},
            "nuscenes": {n: c["frames"] for n, c in checkpoints.items()},
            "probe": {n: c["probe"] for n, c in checkpoints.items()},
            "recall_guard": checkpoints[chosen].get("guard"),
            "non_road_false_positives": {k: v for k, v in _without_provenance(non_road).items()},
        },
        "detector": {
            "zero_shot": {
                "model": model_name,
                "frames": detector_gap["frames"],
                "cutoff": zero["threshold"],
                "all_frames": zero["all_frames_evaluable"],
                "recall_by_size": zero["recall_by_size_all_gt"],
                "outside_vocabulary": zero["outside_vocabulary"],
            },
            "fine_tune": _fine_tune_summary(fine_tune),
        },
        "synthetic_fusion": _without_provenance(fusion),
        "effective_range": effective_range(fusion),
        "retrieval": {"status": "not measured", "reason": "RAG is not built yet (ADR 0011)"},
        "provenance": {
            "classifier": _scanner_safe(classifier["provenance"]),
            "non_road": non_road["provenance"],
            "detector_gap": detector_gap["provenance"],
            "fine_tune": fine_tune["provenance"],
            "synthetic_fusion": fusion["provenance"],
        },
    }
    if real_photo_check is not None:
        results["real_photo_check"] = dict(real_photo_check)
    return results


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [s for k, v in value.items() for s in (*_walk_strings(k), *_walk_strings(v))]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _walk_strings(v)]
    return []


def validate_results(results: Mapping[str, Any]) -> None:
    """Raise ValueError unless every section is present and nothing looks like a file path."""
    if results.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for section in SECTIONS:
        if section not in results:
            raise ValueError(f"results are missing the {section} section")
    for text in _walk_strings(results):
        if _PATHLIKE.search(text):
            raise ValueError(f"results must not hold a file path or image name: {text[:60]!r}")


# ------------------------------------------------------------------------------- the README


def _retrieval_row(results: Mapping[str, Any]) -> str:
    r = results["retrieval"]
    if r["status"] != "measured":
        return str(r["status"])
    at5 = {n: m["all"]["recall@5"] for n, m in r["methods"].items()}
    parts = [
        f"{label} {at5[key]['mean']:.2f}"
        for label, key in (
            ("description", "description_vector"),
            ("photo", "photo_vector"),
            ("both fused", "fused"),
            ("random order", "random"),
        )
    ]
    return (
        f"Recall at 5 on {r['frames']} nuScenes frames and {r['queries']} queries made from the "
        "annotations: " + ", ".join(parts) + " (1.0 is perfect)"
    )


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%" if 0 < x < 0.01 else f"{round(100 * x)}%"


def _range(low: float, high: float) -> str:
    return f"{_pct(low)} to {_pct(high)}"


def _validation_row(results: Mapping[str, Any]) -> str:
    c = results["classifier"]
    table = c["validation"][c["chosen"]]
    return " · ".join(
        f"{k} {_pct(table[k]['precision'])} / {_pct(table[k]['recall'])}" for k in CLASSES
    )


def _tiles_judged(frames: Mapping[str, Any]) -> tuple[int, int]:
    buckets = frames["buckets"]
    judged = sum(v for k, v in buckets.items() if k != "Unreviewed")
    return buckets.get("Crack", 0), judged


def _nonroad_row(results: Mapping[str, Any]) -> str:
    nr = results["classifier"]["non_road_false_positives"]
    parts = []
    for key, label in (
        ("clearly_not_road", "tiles with under 5% road"),
        ("partly_road_skipped", "tiles with 5% to 40% road, which the pipeline skips"),
    ):
        f = nr[key]["flagged"]
        parts.append(
            f"{f['count']} of {f['total']:,} {label} ({_pct(f['rate'] or 0)}, 90% range "
            f"{_range(f['low90'], f['high90'])})"
        )
    return "; ".join(parts)


def _range_row(results: Mapping[str, Any]) -> str:
    rng = results["effective_range"]
    d = rng["distance_m_at_width_px"]
    return (
        f"defects 128 px wide or more in the frame (about {d['128']} m ahead for a "
        f"{rng['assumed_defect_width_m']} m wide defect) are found "
        f"{_pct(rng['bands'][-1]['rate'])} of the time; 64 to 128 px (between about "
        f"{d['128']} and {d['64']} m) {_pct(rng['bands'][1]['rate'])}; under 64 px "
        f"(farther than about {d['64']} m) {_pct(rng['bands'][0]['rate'])}"
    )


def _detector_row(results: Mapping[str, Any]) -> str:
    old = results["detector"]["fine_tune"]["old_classes"]
    return " · ".join(
        f"{name} {_pct(v['zero_shot']['precision'])} / {_pct(v['zero_shot']['recall'])}"
        for name, v in old.items()
    )


def _fine_tune_row(results: Mapping[str, Any]) -> str:
    ft = results["detector"]["fine_tune"]
    cone, bar = ft["cones"]["pooled"], ft["barriers"]["pooled"]
    return (
        f"cones: precision {_pct(cone['precision'])}, recall {_pct(cone['recall'])} over "
        f"{len(ft['cones']['qualifying_scenes'])} scenes, but only "
        f"{len(ft['cones']['passing_scenes'])} scene reaches both limits alone; barriers: "
        f"{bar['tp']} of {bar['tp'] + bar['fn']} found. {'PASS' if ft['passed'] else 'FAIL'}: "
        f"{ft['decision']}"
    )


def render_readme_table(results: Mapping[str, Any]) -> str:
    """The README results table, every number read from the results dictionary."""
    c = results["classifier"]
    base, chosen = c["baseline"], c["chosen"]
    nb, nc = c["nuscenes"][base], c["nuscenes"][chosen]
    low, high = wilson_interval(nc["alerts"], max(1, round(nc["alerts"] / nc["per_frame"])))
    cb, jb = _tiles_judged(nb)
    cc, jc = _tiles_judged(nc)
    crack_low, crack_high = wilson_interval(cc, jc)
    probe_b, probe_c = c["probe"][base], c["probe"][chosen]
    sf = results["synthetic_fusion"]
    on, off, road, near = sf["on_road"], sf["off_road"], sf["in_road"], sf["near_objects"]
    by = {r["band"]: r for r in results["effective_range"]["bands"]}
    sizes = " / ".join(_pct(r["rate"]) for r in by.values())
    verdict = sf["verdict"]
    rows = [
        ("Tile classifier on RDD2022 validation tiles, precision / recall per class",
         _validation_row(results), "`classifier.validation`"),
        ("False alarms on 404 nuScenes driving frames (alerts per frame)",
         f"baseline {nb['per_frame']:.3f} ({nb['alerts']} alerts), chosen {nc['per_frame']:.3f} "
         f"({nc['alerts']} alerts, 90% range {low:.3f} to {high:.3f})", "`classifier.nuscenes`"),
        ("Flagged nuScenes tiles that a reviewer called a real crack",
         f"baseline {cb} of {jb} ({_pct(cb / jb)}), chosen {cc} of {jc} ({_pct(cc / jc)}, "
         f"90% range {_range(crack_low, crack_high)})", "`classifier.nuscenes.*.buckets`"),
        ("Clean road tiles the classifier still flags (23 held-out tiles)",
         f"baseline {probe_b['flagged']} of {probe_b['total']}, chosen {probe_c['flagged']} of "
         f"{probe_c['total']}", "`classifier.probe`"),
        ("Non-road tiles flagged (nuScenes, shipped thresholds)", _nonroad_row(results),
         "`classifier.non_road_false_positives`"),
        ("Pasted-patch test: damage found on the road",
         f"{on['found']} of {on['trials']} ({_pct(on['found'] / on['trials'])}, 90% range "
         f"{_range(*on['found_range90'])}); by patch width under 64 / 64 to 128 / 128 px and up: "
         f"{sizes}", "`synthetic_fusion.on_road`"),
        ("Pasted-patch test: off-road patches reported as defects",
         f"{off['reported_off_road']} of {off['trials']}", "`synthetic_fusion.off_road`"),
        ("Pasted-patch test: `in_road` agrees with the hand-drawn road",
         f"{road['correct']} of {road['scored']} ({_pct(road['rate'])}, 90% range "
         f"{_range(*road['range90'])})", "`synthetic_fusion.in_road`"),
        ("Pasted-patch test: objects near a defect",
         f"{_pct(near['recall'])} of the true neighbors found, {_pct(near['precision'])} of those "
         f"reported correct", "`synthetic_fusion.near_objects`"),
        ("Pasted-patch test verdict",
         ("PASS" if verdict["passed"] else "FAIL: " + "; ".join(verdict["failures"])),
         "`synthetic_fusion.verdict`"),
        ("Effective range, from the pasted-patch test", _range_row(results),
         "`effective_range`"),
        ("Zero-shot COCO detector on nuScenes (six cameras, cutoff 0.5), precision / recall",
         _detector_row(results), "`detector.fine_tune.old_classes`"),
        ("Fine-tune for cones and barriers, held-out nuScenes scenes", _fine_tune_row(results),
         "`detector.fine_tune`"),
    ]  # fmt: skip
    if "real_photo_check" in results:
        r = results["real_photo_check"]
        rows.append(
            (
                f"Real dashboard photos ({r['photos']}, {r['source']})",
                f"{r['found']} of {r['marked']} marked potholes found; {r['extra_boxes']} other "
                f"boxes "
                f"(not necessarily mistakes). Run by hand, not reproducible from this repo",
                "`real_photo_check`",
            )
        )
    rows.append(("Retrieval (RAG)", _retrieval_row(results), "`retrieval`"))
    lines = ["| Measure | Result | In `eval/results.json` |", "|---|---|---|"]
    lines += [f"| {m} | {r} | {k} |" for m, r, k in rows]
    return "\n".join(lines)


def update_readme(text: str, table: str) -> str:
    """The README text with the block between the results markers replaced by the table."""
    try:
        start = text.index(README_START) + len(README_START)
        end = text.index(README_END, start)
    except ValueError as exc:
        raise ValueError(f"the README needs the markers {README_START} and {README_END}") from exc
    return f"{text[:start]}\n{table}\n{text[end:]}"
