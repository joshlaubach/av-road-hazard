"""Where do the chosen classifier's reviewed alerts sit relative to each class threshold?

Usage: DEFECT_CHECKPOINT=best.pt USE_STUBS=0 python -m tools.score_table --nuscenes-root DIR

Phase 1 of the cascade idea: before building a second-stage model for tiles just above a class
threshold, count the human calls (Crack, No crack, ...) for alerts in the band [t, t + 0.2) and
above it. The band is fixed in advance. Labels exist only for flagged tiles, so this looks at the
precision side only. Stops before printing a table if the setup differs from the run it reproduces.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from common.config import get_config
from common.hashing import file_sha256
from defects.infer import defect_settings
from defects.labels import CODES
from eval.classifier import REVIEW_CALLS, Alert, Box
from perception.infer import detector_settings, tuned_settings
from tools.eval_classifier import (
    Reviewed,
    load_classifier,
    load_reviewed,
    model_files,
    prepare_contexts,
)

BAND = 0.2
CALLS = (*REVIEW_CALLS, "Unreviewed")
Tile = tuple[str, Box]

# The setup the historical 53-alert run is expected to have used (defaults in configs/default.yaml).
CLASSIFIER_PREFIX = "762f4b38"
DETECTOR_NAME = "fasterrcnn_mobilenet_v3_large_fpn"
DETECTOR_SHA256 = "f7731b02ff6365453ad4fb85c008f4dc86dea63a2d4d77377f986fa54b797e21"
DETECTOR_THRESHOLD = 0.5
EXPECTED_ALERTS, EXPECTED_TILES = 53, 52
EXPECTED_CALLS = {"Crack": 7, "No crack": 36, "Ambiguous": 8, "Not road": 1, "Unreviewed": 0}

# Review batches in chronological order: a later call overrides an earlier one for the same tile.
# Batch 1's human calls are in annotation_nuscenes1 (answers.csv is a placeholder, answers_ai.csv
# is AI-filled).
REVIEWS = (
    ("annotation_key_nuscenes1/key.json", "annotation_nuscenes1/annotations_nuscenes1.csv"),
    ("annotation_key_nuscenes2/key.json", "annotation_key_nuscenes2/answers_human.csv"),
    ("annotation_key_nuscenes3/key.json", "annotation_key_nuscenes3/answers_human.csv"),
)


def in_band(score: float, threshold: float) -> bool:
    return threshold <= score < threshold + BAND


def load_reviews(pairs: Iterable[tuple[Path, Path]]) -> Reviewed:
    """The human call per (frame, box); later pairs override earlier ones."""
    merged: Reviewed = {}
    for key_file, calls_csv in pairs:
        merged = {**merged, **load_reviewed(key_file, calls_csv)}
    return merged


def class_table(
    alerts: Sequence[Alert], reviewed: Reviewed, thresholds: Mapping[str, float]
) -> dict[str, dict[str, dict[str, int]]]:
    """Per class: class alerts in the band and above it, counted by human call."""
    table = {
        n: {"in_band": dict.fromkeys(CALLS, 0), "above": dict.fromkeys(CALLS, 0)} for n in CODES
    }
    for a in alerts:
        where = "in_band" if in_band(a.confidence, thresholds[a.name]) else "above"
        table[a.name][where][reviewed.get((a.frame, a.box), "Unreviewed")] += 1
    return table


def in_band_only_tiles(
    alerts: Sequence[Alert], thresholds: Mapping[str, float], *, d00_above: bool = False
) -> set[Tile]:
    """Tiles whose every class alert is in the band: they would stop alerting if the band were
    suppressed. d00_above counts D00 alerts as above the band (D00's band covers every score)."""
    flags: dict[Tile, list[bool]] = defaultdict(list)
    for a in alerts:
        banded = in_band(a.confidence, thresholds[a.name]) and not (d00_above and a.name == "D00")
        flags[(a.frame, a.box)].append(banded)
    return {tile for tile, banded in flags.items() if all(banded)}


def count_by_call(tiles: Iterable[Tile], reviewed: Reviewed) -> dict[str, int]:
    counts = Counter(reviewed.get(tile, "Unreviewed") for tile in tiles)
    return {call: counts[call] for call in CALLS}


def format_report(
    alerts: Sequence[Alert], reviewed: Reviewed, thresholds: Mapping[str, float]
) -> str:
    lines: list[str] = []
    for name, table in class_table(alerts, reviewed, thresholds).items():
        t = thresholds[name]
        lines.append(f"\n{name}  threshold {t}, band [{t}, {t + BAND:.2f})")
        if t + BAND > 1.0:
            lines.append("  uninformative: every flagged score is in the band")
        for where, counts in table.items():
            lines.append(f"  {where:8} {counts}")
    tiles = {(a.frame, a.box) for a in alerts}
    lines.append("\nTile level (HYPOTHETICAL MAXIMUM: tiles that stop alerting if the band is cut)")
    stops = {}
    for label, d00_above in (("D00 as is", False), ("D00 above band", True)):
        stops[label] = in_band_only_tiles(alerts, thresholds, d00_above=d00_above)
        lines.append(f"  {label}: stops {count_by_call(stops[label], reviewed)}")
        lines.append(f"  {label}: keeps {count_by_call(tiles - stops[label], reviewed)}")
    differ = sorted(stops["D00 as is"] ^ stops["D00 above band"])
    lines.append(f"  tiles that differ between the two: {differ}")
    return "\n".join(lines)


def check_setup(cfg: dict) -> list[str]:  # type: ignore[type-arg]
    """Header lines naming what produced the alerts; SystemExit if it is not the expected setup."""
    problems: list[str] = []
    if cfg["pipeline"]["use_stubs"]:
        problems.append("USE_STUBS must be 0")
    if tuned_settings(cfg) is not None:
        problems.append("the fine-tuned detector is configured; unset DETECTOR_CHECKPOINT")
    name, threshold = detector_settings(cfg)
    if (name, threshold) != (DETECTOR_NAME, DETECTOR_THRESHOLD):
        problems.append(f"detector is {name} at {threshold}, expected {DETECTOR_NAME} at 0.5")
    files = model_files()
    detector_sha = file_sha256(files["detector"])
    if detector_sha != DETECTOR_SHA256:
        problems.append(f"detector weights hash {detector_sha} differs from the pinned one")
    checkpoint = cfg["defects"]["classifier_checkpoint"]
    classifier_sha = file_sha256(Path(checkpoint)) if checkpoint else ""
    if not classifier_sha.startswith(CLASSIFIER_PREFIX):
        problems.append(
            f"classifier hash {classifier_sha or 'unset'} does not start {CLASSIFIER_PREFIX}"
        )
    if problems:
        raise SystemExit("Stopped: " + "; ".join(problems))
    return [
        f"classifier {classifier_sha}",
        f"detector {name} score>={threshold} sha256 {detector_sha}",
        *(f"{k} sha256 {file_sha256(p)}" for k, p in files.items() if k.startswith("road/")),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--nuscenes-version", default="v1.0-mini")
    parser.add_argument("--reviews-dir", type=Path, default=Path("data/nuscenes/derived"))
    args = parser.parse_args(argv)
    cfg = get_config()
    header = check_setup(cfg)
    pairs = [(args.reviews_dir / k, args.reviews_dir / c) for k, c in REVIEWS]
    for key_file, calls_csv in pairs:
        batch = load_reviewed(key_file, calls_csv)
        cracks = sum(call == "Crack" for call in batch.values())
        header.append(
            f"{key_file.parent.name}: {len(batch)} tiles, {cracks} Crack, "
            f"keys {file_sha256(key_file)[:12]} calls {file_sha256(calls_csv)[:12]}"
        )
    reviewed = load_reviews(pairs)
    thresholds = dict(zip(CODES, defect_settings(cfg).thresholds, strict=True))
    from tools.eval_frames import collect_alerts

    contexts = prepare_contexts(args.nuscenes_root, args.nuscenes_version)
    alerts = collect_alerts(
        contexts,
        args.nuscenes_root,
        classifier=load_classifier(Path(cfg["defects"]["classifier_checkpoint"])),
    )
    tiles = {(a.frame, a.box) for a in alerts}
    seen = count_by_call(tiles, reviewed)
    print(
        "\n".join(
            [
                *header,
                f"thresholds {thresholds}",
                f"alerts {len(alerts)}, tiles {len(tiles)}, calls {seen}",
            ]
        )
    )
    if (len(alerts), len(tiles), seen) != (EXPECTED_ALERTS, EXPECTED_TILES, EXPECTED_CALLS):
        print(
            f"Totals differ from the expected {EXPECTED_ALERTS} alerts, {EXPECTED_TILES} tiles, "
            f"{EXPECTED_CALLS}. Not printing the table; find out why first."
        )
        return 1
    print(format_report(alerts, reviewed, thresholds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
