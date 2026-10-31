"""Pick the scenes and folds for the detector test and write them down (ADR 0015).

Usage: python -m tools.plan_detector_scenes --root data/nuscenes_trainval/raw
           [--version v1.0-part1] [--folds 5] --out configs/detector_scenes.yaml
The rule uses only the metadata and is fixed before any model is trained: a scene is chosen if it
has at least LIMIT barriers or LIMIT traffic cones that are visible enough to count (the same
visibility cutoff training and scoring use), plus the first PLAIN scenes by name that have neither,
to check the detector does not invent them. Chosen scenes are ranked richest first and dealt to the
folds in turn, so the richest scenes land in different folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from eval.detector import MIN_VISIBILITY_LEVEL
from perception.finetune import deal_into_folds
from perception.labels import nuscenes_label

LIMIT = 20
PLAIN = 10


def visible_counts(folder: Path) -> dict[str, tuple[int, int]]:
    """Scene name to (barrier, traffic cone) annotations visible enough to count."""

    def load(name: str) -> list[dict[str, str]]:
        return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    scene_name = {s["token"]: s["name"] for s in load("scene")}
    sample_scene = {s["token"]: scene_name[s["scene_token"]] for s in load("sample")}
    category = {c["token"]: c["name"] for c in load("category")}
    instance_label = {
        i["token"]: nuscenes_label(category[i["category_token"]]) for i in load("instance")
    }
    counts = {name: [0, 0] for name in scene_name.values()}
    for a in load("sample_annotation"):
        column = {"barrier": 0, "traffic_cone": 1}.get(instance_label[a["instance_token"]] or "")
        if column is not None and int(a["visibility_token"]) >= MIN_VISIBILITY_LEVEL:
            counts[sample_scene[a["sample_token"]]][column] += 1
    return {name: (b, c) for name, (b, c) in counts.items()}


def choose_scenes(
    counts: Mapping[str, tuple[int, int]], limit: int = LIMIT, plain: int = PLAIN
) -> list[str]:
    """The test scenes, ranked richest first (then by name), plain-road scenes last."""
    rich = [s for s, (b, c) in counts.items() if b >= limit or c >= limit]
    rich.sort(key=lambda s: (-sum(counts[s]), s))
    empty = sorted(s for s, (b, c) in counts.items() if b == 0 and c == 0)
    return rich + empty[:plain]


def plan_text(ranked: list[str], folds: int, notes: Mapping[str, str]) -> str:
    """The plan file: a table of scene name to fold number, dealt in ranked order."""
    plan = deal_into_folds(ranked, folds)
    rows = [f"  {s}: {plan[s]}" + (f"  # {notes[s]}" if s in notes else "") for s in ranked]
    header = (
        "# Scenes and folds for the ADR 0015 detector test, made by tools.plan_detector_scenes.\n"
        "# Written before any model was trained; do not edit by hand. Scenes are ranked by\n"
        "# barriers plus cones, richest first, and dealt to the folds in turn.\n"
    )
    return header + f"folds: {folds}\nscenes:\n" + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-part1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    folder = args.root / args.version
    counts = visible_counts(folder)
    ranked = choose_scenes(counts)
    log_location = {
        x["token"]: x["location"] for x in json.loads((folder / "log.json").read_text("utf-8"))
    }
    where = {
        s["name"]: log_location[s["log_token"]]
        for s in json.loads((folder / "scene.json").read_text("utf-8"))
    }
    notes = {s: f"{where[s]}, {counts[s][0]} barriers, {counts[s][1]} cones" for s in ranked}
    args.out.write_text(plan_text(ranked, args.folds, notes), encoding="utf-8")
    print(f"{len(ranked)} scenes in {args.folds} folds -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
