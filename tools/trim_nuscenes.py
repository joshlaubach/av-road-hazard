"""Trim a nuScenes release's tables to the scenes whose camera images are on disk.

Usage: python -m tools.trim_nuscenes --root data/nuscenes_trainval/raw
           [--version v1.0-trainval] [--out-version v1.0-part1]
Writes ROOT/OUT_VERSION/*.json with only those scenes, so the devkit loads in about a gigabyte
instead of several: NuScenes(version="v1.0-part1", dataroot=ROOT). The images and the full tables
are not touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

Table = list[dict[str, Any]]
TABLES = (
    "attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
    "sample", "sample_annotation", "sample_data", "scene", "sensor", "visibility",
)  # fmt: skip
SHARED = ("sensor", "category", "attribute", "visibility")  # small lookups, kept whole
FRONT = "samples/CAM_FRONT/"


def scenes_with_images(tables: Mapping[str, Table], root: Path) -> set[str]:
    """Names of the scenes whose first front-camera image exists under root."""
    front = {
        d["sample_token"]: d["filename"]
        for d in tables["sample_data"]
        if d["is_key_frame"] and d["filename"].startswith(FRONT)
    }
    return {
        s["name"]
        for s in tables["scene"]
        if s["first_sample_token"] in front and (root / front[s["first_sample_token"]]).is_file()
    }


def trim_tables(tables: Mapping[str, Table], keep_scenes: Collection[str]) -> dict[str, Table]:
    """The tables cut down to the named scenes, with no row pointing at a row that was dropped."""

    def pick(name: str, tokens: set[str]) -> Table:
        return [r for r in tables[name] if r["token"] in tokens]

    scenes = [s for s in tables["scene"] if s["name"] in keep_scenes]
    scene_tokens = {s["token"] for s in scenes}
    samples = [s for s in tables["sample"] if s["scene_token"] in scene_tokens]
    sample_tokens = {s["token"] for s in samples}
    data = [d for d in tables["sample_data"] if d["sample_token"] in sample_tokens]
    annotations = [a for a in tables["sample_annotation"] if a["sample_token"] in sample_tokens]
    log_tokens = {s["log_token"] for s in scenes}
    maps = [
        {**m, "log_tokens": [t for t in m["log_tokens"] if t in log_tokens]} for m in tables["map"]
    ]
    out: dict[str, Table] = {
        "scene": scenes,
        "sample": samples,
        "sample_data": data,
        "sample_annotation": annotations,
        "ego_pose": pick("ego_pose", {d["ego_pose_token"] for d in data}),
        "calibrated_sensor": pick(
            "calibrated_sensor", {d["calibrated_sensor_token"] for d in data}
        ),
        "instance": pick("instance", {a["instance_token"] for a in annotations}),
        "log": pick("log", log_tokens),
        "map": [m for m in maps if m["log_tokens"]],
    }
    out.update({name: list(tables[name]) for name in SHARED})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--out-version", default="v1.0-part1")
    args = parser.parse_args(argv)

    source = args.root / args.version
    tables = {name: json.loads((source / f"{name}.json").read_text("utf-8")) for name in TABLES}
    keep = scenes_with_images(tables, args.root)
    if not keep:
        print(f"no scene under {args.root} has its first front image on disk", file=sys.stderr)
        return 2
    trimmed = trim_tables(tables, keep)
    target = args.root / args.out_version
    target.mkdir(exist_ok=True)
    for name, rows in trimmed.items():
        (target / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    print(f"{len(keep)} scenes -> {target}", {name: len(rows) for name, rows in trimmed.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
