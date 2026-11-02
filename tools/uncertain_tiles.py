"""Pick the tiles the classifier is least sure about, as a blind batch for hand labeling.

Usage: DEFECT_CHECKPOINT=best.pt USE_STUBS=0 python -m tools.uncertain_tiles [--total 100]

Scores every tile the pipeline would score on nuScenes trainval part 1 (road mask and object
filters as in production), on scenes the reviewed test tiles never touched, and keeps tiles whose
class probability is within 0.2 of that class's threshold, below or above. The pick is balanced
across class and side, capped per scene, and seeded. Output goes to the gitignored data/ folder:
PNGs and a blank answer sheet for the annotator, and a separate key with the scores, so the
annotator never sees a score. These tiles are for TRAINING; the 109 reviewed tiles in
annotation_nuscenes{1,2,3} stay the test and must not be mixed in.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from common.config import get_config
from defects.infer import defect_settings
from defects.labels import CODES
from eval.classifier import Alert
from tools.score_table import check_setup

WINDOW = 0.2
DERIVED = Path("data/nuscenes/derived")


@dataclass(frozen=True)
class Candidate:
    scene: str
    alert: Alert
    side: str  # "below" or "above" the class threshold


def in_window(probability: float, threshold: float) -> bool:
    return threshold - WINDOW <= probability < threshold + WINDOW


def side(probability: float, threshold: float) -> str:
    return "below" if probability < threshold else "above"


def pick(
    candidates: Sequence[Candidate], *, total: int, per_scene: int, seed: int
) -> list[Candidate]:
    """Round-robin over (class, side) groups, random within each; a tile once, scenes capped."""
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[(c.alert.name, c.side)].append(c)
    rng = random.Random(seed)
    queues = [rng.sample(groups[key], len(groups[key])) for key in sorted(groups)]
    chosen: list[Candidate] = []
    seen_tiles: set[tuple[str, tuple[int, int, int, int]]] = set()
    per_scene_count: Counter[str] = Counter()
    for queue in itertools.zip_longest(*queues):
        for c in filter(None, queue):
            tile = (c.alert.frame, c.alert.box)
            if tile in seen_tiles or per_scene_count[c.scene] >= per_scene:
                continue
            chosen.append(c)
            seen_tiles.add(tile)
            per_scene_count[c.scene] += 1
            if len(chosen) == total:
                return chosen
    return chosen


def front_frames(
    root: Path, version: str, excluded: set[str], stride: int
) -> list[tuple[str, str, bool]]:
    """(scene name, CAM_FRONT path relative to root, night) for every stride-th frame on disk."""
    from nuscenes.nuscenes import NuScenes

    from eval.detector import is_night

    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    frames = []
    for scene in nusc.scene:
        if scene["name"] in excluded:
            continue
        token, index = scene["first_sample_token"], 0
        while token:
            sample = nusc.get("sample", token)
            rel = nusc.get("sample_data", sample["data"]["CAM_FRONT"])["filename"]
            if index % stride == 0 and (root / rel).is_file():
                frames.append((scene["name"], rel, is_night(scene["description"])))
            token, index = sample["next"], index + 1
    return frames


def candidates_for(frames: Sequence[tuple[str, str, bool]], root: Path) -> list[Candidate]:
    """Every scored tile-class pair within WINDOW of its threshold."""
    from defects.road_crop import MODEL_SUBDIR, RoadSegmenter
    from perception.infer import detect
    from tools.eval_classifier import load_classifier
    from tools.eval_frames import prepare_frames

    cfg = get_config()
    settings = defect_settings(cfg)
    thresholds = dict(zip(CODES, settings.thresholds, strict=True))
    classifier = load_classifier(Path(cfg["defects"]["classifier_checkpoint"]))
    segmenter = RoadSegmenter(Path(cfg["paths"]["checkpoint_dir"]) / MODEL_SUBDIR)
    scenes = {rel: scene for scene, rel, _ in frames}
    contexts = prepare_frames([(rel, night) for _, rel, night in frames], root, segmenter, detect)
    from defects.tile_defects import score_frame_tiles

    found: list[Candidate] = []
    for context in contexts:
        with Image.open(root / context.path) as image:
            frame = np.asarray(image.convert("RGB"))
        scored = score_frame_tiles(
            classifier.model,
            frame,
            list(context.objects),
            classifier.cfg,
            max_object_cover=settings.max_object_cover,
            road_mask=context.mask(),
            min_road_cover=settings.min_road_cover,
        )
        for s in scored:
            if s.probabilities is None:
                continue
            box = tuple(round(v) for v in s.tile.box)
            for name, p in zip(CODES, s.probabilities, strict=True):
                if in_window(p, thresholds[name]):
                    alert = Alert(context.path, box, name, p, context.night, s.road_cover or 1.0)  # type: ignore[arg-type]
                    found.append(Candidate(scenes[context.path], alert, side(p, thresholds[name])))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes_trainval/raw"))
    parser.add_argument("--version", default="v1.0-part1")
    parser.add_argument("--mini-root", type=Path, default=Path("data/nuscenes/raw/v1.0-mini"))
    parser.add_argument("--out-dir", type=Path, default=DERIVED / "annotation_nuscenes4")
    parser.add_argument("--key-dir", type=Path, default=DERIVED / "annotation_key_nuscenes4")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--per-scene", type=int, default=3)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for existing in (args.out_dir / "annotations.csv", args.key_dir / "key.json"):
        if existing.exists():  # stop now, not after the ten-minute scoring
            raise SystemExit(f"{existing} already exists; refusing to overwrite a batch")
    for line in check_setup(get_config()):
        print(line)
    excluded = {s["name"] for s in json.loads((args.mini_root / "scene.json").read_text())}
    frames = front_frames(args.root, args.version, excluded, args.stride)
    print(
        f"{len(frames)} frames from {len({s for s, _, _ in frames})} scenes (mini scenes excluded)"
    )
    pool = candidates_for(frames, args.root)
    chosen = pick(pool, total=args.total, per_scene=args.per_scene, seed=args.seed)
    from tools.eval_frames import export_unreviewed

    count = export_unreviewed(
        [c.alert for c in chosen], args.root, args.out_dir, args.key_dir, {}, prefix="n4"
    )
    print(f"{len(pool)} uncertain tile-class pairs; wrote {count} tiles to {args.out_dir}")
    print("by class and side:", dict(Counter((c.alert.name, c.side) for c in chosen)))
    print("scenes:", len({c.scene for c in chosen}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
