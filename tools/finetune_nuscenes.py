"""Fine-tune a copy of the chosen tile classifier on hand-labeled nuScenes tiles.

Usage: python -m tools.finetune_nuscenes {smoke,cv,final} [--config configs/finetune_nuscenes.yaml]
  smoke  20 steps on all new tiles; checks the checkpoint loads and the RDD2022 guard runs, and
         prints the time per step (the cost of one round is steps x that)
  cv     scene-grouped folds over the new tiles; picks the round by the recall guard; exploratory
  control replay only (no new tiles) for the same steps: does continued training alone move recall?
  final  trains on all new tiles for the chosen round, then scores the reviewed mini frames

The new tiles come from the batch the learner labeled by hand. A Crack tile counts only the damage
kinds it names; every other class is ignored in the loss for that tile. Only the last block and the
head train, each round replays the whole original training set once and adds the new tiles at a
small share. The shipped checkpoint, configs and contracts are not touched; weights go under the
experiment's out_dir, outside git. Plan: reviewed in five rounds before this was written.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from common.config import get_config
from common.hashing import file_sha256
from defects.labels import CODES
from eval.classifier import Box, check_calls

CLASS_OF = {"longitudinal": 0, "transverse": 1, "alligator": 2, "pothole": 3}  # CODES order
CLEAN_CALLS = ("No crack", "Not road")
Pair = tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class NewTile:
    frame: str
    box: Box
    scene: str
    call: str
    target: tuple[float, ...]
    mask: tuple[float, ...]


def targets_for(call: str, kinds: str) -> Pair:
    """(target, mask) per class. A clean tile counts all four; a Crack counts only what it names."""
    if call in CLEAN_CALLS:
        return (0, 0, 0, 0), (1, 1, 1, 1)
    if call != "Crack":
        raise ValueError(f"call {call!r} cannot be used for training")
    words = [w.strip().lower() for w in kinds.replace(";", ",").split(",") if w.strip()]
    if not words:
        raise ValueError("a Crack tile needs at least one kind in the type column")
    unknown = [w for w in words if w not in CLASS_OF]
    if unknown:
        raise ValueError(f"unknown damage kind {unknown}; use {sorted(CLASS_OF)}")
    named = tuple(1.0 if i in {CLASS_OF[w] for w in words} else 0.0 for i in range(len(CODES)))
    return named, named


def load_new_tiles(csv_path: Path, key_path: Path, scene_of: Mapping[str, str]) -> list[NewTile]:
    """The usable labeled tiles (Ambiguous dropped), in key order; blank or unknown calls fail."""
    key = json.loads(key_path.read_text())
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {r["tile_id"]: r for r in csv.DictReader(handle)}
    calls = check_calls(list(key), {t: r["call"].strip() for t, r in rows.items()})
    tiles = []
    for tile_id, entry in key.items():
        if calls[tile_id] == "Ambiguous":
            continue
        if entry["frame"] not in scene_of:
            raise ValueError(f"no scene known for {entry['frame']}")
        target, mask = targets_for(calls[tile_id], rows[tile_id]["type"])
        x1, y1, x2, y2 = (int(v) for v in entry["box"])
        scene = scene_of[entry["frame"]]
        tiles.append(NewTile(entry["frame"], (x1, y1, x2, y2), scene, calls[tile_id], target, mask))
    return tiles


def masked_bce(logits: Any, target: Any, mask: Any) -> Any:
    """Binary cross-entropy over the counted classes only (all counted = the usual mean)."""
    from torch.nn.functional import binary_cross_entropy_with_logits

    loss = binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _under(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def freeze_all_but(model: Any, prefixes: Sequence[str]) -> set[str]:
    """Only parameters under the prefixes keep training; returns their names."""
    names = [n for n, _ in model.named_parameters()]
    for prefix in prefixes:
        if not any(_under(n, [prefix]) for n in names):
            raise ValueError(f"no parameter under {prefix!r}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = _under(name, prefixes)
    return {n for n in names if _under(n, prefixes)}


def set_train_mode(model: Any, prefixes: Sequence[str]) -> None:
    """Train mode for the kept parts only, so frozen BatchNorm statistics cannot drift.

    The root flag stays on because timm reads it for the head dropout.
    """
    model.eval()
    model.training = True
    for prefix in prefixes:
        model.get_submodule(prefix).train()


class RoundSampler:
    """One round: every original tile once, plus n_new new-tile slots spread evenly, shuffled."""

    def __init__(self, n_orig: int, n_tiles: int, share: float, seed: int) -> None:
        self.n_orig, self.n_tiles, self.seed, self.round = n_orig, n_tiles, seed, 0
        self.n_new = round(share * n_orig / (1 - share)) if n_tiles else 0

    def set_round(self, number: int) -> None:
        self.round = number

    def __len__(self) -> int:
        return self.n_orig + self.n_new

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed * 1000 + self.round)
        new: list[int] = []
        while len(new) < self.n_new:
            cycle = list(range(self.n_orig, self.n_orig + self.n_tiles))
            rng.shuffle(cycle)
            new += cycle
        order = [*range(self.n_orig), *new[: self.n_new]]
        rng.shuffle(order)
        return iter(order)


def scene_folds(scenes: Sequence[str], k: int, seed: int) -> list[int]:
    """A fold number per tile; every tile of a scene shares one."""
    unique = sorted(set(scenes))
    random.Random(seed).shuffle(unique)
    fold_of = {scene: i % k for i, scene in enumerate(unique)}
    return [fold_of[s] for s in scenes]


def pick_round(passed: Sequence[Sequence[bool]]) -> int | None:
    """The highest round (1-based) whose recall guard passed in every fold, or None."""
    rounds = len(passed[0])
    for number in range(rounds, 0, -1):
        if all(fold[number - 1] for fold in passed):
            return number
    return None


def pooled_metrics(
    tiles: Sequence[NewTile],
    baseline: np.ndarray,
    tuned: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """False alarms on clean tiles (any class) and recall of each named class, both models.

    Counts come with 90% Wilson ranges. Exploratory: the tiles are few.
    """
    from tools.eval_classifier import with_interval

    cut = np.asarray(thresholds)
    flags = {"baseline": baseline >= cut, "finetuned": tuned >= cut}
    clean = [i for i, t in enumerate(tiles) if t.call in CLEAN_CALLS]
    result: dict[str, Any] = {"clean": {"tiles": len(clean)}}
    for name, flagged in flags.items():
        count = int(flagged[clean].any(axis=1).sum()) if clean else 0
        result["clean"][f"flagged_{name}"] = with_interval(count, len(clean))
    for c, code in enumerate(CODES):
        named = [i for i, t in enumerate(tiles) if t.mask[c] == 1 and t.target[c] == 1]
        result[code] = {"positives": len(named)}
        for name, flagged in flags.items():
            found = int(flagged[named, c].sum()) if named else 0
            result[code][f"found_{name}"] = with_interval(found, len(named))
    return result


def check_pins(recorded: Mapping[str, Any], extra_negatives: Path, tile_index: Path) -> None:
    """Stop if the replay inputs differ from the ones the chosen run recorded."""
    for label, key, path in (
        ("extra negatives", "extra_negatives_sha256", extra_negatives),
        ("tile index", "tile_index_sha256", tile_index),
    ):
        if file_sha256(path) != recorded[key]:
            raise ValueError(f"{label} file {path} differs from the chosen run's recorded hash")


# ---- everything below needs the real data, models and GPU: exercised by the smoke stage ----


def scenes_of_frames(root: Path, version: str) -> dict[str, str]:
    """CAM_FRONT path (relative to root) -> scene name, for every sample in the version."""
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    scenes = {s["token"]: s["name"] for s in nusc.scene}
    return {
        nusc.get("sample_data", s["data"]["CAM_FRONT"])["filename"]: scenes[s["scene_token"]]
        for s in nusc.sample
    }


def resolve_config(
    recorded: dict[str, Any], ctx_cfg: dict[str, Any], exp: dict[str, Any]
) -> dict[str, Any]:
    """The chosen run's own recorded config, with this workspace's paths and the experiment's lr.

    The model and tile settings must equal the local training config's, or the replay would not
    be the recipe the chosen checkpoint was trained with.
    """
    for key in (
        "model_name",
        "head_dropout",
        "drop_path",
        "input_size",
        "tile_size",
        "tile_overlap",
        "clean_per_positive",
        "min_clean_per_image",
        "positive_overlap",
        "negative_overlap",
    ):
        if recorded[key] != ctx_cfg[key]:
            raise ValueError(f"{key} differs between the recorded and the local training config")
    relocated = {
        k: ctx_cfg[k] for k in ("tile_index", "tile_cache_dir", "sample_index", "splits_file")
    }
    return {
        **recorded,
        **relocated,
        "pretrained": False,
        "lr": float(exp["lr"]),
        "extra_negatives": str(Path(exp["extra_negatives"]).resolve()),
    }


@dataclass(frozen=True)
class Setup:
    exp: dict[str, Any]
    cfg: dict[str, Any]
    ctx: Any
    base: Path
    recorded: dict[str, Any]
    device: str


def make_setup(config_path: Path) -> Setup:
    from defects.runtime import select_device
    from tools.eval_classifier import load_context

    exp = yaml.safe_load(config_path.read_text())
    ctx = load_context(Path(exp["train_config"]))
    recorded = json.loads(Path(exp["baseline_checkpoint"]).with_suffix(".json").read_text())
    check_pins(recorded, Path(exp["extra_negatives"]), Path(ctx.cfg["tile_index"]))
    cfg = {
        **resolve_config(recorded["config"], ctx.cfg, exp),
        "experiment_config_sha256": file_sha256(config_path),
    }
    return Setup(exp, cfg, ctx, ctx.base, recorded, select_device())


def original_training_set(setup: Setup) -> tuple[Any, int]:
    """The chosen run's fold-0 training tiles (pool + extra negatives) as a dataset."""
    from defects.splits import read_splits
    from defects.tile_dataset import TileDataset, check_cached, tile_pools
    from defects.tile_index import read_tile_index
    from defects.train import _extra_negatives

    cfg = setup.cfg
    train_ids, val_ids, _ = read_splits((setup.base / cfg["splits_file"]).resolve()).folds[0]
    index = read_tile_index(Path(cfg["tile_index"]))
    seed = int(get_config()["seed"])
    pool, _ = tile_pools(index, train_ids, val_ids, cfg, seed=seed)
    tiles = [*pool, *_extra_negatives(index, train_ids, cfg)]
    check_cached(tiles, Path(cfg["tile_cache_dir"]))
    return TileDataset(tiles, Path(cfg["tile_cache_dir"]), cfg, training=True), len(tiles)


class MixedDataset:
    """Original tiles (mask all ones) followed by the new tiles cropped from their frames."""

    def __init__(
        self, original: Any, tiles: Sequence[NewTile], root: Path, cfg: dict[str, Any]
    ) -> None:
        from defects.dataset import image_transform

        self.original, self.tiles, self.root = original, list(tiles), root
        self.transform = image_transform(cfg, training=True)

    def __len__(self) -> int:
        return len(self.original) + len(self.tiles)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        import torch
        from PIL import Image

        if index < len(self.original):
            image, labels = self.original[index]
            return image, labels, torch.ones(len(CODES))
        tile = self.tiles[index - len(self.original)]
        with Image.open(self.root / tile.frame) as frame:
            crop = frame.convert("RGB").crop(tile.box)
        target = torch.tensor(tile.target, dtype=torch.float32)  # like the original tiles' labels
        mask = torch.tensor(tile.mask, dtype=torch.float32)
        return self.transform(crop), target, mask


def fine_tune(
    setup: Setup, tiles: Sequence[NewTile], out_dir: Path, rounds: int, max_steps: int | None = None
) -> tuple[list[Path], list[float], dict[str, float]]:
    """Fine-tune a copy of the baseline, one checkpoint per round: paths, step times, views."""
    import torch
    from torch.utils.data import DataLoader

    from defects.model import load_checkpoint
    from defects.runtime import forward_logits
    from defects.train import _save_checkpoint, setup_determinism
    from eval.provenance import provenance

    exp, cfg, device = setup.exp, setup.cfg, setup.device
    setup_determinism(int(get_config()["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(Path(exp["baseline_checkpoint"]), cfg, device=device)
    freeze_all_but(model, exp["trainable"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    original, n_orig = original_training_set(setup)
    sampler = RoundSampler(n_orig, len(tiles), float(exp["share"]), int(exp["seed"]))
    loader: DataLoader[Any] = DataLoader(
        cast(Any, MixedDataset(original, tiles, Path(exp["image_root"]), cfg)),
        batch_size=cfg["batch_size"],
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=cfg["num_workers"],
        persistent_workers=bool(cfg.get("persistent_workers", False)) and cfg["num_workers"] > 0,
    )
    paths, step_times = [], []
    views = Counter(i for i in sampler if i >= n_orig)
    stats = {
        "views_min": float(min(views.values(), default=0)),
        "views_max": float(max(views.values(), default=0)),
    }
    for number in range(rounds):
        sampler.set_round(number)
        set_train_mode(model, exp["trainable"])
        total, seen = 0.0, 0
        for step, (images, target, mask) in enumerate(loader):
            started = time.perf_counter()
            loss = masked_bce(
                forward_logits(model, images, device, cfg), target.to(device), mask.to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, seen = total + float(loss.detach()) * len(images), seen + len(images)
            step_times.append(time.perf_counter() - started)
            if step % 500 == 0:
                print(
                    f"round {number + 1} step {step}/{len(loader)} loss {total / seen:.4f}",
                    flush=True,
                )
            if max_steps and step + 1 >= max_steps:
                break
        path = out_dir / f"round{number + 1}.pt"
        prov = {
            **provenance(
                {"base_checkpoint": Path(exp["baseline_checkpoint"])},
                thresholds={},
                repo=Path.cwd(),
            ),
            "config": cfg,
            "class_codes": list(CODES),
            "seed": int(exp["seed"]),
            "round": number + 1,
            "new_tiles": len(tiles),
            **stats,
        }
        _save_checkpoint(model, path, prov, number + 1, {"train_loss": total / max(seen, 1)})
        paths.append(path)
    return paths, step_times, stats


def rdd_guard(setup: Setup, checkpoint: Path) -> dict[str, Any]:
    """The existing RDD2022 validation recall guard of a checkpoint against the baseline."""
    from tools.eval_classifier import evaluate

    base = Path(setup.exp["baseline_checkpoint"])
    report = evaluate({"baseline": base, "ft": checkpoint}, setup.ctx, repo=Path.cwd())
    guard: dict[str, Any] = report["checkpoints"]["ft"]["guard"]
    return guard


def score_new_tiles(setup: Setup, checkpoint: Path, tiles: Sequence[NewTile]) -> np.ndarray:
    """Class probabilities per tile through the deterministic (inference) transform."""
    from PIL import Image

    from defects.model import load_checkpoint
    from tools.eval_classifier import score_crops

    model = load_checkpoint(checkpoint, setup.cfg, device=setup.device).eval()
    crops = []
    for tile in tiles:
        with Image.open(Path(setup.exp["image_root"]) / tile.frame) as frame:
            crops.append(frame.convert("RGB").crop(tile.box))
    return score_crops(model, setup.device, crops, setup.ctx.__class__(setup.cfg, setup.base))


def load_tiles(setup: Setup) -> list[NewTile]:
    exp = setup.exp
    scenes = scenes_of_frames(Path(exp["image_root"]), exp["nuscenes_version"])
    return load_new_tiles(Path(exp["annotations"]), Path(exp["key"]), scenes)


def smoke(setup: Setup) -> None:
    out = Path(setup.exp["out_dir"]) / "smoke"
    tiles = load_tiles(setup)
    print(f"{len(tiles)} usable tiles from {len({t.scene for t in tiles})} scenes")
    paths, times, stats = fine_tune(setup, tiles, out, rounds=1, max_steps=20)
    from tools.eval_classifier import load_classifier

    load_classifier(paths[0])  # the sidecar config is complete enough for production loading
    n_orig = original_training_set(setup)[1]
    sampler = RoundSampler(n_orig, len(tiles), float(setup.exp["share"]), int(setup.exp["seed"]))
    steps = len(sampler) // setup.cfg["batch_size"] + 1
    per_step = sum(times[2:]) / max(len(times) - 2, 1)  # the first steps include warm-up
    print(f"views per new tile per round {stats}")
    minutes = per_step * steps / 60
    print(f"{per_step:.3f} s/step; one round of {steps} steps is about {minutes:.0f} min")
    print("RDD2022 guard on the smoke checkpoint:", json.dumps(rdd_guard(setup, paths[0]))[:300])


def cross_validate(
    setup: Setup, tiles: Sequence[NewTile], max_steps: int | None = None
) -> dict[str, Any]:
    """Scene-grouped folds: train on the others, score the held-out scenes; pick the round."""
    from tools.eval_classifier import shipped_thresholds

    exp, out = setup.exp, Path(setup.exp["out_dir"]) / "cv"
    rounds, k = int(exp["rounds"]), int(exp["folds"])
    folds = scene_folds([t.scene for t in tiles], k, int(exp["seed"]))
    baseline = Path(exp["baseline_checkpoint"])
    held_all: list[NewTile] = []
    base_all: list[np.ndarray] = []
    tuned_all: list[list[np.ndarray]] = [[] for _ in range(rounds)]
    report: dict[str, Any] = {"folds": []}
    for fold in range(k):
        train = [t for t, f in zip(tiles, folds, strict=True) if f != fold]
        held = [t for t, f in zip(tiles, folds, strict=True) if f == fold]
        paths, _, stats = fine_tune(setup, train, out / f"fold{fold}", rounds, max_steps)
        started = time.perf_counter()
        guards = [rdd_guard(setup, path) for path in paths]
        print(f"fold {fold}: guards in {time.perf_counter() - started:.0f}s", flush=True)
        held_all += held
        base_all.append(score_new_tiles(setup, baseline, held))
        for number, path in enumerate(paths):
            tuned_all[number].append(score_new_tiles(setup, path, held))
        report["folds"].append(
            {
                "fold": fold,
                "train_tiles": len(train),
                "held_out_tiles": len(held),
                "held_out_scenes": sorted({t.scene for t in held}),
                "views": stats,
                "guard_passed": [g["passed"] for g in guards],
                "guard_reasons": [g["reasons"] for g in guards],
            }
        )
    chosen = pick_round([f["guard_passed"] for f in report["folds"]])
    report["chosen_round"] = chosen
    thresholds = shipped_thresholds()
    report["pooled_by_round"] = {
        str(number + 1): pooled_metrics(
            held_all, np.concatenate(base_all), np.concatenate(tuned_all[number]), thresholds
        )
        for number in range(rounds)
    }
    report["verdict"] = (
        "rejected: no round passed the recall guard in every fold; the baseline stays"
        if chosen is None
        else f"round {chosen} passed the guard in every fold (exploratory results only)"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def alert_tiles(alerts: Sequence[Any]) -> set[tuple[str, Box]]:
    return {(a.frame, a.box) for a in alerts}


def regression_panel(setup: Setup, checkpoint: Path) -> dict[str, Any]:
    """The reviewed mini frames: baseline vs the fine-tuned model against the pre-set bars."""
    from tools.eval_classifier import load_classifier, prepare_contexts
    from tools.eval_frames import collect_alerts, export_unreviewed
    from tools.score_table import (
        CALLS,
        EXPECTED_ALERTS,
        EXPECTED_CALLS,
        EXPECTED_TILES,
        REVIEWS,
        check_setup,
        count_by_call,
        load_reviews,
    )

    exp = setup.exp
    check_setup(get_config())  # the baseline setup the reviewed totals were measured with
    reviews = Path(exp["reviews_dir"])
    reviewed = load_reviews([(reviews / k, reviews / c) for k, c in REVIEWS])
    root = Path(exp["mini_root"])
    contexts = prepare_contexts(root, exp["mini_version"])
    base = collect_alerts(
        contexts, root, classifier=load_classifier(Path(exp["baseline_checkpoint"]))
    )
    tuned = collect_alerts(contexts, root, classifier=load_classifier(checkpoint))
    base_tiles, tuned_tiles = alert_tiles(base), alert_tiles(tuned)
    reproduced = (len(base), len(base_tiles), count_by_call(base_tiles, reviewed)) == (
        EXPECTED_ALERTS,
        EXPECTED_TILES,
        EXPECTED_CALLS,
    )
    kept = count_by_call(base_tiles & tuned_tiles, reviewed)
    stopped = count_by_call(base_tiles - tuned_tiles, reviewed)
    new_tiles = tuned_tiles - base_tiles
    exported = 0
    if reproduced and new_tiles:
        exported = export_unreviewed(
            [a for a in tuned if (a.frame, a.box) in new_tiles],
            root,
            Path(exp["export_dir"]),
            Path(exp["export_key_dir"]),
            reviewed,
            prefix="n5",
        )
    return {
        "baseline_reproduced": reproduced,
        "baseline_alerts": len(base),
        "finetuned_alerts": len(tuned),
        "baseline_tiles_still_alerting": kept,
        "baseline_tiles_stopped": stopped,
        "new_alert_tiles": len(new_tiles),
        "new_unreviewed_tiles_exported": exported,
        "calls": list(CALLS),
    }


def final(setup: Setup, tiles: Sequence[NewTile], max_steps: int | None = None) -> dict[str, Any]:
    """Train on every new tile for the chosen round, then run the regression panel once."""
    out = Path(setup.exp["out_dir"])
    cv_report = json.loads((out / "cv" / "report.json").read_text())
    chosen = cv_report["chosen_round"]
    if chosen is None:
        raise SystemExit("cv rejected the fine-tune (no round passed everywhere); nothing to train")
    paths, _, stats = fine_tune(setup, tiles, out / "final", int(chosen), max_steps)
    guard = rdd_guard(setup, paths[-1])
    report: dict[str, Any] = {"round": chosen, "views": stats, "guard": guard}
    if not guard["passed"]:  # before any result from this model is shown or exported
        report["verdict"] = "rejected: the final model failed the RDD2022 guard; baseline stays"
    else:
        report["panel"] = regression_panel(setup, paths[-1])
        report["verdict"] = "guard passed; read the panel against the pre-set bars"
    (out / "final").mkdir(parents=True, exist_ok=True)
    (out / "final" / "report.json").write_text(json.dumps(report, indent=1))
    return report


def control(setup: Setup, max_steps: int | None = None) -> dict[str, Any]:
    """Replay only, no new tiles: does continued training alone move the RDD2022 recall?"""
    out = Path(setup.exp["out_dir"]) / "control"
    paths, _, _ = fine_tune(setup, [], out, rounds=1, max_steps=max_steps)
    guard = rdd_guard(setup, paths[0])
    report = {"steps": max_steps, "new_tiles": 0, "guard": guard}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "cv", "final", "control"))
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_nuscenes.yaml"))
    parser.add_argument("--max-steps", type=int, default=None, help="dry run: steps per round")
    args = parser.parse_args(argv)
    setup = make_setup(args.config)
    if args.stage == "smoke":
        smoke(setup)
        return 0
    if args.stage == "control":
        print(json.dumps(control(setup, args.max_steps), indent=1))
        return 0
    tiles = load_tiles(setup)
    run = cross_validate if args.stage == "cv" else final
    result = run(setup, tiles, args.max_steps)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
