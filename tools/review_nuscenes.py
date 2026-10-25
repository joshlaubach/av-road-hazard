"""Inspect a tile checkpoint's near-threshold scores on local nuScenes CAM_FRONT images.

Usage: python -m tools.review_nuscenes --images-dir data/nuscenes/raw/samples/CAM_FRONT
       --checkpoint ../runs/tile-fold0/fold0/best.pt --review-config configs/defects_review.yaml
       --out-dir data/nuscenes/derived/threshold_review

The contact sheet contains nuScenes pixels. Keep it local in gitignored data/.
This is qualitative error analysis, not an accuracy estimate: nuScenes has no defect labels.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from common.config import get_config
from defects.dataset import image_transform
from defects.labels import CODES
from defects.patches import Tile, tile_grid
from defects.runtime import forward_logits, select_device
from defects.train import load_checkpoint

OUTLINE = (255, 0, 255)


@dataclass(frozen=True)
class ReviewHit:
    """The strongest tile and class score on one image."""

    image: Path
    score: float
    class_code: str
    tile: Tile
    image_size: tuple[int, int]


def select_hits(
    hits: list[ReviewHit], *, threshold: float, margin: float, per_side: int, seed: int
) -> dict[str, list[ReviewHit]]:
    """Sample frames on either side of the cutoff, regardless of input order."""
    if not 0 <= threshold - margin < threshold < threshold + margin <= 1:
        raise ValueError("score margin must fit on both sides of the threshold within [0, 1]")
    if per_side < 1:
        raise ValueError("samples_per_side must be positive")
    rng = random.Random(seed)
    bands = {
        "below": sorted(
            (hit for hit in hits if threshold - margin <= hit.score < threshold),
            key=lambda hit: str(hit.image),
        ),
        "above": sorted(
            (hit for hit in hits if threshold <= hit.score < threshold + margin),
            key=lambda hit: str(hit.image),
        ),
    }
    return {
        name: sorted(rng.sample(pool, min(len(pool), per_side)), key=lambda hit: hit.score)
        for name, pool in bands.items()
    }


def annotate_frame(image: Image.Image, hit: ReviewHit, *, max_size: tuple[int, int]) -> Image.Image:
    """Draw the triggering tile in frame coordinates after resizing for display."""
    if image.size != hit.image_size:
        raise ValueError(f"image size changed for {hit.image}")
    shown = image.convert("RGB").copy()
    shown.thumbnail(max_size, Image.Resampling.LANCZOS)
    sx, sy = shown.width / image.width, shown.height / image.height
    x1, y1, x2, y2 = hit.tile.box
    ImageDraw.Draw(shown).rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=OUTLINE, width=3)
    return shown


def score_image(
    image_path: Path,
    model: Any,
    cfg: dict[str, Any],
    *,
    tile_size: int,
    overlap: float,
    device: str,
) -> ReviewHit:
    """Keep the highest sigmoid score over every tile and defect class."""
    import torch

    transform = image_transform(cfg, training=False)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    tiles = tile_grid(image.size, tile_size, overlap)
    best_score = -1.0
    best_tile = tiles[0]
    best_code = CODES[0]
    with torch.inference_mode():
        for start in range(0, len(tiles), int(cfg["batch_size"])):
            chunk = tiles[start : start + int(cfg["batch_size"])]
            batch = torch.stack([transform(image.crop(tile.box)) for tile in chunk])
            probabilities = forward_logits(model, batch, device, cfg).sigmoid()
            flat = int(probabilities.argmax().item())
            tile_index, class_index = divmod(flat, len(CODES))
            score = float(probabilities.flatten()[flat].item())
            if score > best_score:
                best_score = score
                best_tile = chunk[tile_index]
                best_code = CODES[class_index]
    return ReviewHit(image_path, best_score, best_code, best_tile, image.size)


def render_contact_sheet(groups: dict[str, list[ReviewHit]], path: Path) -> None:
    """Show each selected frame beside its strongest tile, with score and class code."""
    rows = [(band, hit) for band in ("below", "above") for hit in groups[band]]
    row_height = 340
    sheet = Image.new("RGB", (940, max(1, len(rows)) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    if not rows:
        draw.text((20, 20), "No images scored within the review bands.", fill="black")
    for index, (band, hit) in enumerate(rows):
        y = index * row_height
        draw.text(
            (20, y + 15),
            f"{band.upper()}  {hit.class_code}  score={hit.score:.4f}  {hit.image.name}",
            fill="black",
        )
        with Image.open(hit.image) as opened:
            image = opened.convert("RGB")
        frame = annotate_frame(image, hit, max_size=(600, 270))
        sheet.paste(frame, (20, y + 50))
        crop = image.crop(hit.tile.box)
        crop.thumbnail((260, 260), Image.Resampling.LANCZOS)
        sheet.paste(crop, (650, y + 50))
        draw.text((650, y + 315), f"tile row={hit.tile.row}, col={hit.tile.col}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _hit_json(hit: ReviewHit, images_dir: Path) -> dict[str, Any]:
    return {
        "image": hit.image.relative_to(images_dir).as_posix(),
        "score": hit.score,
        "class_code": hit.class_code,
        "tile_box": list(hit.tile.box),
        "tile_row": hit.tile.row,
        "tile_col": hit.tile.col,
        "image_size": list(hit.image_size),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    images_dir = args.images_dir.resolve()
    paths = sorted(images_dir.glob("*.jpg"))
    if not paths:
        parser.error(f"no JPEG images found in {images_dir}")
    checkpoint = args.checkpoint.resolve()
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    if sidecar.get("input") != "tiles":
        parser.error("checkpoint was not trained on tiles")
    cfg: dict[str, Any] = sidecar["config"]
    tiling = sidecar["tile_params"]
    review: dict[str, Any] = yaml.safe_load(args.review_config.read_text())
    threshold = float(cfg["threshold"])
    margin = float(review["score_margin"])
    per_side = int(review["samples_per_side"])
    seed = int(get_config()["seed"])
    device = select_device()
    model = load_checkpoint(checkpoint, cfg, device=device).eval()

    hits = []
    for index, image_path in enumerate(paths, start=1):
        hits.append(
            score_image(
                image_path,
                model,
                cfg,
                tile_size=int(tiling["tile_size"]),
                overlap=float(tiling["overlap"]),
                device=device,
            )
        )
        if index % 25 == 0 or index == len(paths):
            print(f"scored {index}/{len(paths)} images", flush=True)

    groups = select_hits(hits, threshold=threshold, margin=margin, per_side=per_side, seed=seed)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    render_contact_sheet(groups, out_dir / "contact_sheet.png")
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "threshold": threshold,
        "score_margin": margin,
        "seed": seed,
        "images_scored": len(paths),
        "available_below": sum(threshold - margin <= hit.score < threshold for hit in hits),
        "available_above": sum(threshold <= hit.score < threshold + margin for hit in hits),
        "selected": {
            band: [_hit_json(hit, images_dir) for hit in group] for band, group in groups.items()
        },
    }
    (out_dir / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {out_dir / 'contact_sheet.png'} and {out_dir / 'review.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
