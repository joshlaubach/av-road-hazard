"""The pure parts of the sealed test-split scoring: protocol, tile selection, batches and numbers.

tools/eval_test_split.py does the guards and the file reads; this module decides which tiles are
scored, scores them in batches (only one image and one batch of crops in memory at a time), and
turns the probabilities into per-class numbers. Nothing here reads a file or loads a model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from defects.labels import CODES
from defects.tile_index import TileIndex, TileRecord, evaluation_pool, training_pool
from eval.classifier import class_report, flags, wilson_interval

MEASUREMENTS = ("pooled", "all_tiles")
REQUIRED_FIELDS = (
    "checkpoint_sha256",
    "splits_sha256",
    "tile_index_sha256",
    "thresholds",
    "pooled_sampling",
    "measurements",
    "metrics",
    "rule",
    "registered_date",
)
DEFAULT_BATCH = 256

Scorer = Callable[[list[Any]], np.ndarray]  # crops in, (crops, classes) probabilities out
OpenImage = Callable[[str], Any]  # image id in, RGB picture out


@dataclass(frozen=True)
class Protocol:
    """The frozen settings from eval/test_split_protocol.json; thresholds in CODES order."""

    checkpoint_sha256: str
    splits_sha256: str
    tile_index_sha256: str
    thresholds: tuple[float, ...]
    seed: int
    clean_per_positive: float
    min_clean: int


def parse_protocol(raw: Mapping[str, Any]) -> Protocol:
    """The protocol's settings; a missing field or a wrong class list is an error."""
    missing = [name for name in REQUIRED_FIELDS if name not in raw]
    if missing:
        raise ValueError(f"protocol is missing {missing}")
    thresholds = raw["thresholds"]
    if set(thresholds) != set(CODES):
        raise ValueError(f"protocol thresholds must name exactly {list(CODES)}")
    if list(raw["measurements"]) != list(MEASUREMENTS):
        raise ValueError(f"protocol measurements must be {list(MEASUREMENTS)}")
    sampling = raw["pooled_sampling"]
    return Protocol(
        checkpoint_sha256=str(raw["checkpoint_sha256"]),
        splits_sha256=str(raw["splits_sha256"]),
        tile_index_sha256=str(raw["tile_index_sha256"]),
        thresholds=tuple(float(thresholds[code]) for code in CODES),
        seed=int(sampling["seed"]),
        clean_per_positive=float(sampling["clean_per_positive"]),
        min_clean=int(sampling["min_clean_per_image"]),
    )


def hash_mismatches(protocol: Protocol, actual: Mapping[str, str]) -> list[str]:
    """The names of the sealed files whose SHA-256 is not the one the protocol records."""
    expected = {
        "checkpoint": protocol.checkpoint_sha256,
        "splits": protocol.splits_sha256,
        "tile_index": protocol.tile_index_sha256,
    }
    return [name for name, digest in expected.items() if actual.get(name) != digest]


def scored_tiles(
    index: TileIndex, test_ids: Sequence[str], protocol: Protocol
) -> dict[str, list[TileRecord]]:
    """The tiles of each measurement. all_tiles holds every non-ignored tile, and pooled is a
    subset of it, so each tile is cropped and scored once."""
    ids = sorted(test_ids)
    return {
        "pooled": training_pool(
            index,
            ids,
            clean_per_positive=protocol.clean_per_positive,
            min_clean=protocol.min_clean,
            seed=protocol.seed,
        ),
        "all_tiles": evaluation_pool(index, ids),
    }


def _crop_batches(
    tiles: Sequence[TileRecord],
    index: TileIndex,
    open_image: OpenImage,
    batch_size: int,
) -> Iterator[list[tuple[TileRecord, Any]]]:
    """Crops in batches of batch_size, opening each image once; tiles must be grouped by image."""
    current: str | None = None
    picture: Any = None
    batch: list[tuple[TileRecord, Any]] = []
    for tile in tiles:
        if tile.image_id != current:
            current = tile.image_id
            picture = open_image(current)
            expected = index.images[current].size
            if tuple(picture.size) != tuple(expected):
                raise ValueError(
                    f"image size {picture.size} does not match the tile index {expected}"
                )
        batch.append((tile, picture.crop(tile.box)))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def score_tiles(
    tiles: Sequence[TileRecord],
    index: TileIndex,
    open_image: OpenImage,
    scorer: Scorer,
    *,
    batch_size: int = DEFAULT_BATCH,
) -> dict[str, np.ndarray]:
    """Class probabilities for every tile, keyed by tile id, scored in batches of batch_size."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    probs: dict[str, np.ndarray] = {}
    for batch in _crop_batches(tiles, index, open_image, batch_size):
        rows = np.asarray(scorer([crop for _, crop in batch]), dtype=float)
        if rows.shape != (len(batch), len(CODES)):
            raise ValueError(
                f"scorer returned shape {rows.shape}, expected {(len(batch), len(CODES))}"
            )
        probs |= {tile.tile_id: row for (tile, _), row in zip(batch, rows, strict=True)}
    return probs


def _number(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _interval(successes: int, total: int) -> dict[str, float] | None:
    """Wilson 90% interval for a rate; None when there is no denominator to divide by."""
    if total == 0:
        return None
    low, high = wilson_interval(successes, total)
    return {"low90": low, "high90": high}


def measure(
    tiles: Sequence[TileRecord],
    probs: Mapping[str, np.ndarray],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Per class: precision, recall, AUROC, flagged, positives, tiles and 90% Wilson intervals."""
    truth = np.asarray([t.labels for t in tiles], dtype=float).reshape(len(tiles), len(CODES))
    scores = np.asarray([probs[t.tile_id] for t in tiles], dtype=float).reshape(
        len(tiles), len(CODES)
    )
    report = class_report(truth, scores, thresholds, CODES)
    hits = (flags(scores, thresholds) & (truth > 0.5)).sum(axis=0)
    classes: dict[str, Any] = {}
    for k, code in enumerate(CODES):
        row = report[code]
        flagged, positives = int(row["flagged"]), int(row["positives"])
        classes[code] = {
            "precision": _number(row["precision"]) if flagged else None,
            "recall": _number(row["recall"]) if positives else None,
            "auroc": _number(row["auroc"]),
            "flagged": flagged,
            "positives": positives,
            "tiles": len(tiles),
            "precision_interval": _interval(int(hits[k]), flagged),
            "recall_interval": _interval(int(hits[k]), positives),
        }
    return {"tiles": len(tiles), "classes": classes}
