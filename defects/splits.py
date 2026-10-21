"""Persisted group-safe final test, outer folds and inner validation partitions."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

Fold = tuple[list[str], list[str], list[str]]


@dataclass
class Splits:
    test: list[str]
    folds: list[Fold]

    def validate(self, ids: Sequence[str]) -> None:
        """Reject corrupt membership before any model sees the data."""
        universe = set(ids)
        if len(universe) != len(ids) or not self.test or len(self.folds) < 2:
            raise ValueError("Need unique ids, a test set and at least two folds")
        counts: dict[str, int] = defaultdict(int)
        for fold in self.folds:
            parts = [self.test, *fold]
            seen: set[str] = set()
            for part in parts:
                if not part or len(set(part)) != len(part) or seen.intersection(part):
                    raise ValueError("Empty, duplicate or overlapping split membership")
                seen.update(part)
            if seen != universe:
                raise ValueError("Split ids do not match dataset")
            for item in fold[2]:
                counts[item] += 1
        if set(counts) != universe - set(self.test) or any(n != 1 for n in counts.values()):
            raise ValueError("Each non-test id must be held out exactly once")


def _take(groups: list[list[str]], fraction: float) -> tuple[list[list[str]], list[list[str]]]:
    """Choose a nonempty prefix nearest the requested item count, leaving one group."""
    target = sum(map(len, groups)) * fraction
    total = 0
    choices: list[tuple[float, int]] = []
    for i, group in enumerate(groups[:-1], 1):
        total += len(group)
        choices.append((abs(total - target), i))
    if not choices:
        raise ValueError("Insufficient independent groups for validation")
    cut = min(choices)[1]
    return groups[:cut], groups[cut:]


def make_splits(
    items: Sequence[str],
    group_fn: Callable[[str], str],
    *,
    test_fraction: float = 0.15,
    n_folds: int = 5,
    val_fraction: float = 0.10,
    seed: int,
) -> Splits:
    if len(set(items)) != len(items):
        raise ValueError("Duplicate sample ids")
    if not 0 < test_fraction < 1 or not 0 < val_fraction < 1 or n_folds < 2:
        raise ValueError("Invalid split fractions or fold count")
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in sorted(items):
        grouped[group_fn(item)].append(item)
    groups = [grouped[key] for key in sorted(grouped)]
    if len(groups) < n_folds + 2:
        raise ValueError("Insufficient independent groups for test, folds and inner validation")
    rng = random.Random(seed)
    rng.shuffle(groups)
    test_groups, rest = _take(groups, test_fraction)
    if len(rest) < n_folds or len(rest) - (len(rest) + n_folds - 1) // n_folds < 2:
        raise ValueError("Insufficient groups after final test holdout")
    # Greedy load balancing keeps whole groups and approximates item fractions.
    buckets: list[list[list[str]]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for group in sorted(rest, key=len, reverse=True):
        k = min(range(n_folds), key=lambda i: sizes[i])
        buckets[k].append(group)
        sizes[k] += len(group)
    folds: list[Fold] = []

    def flatten(gs: list[list[str]]) -> list[str]:
        return sorted(item for group in gs for item in group)

    for k in range(n_folds):
        pool = [group for i, bucket in enumerate(buckets) if i != k for group in bucket]
        rng.shuffle(pool)
        val, train = _take(pool, val_fraction)
        folds.append((flatten(train), flatten(val), flatten(buckets[k])))
    splits = Splits(flatten(test_groups), folds)
    splits.validate(items)
    return splits


def write_splits(splits: Splits, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(splits), indent=2) + "\n")


def read_splits(path: Path) -> Splits:
    raw = json.loads(path.read_text())
    splits = Splits(raw["test"], [tuple(fold) for fold in raw["folds"]])
    if splits.folds:
        splits.validate(
            [*splits.test, *splits.folds[0][0], *splits.folds[0][1], *splits.folds[0][2]]
        )
    else:
        raise ValueError("Missing folds")
    return splits
