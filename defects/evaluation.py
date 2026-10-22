"""Country-aware evaluation of the defect classifier, from a saved predictions artifact.

Purpose: a pooled score hides how a model behaves in each country (RDD2022 positive rates run from
4% to 83% by country), and a model that mostly recognises the country can score well without
looking at the road. This module reports metrics per country and per class, with the counts
behind them, and compares the model with a country-prior baseline that never sees a pixel.

Predictions are a model-agnostic artifact (ids, labels, probabilities, meta), so the same report
works for a whole-image model and a tile model. Ids start with the country folder, for tiles
"<image id>#r<row>c<col>". Metrics that are undefined are None with a reason in `notes`; nothing
is silently set to 0 or dropped. Definitions: precision, recall and F1 use probability >=
threshold; F1 = 2tp / (2tp + fp + fn); AUROC uses average ranks for ties; average precision is
the step-wise sum over distinct score thresholds (not interpolated).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from defects.labels import CODES
from defects.voc import ALL

Array = npt.NDArray[np.float64]
METRICS = ("auroc", "average_precision", "precision", "recall", "f1")


def country_of_id(sample_id: str) -> str:
    """The country folder an image or tile id starts with, or ALL when it has none."""
    return sample_id.split("/", 1)[0] if "/" in sample_id else ALL


@dataclass
class Predictions:
    """One row per image or tile: id, 0/1 labels and probabilities per class code, run metadata."""

    ids: Sequence[str]
    labels: npt.NDArray[Any]
    probs: npt.NDArray[Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.ids) == 0:
            raise ValueError("predictions are empty")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("predictions contain duplicate ids")
        expected = (len(self.ids), len(CODES))
        if self.labels.shape != expected or self.probs.shape != expected:
            raise ValueError(
                f"shape mismatch: labels {self.labels.shape}, probs {self.probs.shape}, "
                f"expected {expected}"
            )
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must be binary (0 or 1)")
        if not (np.isfinite(self.probs).all() and ((self.probs >= 0) & (self.probs <= 1)).all()):
            raise ValueError("probabilities must be finite and within [0, 1]")

    @property
    def countries(self) -> tuple[str, ...]:
        return tuple(country_of_id(i) for i in self.ids)


def write_predictions(predictions: Predictions, path: Path) -> None:
    """Save ids, countries, labels, probabilities and meta; no pickled objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(list(predictions.ids), dtype=str),
        countries=np.asarray(predictions.countries, dtype=str),
        labels=predictions.labels.astype(np.uint8),
        probs=predictions.probs.astype(np.float32),
        meta=np.asarray(json.dumps(predictions.meta, sort_keys=True)),
    )


def read_predictions(path: Path) -> Predictions:
    with np.load(path, allow_pickle=False) as data:
        return Predictions(
            ids=[str(i) for i in data["ids"]],
            labels=data["labels"],
            probs=data["probs"].astype(np.float64),
            meta=json.loads(str(data["meta"])),
        )


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics for one class in one group, with the counts and the reason for any None."""

    n: int
    positives: int
    prevalence: float
    auroc: float | None
    average_precision: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    notes: tuple[str, ...]


def _average_ranks(values: Array) -> Array:
    """1-based ranks, with tied values sharing the mean of their ranks."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.nonzero(np.r_[True, ordered[1:] != ordered[:-1]])[0]
    ends = np.r_[starts[1:], len(values)]
    group = np.repeat(np.arange(len(starts)), ends - starts)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ((starts + ends + 1) / 2)[group]
    return ranks


def auroc(labels: Array, probs: Array) -> float | None:
    """Area under the ROC curve; None unless both classes are present."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(probs)
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def average_precision(labels: Array, probs: Array) -> float | None:
    """Step-wise average precision; None when there are no positives."""
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-probs, kind="mergesort")
    ordered_labels, ordered_probs = labels[order], probs[order]
    last_of_tie = np.r_[ordered_probs[1:] != ordered_probs[:-1], True]
    tp = np.cumsum(ordered_labels)[last_of_tie]
    fp = np.cumsum(1 - ordered_labels)[last_of_tie]
    recall = tp / positives
    precision = tp / (tp + fp)
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def class_metrics(labels: Array, probs: Array, *, threshold: float) -> ClassMetrics:
    n = len(labels)
    positives = int(labels.sum())
    notes: list[str] = []
    if positives == 0:
        notes.append("no positive examples")
    if positives == n:
        notes.append("no negative examples")
    predicted = probs >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = positives - tp
    precision = tp / (tp + fp) if tp + fp else None
    if precision is None:
        notes.append("no predicted positives")
    return ClassMetrics(
        n=n,
        positives=positives,
        prevalence=positives / n,
        auroc=auroc(labels, probs),
        average_precision=average_precision(labels, probs),
        precision=precision,
        recall=tp / positives if positives else None,
        f1=2 * tp / (2 * tp + fp + fn) if positives else None,
        notes=tuple(notes),
    )


def evaluate_by_country(
    predictions: Predictions, *, threshold: float
) -> dict[str, dict[str, ClassMetrics]]:
    """Metrics per country and class, plus the pooled population under the key ALL."""
    countries = np.asarray(predictions.countries)
    labels = predictions.labels.astype(np.float64)
    probs = predictions.probs.astype(np.float64)
    groups = {name: countries == name for name in sorted(set(countries.tolist()))}
    groups[ALL] = np.ones(len(countries), dtype=bool)
    return {
        name: {
            code: class_metrics(labels[mask, k], probs[mask, k], threshold=threshold)
            for k, code in enumerate(CODES)
        }
        for name, mask in groups.items()
    }


def macro_over_countries(
    by_country: dict[str, dict[str, ClassMetrics]], code: str, metric: str
) -> tuple[float | None, int]:
    """Mean of a metric over the real countries where it is defined, and how many that was."""
    values = [
        getattr(classes[code], metric)
        for name, classes in by_country.items()
        if name != ALL and getattr(classes[code], metric) is not None
    ]
    return (float(np.mean(values)) if values else None), len(values)


def country_prior_probs(
    train_ids: Sequence[str], train_labels: npt.NDArray[Any], eval_ids: Sequence[str]
) -> Array:
    """Baseline that never sees pixels: each country's training prevalence per class.

    A country absent from training gets the pooled training prevalence, not zero.
    """
    if len(train_ids) == 0:
        raise ValueError("country prior needs training examples")
    labels = np.asarray(train_labels, dtype=np.float64)
    if labels.shape != (len(train_ids), len(CODES)):
        raise ValueError(f"train_labels shape {labels.shape} does not match the ids")
    countries = np.asarray([country_of_id(i) for i in train_ids])
    pooled = labels.mean(axis=0)
    prior = {c: labels[countries == c].mean(axis=0) for c in sorted(set(countries.tolist()))}
    return np.stack([prior.get(country_of_id(i), pooled) for i in eval_ids])


def _metrics_dict(metrics: ClassMetrics) -> dict[str, Any]:
    out = asdict(metrics)
    out["notes"] = list(metrics.notes)
    return out


def _macro_entry(
    by_country: dict[str, dict[str, ClassMetrics]], code: str, metric: str
) -> dict[str, Any]:
    mean, used = macro_over_countries(by_country, code, metric)
    return {"mean": mean, "countries_used": used}


def _macro(by_country: dict[str, dict[str, ClassMetrics]]) -> dict[str, Any]:
    return {
        code: {metric: _macro_entry(by_country, code, metric) for metric in METRICS}
        for code in CODES
    }


def build_report(
    model: Predictions, baseline: Predictions | None, *, threshold: float
) -> dict[str, Any]:
    """JSON-ready report: per country and class the model's metrics and the baseline's."""
    if baseline is not None and list(baseline.ids) != list(model.ids):
        raise ValueError("baseline must be evaluated on exactly the model's ids")
    results = {"model": evaluate_by_country(model, threshold=threshold)}
    if baseline is not None:
        results["baseline"] = evaluate_by_country(baseline, threshold=threshold)
    countries: dict[str, Any] = defaultdict(dict)
    for name in results["model"]:
        for code in CODES:
            countries[name][code] = {
                role: _metrics_dict(per_role[name][code]) for role, per_role in results.items()
            }
    macro = {
        code: {role: _macro(per_role)[code] for role, per_role in results.items()} for code in CODES
    }
    return {
        "threshold": threshold,
        "meta": model.meta,
        "baseline_meta": baseline.meta if baseline is not None else None,
        "countries": dict(countries),
        "macro": macro,
    }
