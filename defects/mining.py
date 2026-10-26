"""Rank label-level errors exclusively from permitted out-of-fold predictions."""

from collections.abc import Sequence, Set
from dataclasses import dataclass
from math import isfinite

from defects.labels import CODES


@dataclass(frozen=True)
class HardExample:
    """One label-level error: which sample, which class code, and how confident the model was."""

    sample_id: str
    code: str
    kind: str
    probability: float


def rank_hard_examples(
    ids: Sequence[str],
    y_true: Sequence[Sequence[float]],
    y_prob: Sequence[Sequence[float]],
    threshold: float,
    *,
    allowed_ids: Set[str],
    test_ids: Set[str],
) -> list[HardExample]:
    """Rank label errors from out-of-fold predictions only; final-test ids are refused."""
    if not test_ids:
        raise ValueError("Final test ids must be nonempty")
    if set(ids) & test_ids:
        raise ValueError("Final test ids are forbidden in hard-example mining")
    if not set(ids) <= allowed_ids:
        raise ValueError("Ids outside the permitted out-of-fold set")
    if len(set(ids)) != len(ids) or not len(ids) == len(y_true) == len(y_prob):
        raise ValueError("Duplicate ids or mismatched prediction lengths")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one")
    errors: list[HardExample] = []
    for item, truth, probs in zip(ids, y_true, y_prob, strict=True):
        if len(truth) != len(CODES) or len(probs) != len(CODES):
            raise ValueError("Expected four class columns")
        for code, target, prob in zip(CODES, truth, probs, strict=True):
            if target not in (0, 1) or not isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError("Invalid label or probability")
            if target == 1 and prob < threshold:
                errors.append(HardExample(item, code, "fn", float(prob)))
            elif target == 0 and prob >= threshold:
                errors.append(HardExample(item, code, "fp", float(prob)))
    return sorted(
        errors,
        key=lambda e: (
            e.kind != "fn",
            e.probability if e.kind == "fn" else -e.probability,
            e.sample_id,
            e.code,
        ),
    )
