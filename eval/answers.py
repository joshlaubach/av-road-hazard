"""Scoring written answers: do they cite the right photos, and only photos that were retrieved?

A question is paired with a rule on a record ("the record lists a pedestrian"). Among the photos the
search returned for it, those whose record satisfies the rule are the expected citations. This
checks that the model reads the records it was given faithfully; it does not check the records
against the photos (the detector evaluation does that) and it does not check search (eval/retrieval
does that). Pure logic: stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fusion.schema import HazardRecord

OBJECT_LABELS = (
    "pedestrian",
    "cyclist",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "barrier",
    "traffic_cone",
)
OBJECT_PHRASES = {"traffic_cone": "traffic cones", "pedestrian": "pedestrians", "bus": "buses"}
DAMAGE = {
    "D00": "a longitudinal crack",
    "D10": "a transverse crack",
    "D20": "an alligator crack",
    "D40": "a pothole",
}
# Questions the records cannot answer at all: nothing to cite, and a model should say so.
UNANSWERABLE = (
    "What is the weather like in these photos?",
    "Can you read any license plates?",
    "Was anyone injured?",
    "Which city were these photos taken in?",
)


@dataclass(frozen=True)
class Question:
    text: str
    kind: str  # "object", "damage" or "unanswerable"
    rule: Callable[[HazardRecord], bool] | None  # None = nothing in a record can answer it


def _has_object(label: str) -> Callable[[HazardRecord], bool]:
    return lambda r: any(o.label == label for o in r.objects)


def _has_damage(code: str) -> Callable[[HazardRecord], bool]:
    return lambda r: any(d.name == code for d in r.defects)


def make_questions() -> list[Question]:
    """Three phrasings per object, two per damage type, and the unanswerable ones: 36 in all."""
    out: list[Question] = []
    for label in OBJECT_LABELS:
        plural = OBJECT_PHRASES.get(label, f"{label.replace('_', ' ')}s")
        single = label.replace("_", " ")
        rule = _has_object(label)
        out += [
            Question(f"Which photos show {plural}?", "object", rule),
            Question(f"Is there a {single} in any of these photos? Cite them.", "object", rule),
            Question(f"List the photos that contain {plural}.", "object", rule),
        ]
    for code, phrase in DAMAGE.items():
        rule = _has_damage(code)
        out += [
            Question(f"Which photos show {phrase}?", "damage", rule),
            Question(f"Where on these roads is there {phrase}? Cite the photos.", "damage", rule),
        ]
    out += [Question(text, "unanswerable", None) for text in UNANSWERABLE]
    return out


def expected_ids(question: Question, retrieved: Sequence[HazardRecord]) -> frozenset[str]:
    if question.rule is None:
        return frozenset()
    return frozenset(r.frame_id for r in retrieved if question.rule(r))


@dataclass(frozen=True)
class Graded:
    """One answer, graded. `cost_usd` is what the model calls for it cost (0 for a local model)."""

    expected: int  # how many retrieved photos satisfy the rule (0 = nothing correct to cite)
    cited: int  # ids the answer cites that were retrieved
    correct: int  # of those, how many satisfy the rule
    invented: int  # ids the answer cites that were NOT retrieved
    failed: bool  # the model gave no usable answer
    cost_usd: float = 0.0


def grade(
    expected: frozenset[str],
    cited: Sequence[str],
    unknown: Sequence[str],
    *,
    failed: bool = False,
    cost_usd: float = 0.0,
) -> Graded:
    return Graded(
        expected=len(expected),
        cited=len(cited),
        correct=sum(1 for c in cited if c in expected),
        invented=len(unknown),
        failed=failed,
        cost_usd=cost_usd,
    )


def cascade(small: Sequence[Graded], big: Sequence[Graded]) -> list[tuple[Graded, bool]]:
    """What the small-then-big cascade would have returned for each question, and whether it
    escalated: the small model's answer if it did not fail and invented no id, else the big one's,
    with both calls' cost."""
    if len(small) != len(big):
        raise ValueError("both models must have answered the same questions")
    out = []
    for first, second in zip(small, big, strict=True):
        if first.failed or first.invented:
            out.append((_with_cost(second, first.cost_usd + second.cost_usd), True))
        else:
            out.append((first, False))
    return out


def _with_cost(g: Graded, cost: float) -> Graded:
    return Graded(g.expected, g.cited, g.correct, g.invented, g.failed, cost)


def summarize(rows: Sequence[Graded], escalated: Sequence[bool] | None = None) -> dict[str, Any]:
    """Counts and rates over the answers (no photo ids, no question texts)."""
    answerable = [g for g in rows if g.expected > 0]
    unanswerable = [g for g in rows if g.expected == 0]
    scored = [g for g in answerable if not g.failed]
    precise = sum(1 for g in scored if g.cited > 0 and g.correct == g.cited)
    recall = [g.correct / g.expected for g in scored]
    out: dict[str, Any] = {
        "questions": len(rows),
        "answerable": len(answerable),
        "failed": sum(1 for g in rows if g.failed),
        "answerable_all_citations_correct": precise,
        "answerable_cited_nothing": sum(1 for g in scored if g.cited == 0),
        "answerable_mean_recall": round(sum(recall) / len(recall), 4) if recall else None,
        "unanswerable": len(unanswerable),
        "unanswerable_cited_a_photo": sum(1 for g in unanswerable if g.cited > 0),
        "answers_with_an_invented_id": sum(1 for g in rows if g.invented > 0),
        "cost_usd_total": round(sum(g.cost_usd for g in rows), 5),
        "cost_usd_per_question": round(sum(g.cost_usd for g in rows) / len(rows), 5)
        if rows
        else None,
    }
    if escalated is not None:
        out["escalated"] = sum(escalated)
    return out


def answers_section(report: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the evaluation report that goes into eval/results.json."""
    return {
        "status": "measured",
        "retrieved_per_question": report["top_k"],
        "questions": report["questions"],
        "configurations": report["configurations"],
        "provenance": report["provenance"],
        "caveat": (
            "Each question is checked against the records the model was given, not against the "
            "photos: this measures whether the model reads and cites the records faithfully. "
            "Questions about weather, plates and similar facts have no record behind them, so "
            "they are counted separately. Cost is estimated from reported token counts at list "
            "prices."
        ),
    }
