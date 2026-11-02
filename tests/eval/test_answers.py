from __future__ import annotations

import pytest

from eval.answers import (
    Graded,
    answers_section,
    cascade,
    expected_ids,
    grade,
    make_questions,
    summarize,
)
from fusion.schema import DefectRecord, HazardRecord, Object


def record(
    frame_id: str, labels: tuple[str, ...] = (), damage: tuple[str, ...] = ()
) -> HazardRecord:
    return HazardRecord(
        frame_id=frame_id,
        source="nuscenes",
        image_size=(10, 10),
        objects=[Object(label=label, score=0.9, box=(0, 0, 1, 1)) for label in labels],
        defects=[
            DefectRecord(name=c, confidence=0.9, box=(0, 0, 1, 1), severity="high", in_road=True)
            for c in damage
        ],
        hazard_score=0.0,
    )


def test_there_are_36_questions_in_three_kinds() -> None:
    questions = make_questions()
    assert len(questions) == 36
    kinds = [q.kind for q in questions]
    assert kinds.count("object") == 24 and kinds.count("damage") == 8
    assert kinds.count("unanswerable") == 4
    assert len({q.text for q in questions}) == 36


def test_expected_citations_are_the_retrieved_photos_whose_record_has_the_thing() -> None:
    retrieved = [
        record("a", ("car", "pedestrian")),
        record("b", ("car",)),
        record("c", (), ("D40",)),
    ]
    by_text = {q.text: q for q in make_questions()}
    assert expected_ids(by_text["Which photos show pedestrians?"], retrieved) == {"a"}
    assert expected_ids(by_text["Which photos show a pothole?"], retrieved) == {"c"}
    assert expected_ids(by_text["Which photos show buses?"], retrieved) == frozenset()
    assert (
        expected_ids(by_text["What is the weather like in these photos?"], retrieved) == frozenset()
    )


def test_grading_counts_correct_cited_and_invented_ids() -> None:
    g = grade(frozenset({"a", "b"}), ["a", "c"], ["ghost"], cost_usd=0.01)
    assert (g.expected, g.cited, g.correct, g.invented, g.cost_usd) == (2, 2, 1, 1, 0.01)


def test_the_cascade_keeps_a_clean_small_answer_and_escalates_otherwise() -> None:
    clean = Graded(1, 1, 1, 0, False, 0.001)
    ghost = Graded(1, 1, 0, 1, False, 0.001)
    down = Graded(1, 0, 0, 0, True, 0.001)
    big = Graded(1, 1, 1, 0, False, 0.02)
    rows = cascade([clean, ghost, down], [big, big, big])
    assert [escalated for _, escalated in rows] == [False, True, True]
    assert rows[0][0].cost_usd == pytest.approx(0.001)
    assert rows[1][0].correct == 1 and rows[1][0].cost_usd == pytest.approx(0.021)


def test_the_cascade_needs_the_same_questions_from_both_models() -> None:
    with pytest.raises(ValueError):
        cascade([Graded(1, 1, 1, 0, False)], [])


def test_summary_separates_answerable_from_unanswerable_questions() -> None:
    rows = [
        Graded(2, 2, 2, 0, False, 0.01),  # all correct
        Graded(2, 2, 1, 0, False, 0.01),  # one wrong citation
        Graded(2, 0, 0, 0, False, 0.01),  # cited nothing
        Graded(0, 1, 0, 0, False, 0.01),  # unanswerable but cited a photo
        Graded(0, 0, 0, 1, False, 0.01),  # unanswerable, invented an id
        Graded(1, 0, 0, 0, True, 0.0),  # failed
    ]
    s = summarize(rows, [False, False, False, False, False, True])
    assert s["questions"] == 6 and s["answerable"] == 4 and s["failed"] == 1
    assert s["answerable_all_citations_correct"] == 1 and s["answerable_cited_nothing"] == 1
    assert s["unanswerable"] == 2 and s["unanswerable_cited_a_photo"] == 1
    assert s["answers_with_an_invented_id"] == 1 and s["escalated"] == 1
    assert s["cost_usd_per_question"] == pytest.approx(0.05 / 6, abs=1e-5)
    assert s["answerable_mean_recall"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_the_results_section_keeps_numbers_only() -> None:
    report = {
        "top_k": 5,
        "questions": 36,
        "configurations": {"x": {"questions": 36}},
        "provenance": {"files": {}},
        "rows": [{"cited": ["n008-secret-image-name"]}],
    }
    section = answers_section(report)
    assert "rows" not in section and section["status"] == "measured"
    assert "records" in section["caveat"]


def test_answers_row_names_each_configuration_and_its_cost() -> None:
    from eval.answers import answers_section
    from eval.results import _answers_row

    config = {
        "answerable": 23,
        "answerable_all_citations_correct": 22,
        "cost_usd_per_question": 0.00017,
    }
    report = {
        "top_k": 5,
        "questions": 36,
        "configurations": {"small alone": config},
        "provenance": {},
    }
    row = _answers_row({"answers": answers_section(report)})
    assert "36 questions" in row and "small alone: 22 of 23" in row and "$0.00017" in row
