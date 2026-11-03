"""The answers report keeps each question and the answer text, so a reader can check by eye."""

from __future__ import annotations

from eval.answers import Question
from fusion.schema import HazardRecord, Meta
from rag.index import Hit
from tools.eval_answers import run_model


class Canned:
    name = "canned"

    def generate(self, system: str, prompt: str) -> str:
        return "The photo [a] shows a road."


def test_the_detail_holds_the_question_and_the_answer_text() -> None:
    record = HazardRecord(
        frame_id="a",
        source="nuscenes",
        image_size=(10, 10),
        hazard_score=0.0,
        meta=Meta(),
    )
    question = Question("is there a road?", "object", lambda r: True)
    graded, detail = run_model(lambda on_usage: Canned(), [question], [[Hit(record, 0.1)]])
    assert len(graded) == 1
    assert detail[0]["question"] == "is there a road?"
    assert detail[0]["text"] == "The photo [a] shows a road."
    assert detail[0]["cited"] == ["a"]
