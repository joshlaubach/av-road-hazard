"""evaluate(): the first checkpoint is the baseline and the recall guard compares against it."""

from pathlib import Path

import numpy as np
import pytest

from tools import eval_classifier
from tools.eval_classifier import Context, evaluate

# two tiles per class: one positive, one negative; probabilities set per checkpoint
TRUTH = np.array(
    [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
)
GOOD = np.array(
    [[0.95, 0, 0, 0], [0.1, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 0.7, 0], [0, 0, 0, 0.8], [0, 0, 0, 0]]
)
LOST_D10 = GOOD.copy()
LOST_D10[2, 1] = 0.1  # the only D10 positive falls under its cut-off


@pytest.fixture
def fake_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    by_checkpoint = {"base.pt": GOOD, "new.pt": LOST_D10}
    monkeypatch.setattr(eval_classifier, "shipped_thresholds", lambda: (0.9, 0.75, 0.65, 0.75))
    monkeypatch.setattr(eval_classifier, "load_model", lambda path, ctx: (Path(path).name, "cpu"))
    monkeypatch.setattr(
        eval_classifier,
        "score_validation",
        lambda model, device, ctx, fold: (TRUTH, by_checkpoint[model]),
    )


def test_the_second_checkpoint_is_guarded_against_the_first(
    fake_scores: None, tmp_path: Path
) -> None:
    for name in ("base.pt", "new.pt"):
        (tmp_path / name).write_bytes(name.encode())

    report = evaluate(
        {"base": tmp_path / "base.pt", "new": tmp_path / "new.pt"},
        Context({}, tmp_path),
        repo=tmp_path,
    )

    assert report["baseline"] == "base"
    assert "guard" not in report["checkpoints"]["base"]
    guard = report["checkpoints"]["new"]["guard"]
    assert guard["passed"] is False
    assert "D10" in guard["reasons"][0]


def test_a_checkpoint_identical_to_the_baseline_passes(fake_scores: None, tmp_path: Path) -> None:
    (tmp_path / "base.pt").write_bytes(b"base")

    report = evaluate(
        {"base": tmp_path / "base.pt", "same": tmp_path / "base.pt"},
        Context({}, tmp_path),
        repo=tmp_path,
    )

    assert report["checkpoints"]["same"]["guard"] == {"passed": True, "reasons": []}


def test_the_report_records_provenance_and_thresholds(fake_scores: None, tmp_path: Path) -> None:
    (tmp_path / "base.pt").write_bytes(b"weights")

    report = evaluate({"base": tmp_path / "base.pt"}, Context({}, tmp_path), repo=tmp_path)

    assert report["provenance"]["thresholds"] == {"D00": 0.9, "D10": 0.75, "D20": 0.65, "D40": 0.75}
    assert "checkpoint/base" in report["provenance"]["files"]
