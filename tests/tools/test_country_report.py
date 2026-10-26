"""Country report CLI on synthetic ids, labels and probabilities; no model or image is involved."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from defects.evaluation import Predictions, read_predictions, write_predictions
from defects.splits import make_splits, write_splits
from tools.country_report import baseline_predictions, main

COUNTRIES = ("Czech", "Japan")


def setup(tmp_path: Path) -> tuple[list[str], Path, Path]:
    """24 images, half per country; Czech D00 is rare in the first half of its ids, common later."""
    ids = [f"{c}/train/annotations/xmls/{c}_{i:02d}" for c in COUNTRIES for i in range(12)]
    rows = []
    for n, image_id in enumerate(ids):
        d00 = int(image_id.startswith("Japan") or n % 4 == 0)
        rows.append({"id": image_id, "image_path": f"{n}.png", "labels": [d00, 0, 0, n % 2]})
    index = tmp_path / "sample_index.json"
    index.write_text(json.dumps(rows))
    splits = make_splits(ids, lambda i: i, test_fraction=0.1, n_folds=5, val_fraction=0.1, seed=42)
    splits_file = tmp_path / "splits.json"
    write_splits(splits, splits_file)
    return ids, index, splits_file


def artifact(tmp_path: Path, ids: list[str], **meta: object) -> Path:
    rng = np.random.default_rng(0)
    labels = np.asarray([[int(i.startswith("Japan")), 0, 0, n % 2] for n, i in enumerate(ids)])
    probs = np.clip(labels * 0.6 + rng.uniform(0.0, 0.4, labels.shape), 0, 1)
    path = tmp_path / "model.npz"
    write_predictions(Predictions(ids=ids, labels=labels, probs=probs, meta=dict(meta)), path)
    return path


def test_baseline_uses_only_the_training_images_of_the_chosen_fold(tmp_path: Path) -> None:
    ids, index, splits_file = setup(tmp_path)
    model = read_predictions(artifact(tmp_path, ids))
    baseline = baseline_predictions(model, index, splits_file, fold=0)
    rows = {r["id"]: r for r in json.loads(index.read_text())}
    train = json.loads(splits_file.read_text())["folds"][0][0]
    for country in COUNTRIES:
        mine = [rows[i]["labels"] for i in train if i.startswith(country)]
        expected = np.mean(mine, axis=0)
        position = next(k for k, i in enumerate(model.ids) if i.startswith(country))
        np.testing.assert_allclose(baseline.probs[position], expected)
    assert list(baseline.ids) == list(model.ids)


def test_a_country_missing_from_training_gets_the_pooled_prevalence(tmp_path: Path) -> None:
    ids, index, splits_file = setup(tmp_path)
    extra = [*ids, "Norway/train/annotations/xmls/Norway_00"]
    labels = np.zeros((len(extra), 4), dtype=int)
    model = Predictions(extra, labels, np.full((len(extra), 4), 0.5))
    baseline = baseline_predictions(model, index, splits_file, fold=0)
    train = json.loads(splits_file.read_text())["folds"][0][0]
    rows = {r["id"]: r for r in json.loads(index.read_text())}
    pooled = np.mean([rows[i]["labels"] for i in train], axis=0)
    np.testing.assert_allclose(baseline.probs[-1], pooled)


def test_cli_writes_a_report_with_model_and_baseline_and_prints_a_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ids, index, splits_file = setup(tmp_path)
    model = artifact(tmp_path, ids, threshold=0.7, model="smoke")
    out = tmp_path / "report.json"
    argv = ["--predictions", str(model), "--out", str(out)]
    argv += ["--baseline-sample-index", str(index), "--splits", str(splits_file), "--fold", "0"]
    assert main(argv) == 0
    report = json.loads(out.read_text())
    assert report["threshold"] == 0.7  # taken from the artifact's metadata
    assert report["meta"]["model"] == "smoke"
    cell = report["countries"]["Japan"]["D00"]
    assert cell["model"]["n"] == 12 and cell["model"]["positives"] == 12
    assert cell["model"]["auroc"] is None and "no negative examples" in cell["model"]["notes"]
    assert report["countries"]["Czech"]["D00"]["baseline"]["n"] == 12
    printed = capsys.readouterr().out
    assert "D00" in printed and "Japan" in printed and "n/a" in printed


def test_threshold_argument_overrides_the_artifact(tmp_path: Path) -> None:
    ids, _, _ = setup(tmp_path)
    out = tmp_path / "report.json"
    main(
        [
            "--predictions",
            str(artifact(tmp_path, ids, threshold=0.7)),
            "--out",
            str(out),
            "--threshold",
            "0.3",
        ]
    )
    assert json.loads(out.read_text())["threshold"] == 0.3


def test_report_without_baseline_arguments_has_no_baseline(tmp_path: Path) -> None:
    ids, _, _ = setup(tmp_path)
    out = tmp_path / "report.json"
    main(["--predictions", str(artifact(tmp_path, ids)), "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["threshold"] == 0.5 and report["baseline_meta"] is None
    assert "baseline" not in report["countries"]["Czech"]["D00"]


def test_baseline_arguments_must_come_together(tmp_path: Path) -> None:
    ids, index, _ = setup(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "--predictions",
                str(artifact(tmp_path, ids)),
                "--out",
                str(tmp_path / "r.json"),
                "--baseline-sample-index",
                str(index),
            ]
        )
