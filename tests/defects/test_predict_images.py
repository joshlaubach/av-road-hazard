"""Prediction script on a tiny CPU model and generated images; no real data or downloads."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from defects.dataset import Sample, build_dataset_manifest, file_sha256
from defects.evaluation import read_predictions
from defects.splits import Splits, write_splits
from defects.train import fit, manifest_path
from tools.predict_images import main

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
pytestmark = pytest.mark.ml

LABELS = (1, 0, 1, 0)


@pytest.fixture
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path("configs/defects_train.yaml").read_text())
    cfg.update(
        checkpoint_dir=str(tmp_path / "weights"),
        pretrained=False,
        input_size=16,
        epochs=1,
        patience=1,
        batch_size=2,
        horizontal_flip=0.0,
        color_jitter=0.0,
        head_dropout=0.0,
        drop_path=0.0,
    )
    rows, samples = [], []
    for i in range(20):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (16, 16), (20 + i, 40, 60)).save(path)
        image_id = f"Czech/img{i}"
        samples.append(Sample(image_id, path, LABELS))
        rows.append({"id": image_id, "image_path": f"{i}.png", "labels": list(LABELS)})
    ids = [s.sample_id for s in samples]
    splits = Splits(
        [ids[19]],
        [(ids[:8], [ids[8]], ids[9:19]), (ids[9:17], ids[17:19], ids[:9])],
    )
    splits_file = tmp_path / "splits.json"
    write_splits(splits, splits_file)
    manifest_path(splits_file).write_text(json.dumps(build_dataset_manifest(samples)))
    index = tmp_path / "sample_index.json"
    index.write_text(json.dumps(rows))
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(cfg))

    def tiny_model(_: dict[str, Any]) -> Any:
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
        )

    monkeypatch.setattr("defects.train.build_model", tiny_model)
    checkpoint = fit(samples, splits_file, 0, cfg, run_id="t").checkpoint
    return {
        "argv": [
            *("--config", str(config), "--checkpoint", str(checkpoint)),
            *("--sample-index", str(index), "--splits", str(splits_file), "--fold", "0"),
        ],
        "splits": splits,
        "checkpoint": checkpoint,
        "out": tmp_path / "predictions.npz",
    }


def test_artifact_holds_the_requested_part_with_labels_probabilities_and_provenance(
    trained: dict[str, Any],
) -> None:
    assert main([*trained["argv"], "--part", "heldout", "--out", str(trained["out"])]) == 0
    result = read_predictions(trained["out"])
    assert list(result.ids) == trained["splits"].folds[0][2]
    assert result.countries == ("Czech",) * 10
    assert result.probs.shape == (10, 4) and ((result.probs >= 0) & (result.probs <= 1)).all()
    assert (result.labels == LABELS).all()
    assert result.meta["checkpoint_sha256"] == file_sha256(trained["checkpoint"])
    assert (result.meta["fold"], result.meta["part"], result.meta["unit"]) == (
        0,
        "heldout",
        "image",
    )
    assert result.meta["threshold"] == 0.5


@pytest.mark.parametrize(("part", "count"), [("train", 8), ("val", 1)])
def test_other_parts_are_available(trained: dict[str, Any], part: str, count: int) -> None:
    main([*trained["argv"], "--part", part, "--out", str(trained["out"])])
    assert len(read_predictions(trained["out"]).ids) == count


def test_the_final_test_set_cannot_be_scored(trained: dict[str, Any]) -> None:
    with pytest.raises(SystemExit):
        main([*trained["argv"], "--part", "test", "--out", str(trained["out"])])
    assert not trained["out"].exists()


def test_an_out_of_range_fold_is_rejected(trained: dict[str, Any]) -> None:
    argv = [a if a != "0" else "5" for a in trained["argv"]]
    with pytest.raises(SystemExit):
        main([*argv, "--part", "val", "--out", str(trained["out"])])
