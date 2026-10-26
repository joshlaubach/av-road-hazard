"""CPU-only ML smoke tests; no real images, downloads or trained weights."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from defects.dataset import Sample, build_dataset_manifest, file_sha256
from defects.labels import CODES
from defects.model import build_model
from defects.splits import Splits, write_splits
from defects.train import fit, load_checkpoint, manifest_path, out_of_fold, predict

# Keep test collection and the base environment independent of all optional packages.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
pytestmark = pytest.mark.ml


@pytest.fixture
def training_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Sample], Path, dict[str, Any]]:
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path("configs/defects_train.yaml").read_text())
    cfg.update(
        checkpoint_dir=str(tmp_path / "weights"),
        pretrained=False,
        input_size=16,
        epochs=8,
        patience=8,
        batch_size=2,
        lr=0.1,
        horizontal_flip=0.0,
        color_jitter=0.0,
        head_dropout=0.0,
        drop_path=0.0,
    )
    samples = []
    for i in range(20):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (16, 16), (20 + i, 40, 60)).save(path)
        samples.append(Sample(str(i), path, (1, 0, 1, 0)))
    splits = Splits(
        ["19"],
        [
            ([str(i) for i in range(8)], ["8"], [str(i) for i in range(9, 19)]),
            ([str(i) for i in range(9, 17)], ["17", "18"], [str(i) for i in range(9)]),
        ],
    )
    path = tmp_path / "splits.json"
    write_splits(splits, path)
    manifest_path(path).write_text(json.dumps(build_dataset_manifest(samples)))

    def tiny_model(cfg: dict[str, Any]) -> Any:
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
        )

    monkeypatch.setattr("defects.train.build_model", tiny_model)
    monkeypatch.setattr("defects.model.build_model", tiny_model)
    return samples, path, cfg


@pytest.mark.parametrize("name", ["efficientnet_b4", "resnet10t"])
def test_model_shape(name: str) -> None:
    cfg = {"model_name": name, "pretrained": False, "head_dropout": 0.3, "drop_path": 0.2}
    torch.set_num_threads(1)
    model = build_model(cfg).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 4)


def test_overfit_repeat_roundtrip_and_provenance(
    training_case: tuple[list[Sample], Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("defects.train._git_dirty", lambda: True)
    samples, splits, cfg = training_case
    first = fit(samples, splits, 0, cfg, run_id="first", device="cpu")
    assert first.history[-1]["train_loss"] < first.history[0]["train_loss"] * 0.5
    model = load_checkpoint(first.checkpoint, cfg)
    probs = predict(model, samples, cfg, device="cpu")
    assert all(
        (p >= 0.5) == bool(y)
        for row, s in zip(probs, samples, strict=True)
        for p, y in zip(row, s.labels, strict=True)
    )
    assert probs == predict(load_checkpoint(first.checkpoint, cfg), samples, cfg, device="cpu")
    again = fit(samples, splits, 0, cfg, run_id="again", device="cpu")
    assert first.history == again.history
    assert probs == predict(load_checkpoint(again.checkpoint, cfg), samples, cfg, device="cpu")
    other = fit(samples, splits, 1, cfg, run_id="first", device="cpu")
    assert other.checkpoint != first.checkpoint
    sidecar = json.loads(first.checkpoint.with_suffix(".json").read_text())
    assert sidecar["checkpoint_sha256"] == file_sha256(first.checkpoint)
    assert sidecar["split_file_sha256"] == file_sha256(splits)
    assert len(sidecar["dataset_identity"]) == 64
    assert sidecar["git_sha"] == "unknown" or len(sidecar["git_sha"]) == 40
    assert sidecar["git_dirty"] is True
    assert sidecar["fold_index"] == 0
    assert sidecar["class_codes"] == list(CODES)
    assert sidecar["seed"] == 42
    assert sidecar["epoch"] == first.best_epoch
    assert sidecar["config"] == cfg
    with pytest.raises(FileExistsError):
        fit(samples, splits, 0, cfg, run_id="first", device="cpu")
    first.checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(first.checkpoint, cfg)


def test_early_stopping_and_manifest_guard(
    training_case: tuple[list[Sample], Path, dict[str, Any]],
) -> None:
    samples, splits, cfg = training_case
    cfg.update(lr=0.0, patience=2, epochs=10)
    result = fit(samples, splits, 0, cfg, run_id="stop", device="cpu")
    assert len(result.history) == 3
    assert result.best_epoch == 0
    Image.new("RGB", (16, 16), "red").save(samples[0].image_path)
    with pytest.raises(ValueError, match="manifest mismatch"):
        fit(samples, splits, 0, cfg, run_id="bad", device="cpu")
    with pytest.raises(ValueError, match="manifest mismatch"):
        out_of_fold(samples, splits, cfg, run_id="bad-oof", device="cpu")


def test_oof_excludes_final_test(
    training_case: tuple[list[Sample], Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from defects.dataset import DefectDataset

    samples, splits, cfg = training_case
    original = DefectDataset.__getitem__

    def guard(self: DefectDataset, index: int) -> tuple[Any, Any]:
        assert self.samples[index].sample_id != "19"
        return original(self, index)

    monkeypatch.setattr(DefectDataset, "__getitem__", guard)
    probs = out_of_fold(samples, splits, cfg, run_id="oof", device="cpu")
    assert set(probs) == {str(i) for i in range(19)}
    assert all(len(row) == 4 and all(0 <= p <= 1 for p in row) for row in probs.values())


@pytest.mark.parametrize("mode", ["fit", "oof"])
def test_run_hashes_each_image_once(
    training_case: tuple[list[Sample], Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from collections import Counter

    from defects import dataset
    from defects.splits import make_splits

    samples, split_file, cfg = training_case
    cfg["epochs"] = 1
    splits = make_splits([s.sample_id for s in samples], lambda x: x, n_folds=3, seed=42)
    write_splits(splits, split_file)
    expected = dataset.dataset_identity(samples)
    counts: Counter[Path] = Counter()
    original = dataset.file_sha256

    def counted(path: Path) -> str:
        counts[path] += 1
        return original(path)

    monkeypatch.setattr(dataset, "file_sha256", counted)
    if mode == "oof":
        out_of_fold(samples, split_file, cfg, run_id=mode, device="cpu")
        folds = range(3)
    else:
        fit(samples, split_file, 0, cfg, run_id=mode, device="cpu")
        folds = range(1)
    assert counts == Counter({s.image_path: 1 for s in samples})
    for fold in folds:
        sidecar = Path(cfg["checkpoint_dir"]) / mode / f"fold{fold}" / "best.json"
        assert json.loads(sidecar.read_text())["dataset_identity"] == expected


def test_training_drops_singleton_and_normalizes_seen_samples(
    training_case: tuple[list[Sample], Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from defects import train

    samples, split_file, cfg = training_case
    cfg.update(epochs=1, lr=0.0)
    splits = Splits(
        ["19"],
        [
            (["0", "1", "2"], [str(i) for i in range(3, 9)], [str(i) for i in range(9, 19)]),
            ([str(i) for i in range(9, 17)], ["17", "18"], [str(i) for i in range(9)]),
        ],
    )
    write_splits(splits, split_file)
    rows = samples[: cfg["batch_size"] + 1]
    assert len(rows) == cfg["batch_size"] + 1
    assert [len(images) for images, _ in train._loader(rows, cfg, training=True)] == [2]
    assert [len(images) for images, _ in train._loader(rows, cfg, training=False)] == [2, 1]
    assert [len(images) for images, _ in train._loader(rows[:2], cfg, training=True)] == [2]

    def batchnorm_model(cfg: dict[str, Any]) -> Any:
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3, 4),
        )

    losses: list[float] = []
    criterion = torch.nn.BCEWithLogitsLoss()

    def record_loss(module: Any, inputs: Any, output: Any) -> None:
        losses.append(float(output.detach()))

    criterion.register_forward_hook(record_loss)
    monkeypatch.setattr(train, "build_model", batchnorm_model)
    monkeypatch.setattr(torch.nn, "BCEWithLogitsLoss", lambda: criterion)
    result = fit(samples, split_file, 0, cfg, run_id="singleton", device="cpu")
    assert result.history[0]["train_loss"] == pytest.approx(losses[0])
