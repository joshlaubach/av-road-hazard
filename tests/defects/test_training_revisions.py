"""Regression checks for the accepted provenance, hashing and persistence findings."""

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from defects import dataset, train
from defects.dataset import Sample, build_dataset_manifest, dataset_identity
from defects.splits import make_splits, write_splits


@pytest.fixture
def run_case(tmp_path: Path) -> tuple[list[Sample], Path, dict[str, Any]]:
    samples = []
    for i in range(20):
        image = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i, 0, 0)).save(image)
        samples.append(Sample(str(i), image, (1, 0, 0, 0)))
    split_file = tmp_path / "splits.json"
    splits = make_splits([s.sample_id for s in samples], lambda x: x, n_folds=3, seed=42)
    write_splits(splits, split_file)
    train.manifest_path(split_file).write_text(json.dumps(build_dataset_manifest(samples)))
    return samples, split_file, {"verify_hashes": True}


def test_oof_hashes_each_image_once(
    run_case: tuple[list[Sample], Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples, split_file, cfg = run_case
    expected_identity = dataset_identity(samples)
    hashes: Counter[Path] = Counter()
    original = dataset.file_sha256
    identities: list[str] = []

    def counted(path: Path) -> str:
        hashes[path] += 1
        return original(path)

    def fake_fit(*args: Any, identity: str, **kwargs: Any) -> train.FitResult:
        identities.append(identity)
        return train.FitResult(Path("unused.pt"), [], 0)

    monkeypatch.setattr(dataset, "file_sha256", counted)
    monkeypatch.setattr(train, "_fit", fake_fit)
    monkeypatch.setattr(train, "load_checkpoint", lambda *a, **kw: object())
    monkeypatch.setattr(train, "predict", lambda model, rows, cfg, **kw: [[0.5] * 4 for _ in rows])
    result = train.out_of_fold(samples, split_file, cfg, run_id="count", device="cpu")
    assert len(result) == 17
    assert hashes == Counter({s.image_path: 1 for s in samples})
    assert identities == [expected_identity] * 3
    Image.new("RGB", (8, 8), "red").save(samples[0].image_path)
    assert dataset_identity(samples) != expected_identity
    with pytest.raises(ValueError, match="manifest mismatch"):
        train.out_of_fold(samples, split_file, cfg, run_id="changed", device="cpu")


def test_standalone_fit_verifies_once(
    run_case: tuple[list[Sample], Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples, split_file, cfg = run_case
    expected = dataset_identity(samples)
    identities: list[str] = []

    def fake_fit(*args: Any, identity: str, **kwargs: Any) -> train.FitResult:
        identities.append(identity)
        return train.FitResult(Path("unused.pt"), [], 0)

    monkeypatch.setattr(train, "_fit", fake_fit)
    train.fit(samples, split_file, 0, cfg, run_id="single")
    assert identities == [expected]
    Image.new("RGB", (8, 8), "red").save(samples[0].image_path)
    with pytest.raises(ValueError, match="manifest mismatch"):
        train.fit(samples, split_file, 0, cfg, run_id="changed")
    assert identities == [expected]


@pytest.mark.parametrize("status, expected", [(" M defects/train.py\n", True), ("", False)])
def test_git_dirty_status(status: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: status)
    assert train._git_dirty() is expected


@pytest.mark.parametrize("write_error", [OSError, RuntimeError])
def test_failed_manifest_write_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_error: type[Exception]
) -> None:
    image = tmp_path / "fake.png"
    Image.new("RGB", (8, 8)).save(image)
    rows = [{"id": str(i), "image_path": image.name, "labels": [0, 0, 0, 0]} for i in range(20)]
    (tmp_path / "samples.json").write_text(json.dumps(rows))
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({str(i): str(i) for i in range(20)}))
    cfg = yaml.safe_load(Path("configs/defects_train.yaml").read_text())
    cfg.update(sample_index="samples.json", splits_file="saved/splits.json", n_folds=3)
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    split_file = tmp_path / "saved/splits.json"
    manifest_file = train.manifest_path(split_file)
    original = Path.write_text

    def fail_manifest(self: Path, *args: Any, **kwargs: Any) -> int:
        if self == manifest_file:
            original(self, "partial")
            raise write_error("simulated manifest write failure")
        return original(self, *args, **kwargs)

    args = ["splits", "--config", str(config_file), "--groups", str(groups)]
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_manifest)
        with pytest.raises(write_error, match="manifest write failure"):
            train.main(args)
    assert not split_file.exists()
    assert not manifest_file.exists()
    assert train.main(args) == 0
    assert split_file.exists() and manifest_file.exists()
