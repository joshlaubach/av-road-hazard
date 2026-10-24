import hashlib
import json
from pathlib import Path

import pytest

from common.config import get_config
from defects.road_crop import MODEL_REVISION, MODEL_SUBDIR
from tools import fetch_road_model as fetch

GOOD = b"good weights"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()
FILES = {"model.safetensors": GOOD_SHA}


class FakeDownloader:
    def __init__(self, payload: bytes = GOOD) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, str, Path]] = []

    def __call__(self, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        self.calls.append((repo_id, filename, revision, local_dir))
        out = local_dir / filename
        out.write_bytes(self.payload)
        return out


def target_of(root: Path) -> Path:
    return root / MODEL_SUBDIR


def test_skips_file_already_present_and_verified(tmp_path: Path) -> None:
    target_of(tmp_path).mkdir(parents=True)
    (target_of(tmp_path) / "model.safetensors").write_bytes(GOOD)
    fake = FakeDownloader()
    entries = fetch.fetch_model(tmp_path, FILES, fake)
    assert fake.calls == []
    assert [e["sha256"] for e in entries] == [GOOD_SHA]


def test_fetches_missing_file_and_cleans_scratch(tmp_path: Path) -> None:
    fake = FakeDownloader()
    entries = fetch.fetch_model(tmp_path, FILES, fake)
    assert (target_of(tmp_path) / "model.safetensors").read_bytes() == GOOD
    assert fake.calls[0][2] == MODEL_REVISION
    assert entries[0]["size_bytes"] == len(GOOD)
    assert not [p for p in target_of(tmp_path).parent.iterdir() if p.name.startswith(".fetch_")]


def test_mismatch_deletes_download_and_keeps_existing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target_of(tmp_path).mkdir(parents=True)
    existing = target_of(tmp_path) / "model.safetensors"
    existing.write_bytes(b"older but kept")
    fake = FakeDownloader(b"tampered")
    with pytest.raises(fetch.FetchError, match="expected"):
        fetch.fetch_model(tmp_path, FILES, fake)
    assert existing.read_bytes() == b"older but kept"
    assert not [p for p in target_of(tmp_path).parent.iterdir() if p.name.startswith(".fetch_")]


def test_main_returns_1_on_mismatch_and_names_hashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = fetch.main(["--checkpoint-dir", str(tmp_path)], FakeDownloader(b"tampered"), FILES)
    out = capsys.readouterr().out
    assert code == 1
    assert "expected" in out and hashlib.sha256(b"tampered").hexdigest() in out


def test_main_returns_1_when_download_raises(tmp_path: Path) -> None:
    def boom(repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        raise OSError("offline")

    assert fetch.main(["--checkpoint-dir", str(tmp_path)], boom, FILES) == 1


@pytest.mark.parametrize("name", ["w.bin", "w.pt", "w.PTH", "w.pkl", "w.ckpt", "w.pickle"])
def test_refuses_pickle_style_filenames(tmp_path: Path, name: str) -> None:
    fake = FakeDownloader()
    with pytest.raises(ValueError, match="pickle"):
        fetch.fetch_model(tmp_path, {name: GOOD_SHA}, fake)
    assert fake.calls == []


def test_dry_run_fetches_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeDownloader()
    code = fetch.main(["--checkpoint-dir", str(tmp_path), "--dry-run"], fake, FILES)
    assert code == 0
    assert fake.calls == []
    assert not target_of(tmp_path).exists()
    assert "would fetch" in capsys.readouterr().out


def test_manifest_entries_have_expected_keys(tmp_path: Path) -> None:
    entry = fetch.fetch_model(tmp_path, FILES, FakeDownloader())[0]
    assert set(entry) == {
        "path",
        "sha256",
        "size_bytes",
        "source",
        "version",
        "license",
        "retrieved",
    }
    assert entry["path"] == f"{MODEL_SUBDIR}/model.safetensors"
    assert entry["version"] == MODEL_REVISION
    assert entry["source"] == "huggingface.co/commaai/comma10k-segnet"


def test_default_target_comes_from_config_checkpoint_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    get_config.cache_clear()
    try:
        assert fetch.main([], FakeDownloader(), FILES) == 0
    finally:
        get_config.cache_clear()
    assert (target_of(tmp_path) / "model.safetensors").read_bytes() == GOOD


def test_main_prints_json_entries_on_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fetch.main(["--checkpoint-dir", str(tmp_path)], FakeDownloader(), FILES) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("[") :])[0]["sha256"] == GOOD_SHA
