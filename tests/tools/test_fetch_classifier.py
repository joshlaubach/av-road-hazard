"""Tests for tools/fetch_classifier.py: pinned hashes decide what is accepted; no network."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from tools import fetch_classifier as fc
from tools.fetch_road_model import Downloader

GOOD = {"best.pt": b"weights", "best.json": b'{"config": {}}'}


def pins(tmp_path: Path, contents: dict[str, bytes] = GOOD) -> Path:
    path = tmp_path / "pins.json"
    files = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    path.write_text(json.dumps({"revision": "main", "files": files}))
    return path


def serving(contents: dict[str, bytes]) -> Downloader:
    def download(repo: str, name: str, revision: str, local_dir: Path) -> Path:
        target = local_dir / name
        target.write_bytes(contents[name])
        return target

    return download


def run(tmp_path: Path, download: Downloader, *extra: str) -> int:
    dest = tmp_path / "dest"
    argv = ["--dest", str(dest), "--repo", "me/classifier", *extra]
    return fc.main(argv, download=download, pins_file=pins(tmp_path))


def test_good_files_are_placed_and_no_scratch_folder_is_left(tmp_path: Path) -> None:
    assert run(tmp_path, serving(GOOD)) == 0
    dest = tmp_path / "dest"
    assert (dest / "best.pt").read_bytes() == b"weights"
    assert sorted(p.name for p in dest.iterdir()) == ["best.json", "best.pt"]


def test_a_file_with_the_wrong_hash_is_refused_and_nothing_is_placed(tmp_path: Path) -> None:
    tampered = {**GOOD, "best.pt": b"swapped"}
    assert run(tmp_path, serving(tampered)) == 1
    assert list((tmp_path / "dest").iterdir()) == []


def test_a_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    run(tmp_path, serving(GOOD))

    def never(*args: object) -> Path:
        raise AssertionError("downloaded a file that was already verified")

    assert run(tmp_path, never) == 0


def test_a_download_error_is_reported_and_exits_one(tmp_path: Path) -> None:
    def broken(*args: object) -> Path:
        raise ConnectionError("no route")

    assert run(tmp_path, broken) == 1


def test_a_dry_run_fetches_nothing(tmp_path: Path) -> None:
    def never(*args: object) -> Path:
        raise AssertionError("dry run downloaded")

    assert run(tmp_path, never, "--dry-run") == 0
    assert not (tmp_path / "dest").exists()


def test_without_a_repo_it_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fc.REPO_ENV, raising=False)
    assert fc.main(["--dest", str(tmp_path)], pins_file=pins(tmp_path)) == 2


def test_the_committed_pins_name_the_two_classifier_files_with_real_hashes() -> None:
    revision, files = fc.load_pins()
    assert revision and set(files) == {"best.pt", "best.json"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in files.values())
