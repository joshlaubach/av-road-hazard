"""Provenance: what a report was measured on, so old numbers stay comparable or clearly are not."""

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from eval.provenance import provenance


def test_files_are_hashed_by_name(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"abc")

    got = provenance({"checkpoint": a}, thresholds={"D00": 0.9}, repo=tmp_path)

    assert got["files"] == {"checkpoint": hashlib.sha256(b"abc").hexdigest()}
    assert got["thresholds"] == {"D00": 0.9}


def test_a_missing_file_is_an_error_not_a_blank(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        provenance({"x": tmp_path / "nope"}, thresholds={}, repo=tmp_path)


def test_outside_a_git_repository_the_commit_is_unknown(tmp_path: Path) -> None:
    got = provenance({}, thresholds={}, repo=tmp_path)

    assert got["git_commit"] == "unknown"
    assert got["git_dirty"] is None


def test_inside_a_git_repository_the_commit_and_dirty_flag_are_recorded(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")

    clean = provenance({}, thresholds={}, repo=tmp_path)
    (tmp_path / "new.txt").write_text("x")
    dirty = provenance({}, thresholds={}, repo=tmp_path)

    assert re.fullmatch(r"[0-9a-f]{40}", clean["git_commit"])
    assert clean["git_dirty"] is False
    assert dirty["git_dirty"] is True


def test_the_date_is_utc_iso(tmp_path: Path) -> None:
    got = provenance({}, thresholds={}, repo=tmp_path)

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", got["date_utc"])


def test_changes_under_notebooks_do_not_make_the_tree_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "run.ipynb").write_text("{}")

    assert provenance({}, thresholds={}, repo=tmp_path)["git_dirty"] is False
