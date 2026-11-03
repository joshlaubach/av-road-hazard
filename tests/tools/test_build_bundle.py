"""Tests for tools/build_bundle.py: the checks, the manifest, and that the real bundle passes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import build_bundle as bundle


def test_an_unlisted_image_or_weight_is_refused() -> None:
    assert bundle.check_file("api/photo.jpg", b"")
    assert bundle.check_file("rag/model.safetensors", b"")


def test_the_two_shrunk_example_photos_are_allowed() -> None:
    assert bundle.check_file("web/public/examples/stellenbosch-potholes/display.jpg", b"") == []
    assert bundle.check_file("web/public/examples/stellenbosch-potholes/full.jpg", b"")


def test_a_nuscenes_file_is_called_out() -> None:
    problems = bundle.check_file("web/public/nuscenes_frame.png", b"")
    assert any("nuScenes" in p for p in problems)


@pytest.mark.parametrize(
    "line",
    [
        "key = 'sk-ant-api03-abcdefghijklmnop'",
        "HF = 'hf_abcdefghijklmnopqrstuvwxyz'",
        'api_key = "abcdefghijklmnop1234"',
        "-----BEGIN RSA PRIVATE KEY-----",
    ],
)
def test_secret_looking_text_is_refused(line: str) -> None:
    assert bundle.check_file("api/x.py", line.encode())


def test_an_empty_key_setting_is_not_a_secret() -> None:
    assert bundle.check_file("api/x.py", b'api_key = os.environ["QDRANT_API_KEY"]') == []


@pytest.mark.parametrize("text", [r"C:\Users\someone\data", "/home/someone/x", "/Users/someone/x"])
def test_a_hardcoded_machine_path_is_refused(text: str) -> None:
    assert any("absolute path" in p for p in bundle.check_file("tools/x.py", text.encode()))


def test_the_manifest_hash_changes_when_a_file_changes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one")
    first = bundle.build_manifest(tmp_path, ["a.txt"], "abc")
    (tmp_path / "a.txt").write_text("two")
    second = bundle.build_manifest(tmp_path, ["a.txt"], "abc")
    assert first["bundle_sha256"] != second["bundle_sha256"] and first["file_count"] == 1


def test_a_modified_bundle_file_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "x.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-qm", "init")
    (tmp_path / "api" / "x.py").write_text("x = 2\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="differ from the last commit"):
        bundle.main(["--out", str(tmp_path / "m.json")])


def test_the_real_bundle_passes_its_own_checks() -> None:
    root = Path.cwd()
    files = bundle.tracked_files(root)
    assert "api/main.py" in files and "docker/space/start.sh" in files
    problems = [p for f in files for p in bundle.check_file(f, (root / f).read_bytes())]
    assert problems == []
