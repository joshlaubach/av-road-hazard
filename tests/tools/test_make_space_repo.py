"""Tests for tools/make_space_repo.py: the layout a Space needs, and nothing else changed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import make_space_repo as space

DOCKERFILE_TEXT = "FROM python:3.12-slim\nARG WITH_ML=0\nRUN echo hi\n"
FRONT_MATTER = "---\nsdk: docker\napp_port: 7860\n---\n"


def tree(root: Path) -> list[str]:
    files = {
        "README.md": "the project readme\n",
        "api/main.py": "x = 1\n",
        space.DOCKERFILE: DOCKERFILE_TEXT,
        space.SPACE_README: FRONT_MATTER,
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sorted(files)


def test_the_space_gets_a_top_level_dockerfile_with_the_real_models_switched_on(
    tmp_path: Path,
) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    space.assemble(root, out, tree(root), "abc")
    text = (out / "Dockerfile").read_text()
    assert "ARG WITH_ML=1" in text and "ARG WITH_ML=0" not in text
    assert text == DOCKERFILE_TEXT.replace("WITH_ML=0", "WITH_ML=1")


def test_stubs_keeps_the_dockerfile_as_it_is(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    space.assemble(root, out, tree(root), "abc", real_models=False)
    assert (out / "Dockerfile").read_text() == DOCKERFILE_TEXT


def test_the_top_level_readme_is_the_space_front_matter(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    space.assemble(root, out, tree(root), "abc")
    assert (out / "README.md").read_text() == FRONT_MATTER


def test_every_other_bundle_file_is_copied_unchanged(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    space.assemble(root, out, tree(root), "abc")
    assert (out / "api" / "main.py").read_text() == "x = 1\n"
    assert (out / space.DOCKERFILE).read_text() == DOCKERFILE_TEXT


def test_the_manifest_names_the_commit_and_every_change_made(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    names = tree(root)
    space.assemble(root, out, names, "abc123")
    manifest = json.loads((out / "BUNDLE_MANIFEST.json").read_text())
    assert manifest["commit"] == "abc123" and manifest["file_count"] == len(names)
    assert len(manifest["space_changes"]) == 3


def test_a_folder_that_already_holds_files_is_refused(tmp_path: Path) -> None:
    root, out = tmp_path / "repo", tmp_path / "out"
    out.mkdir()
    (out / "old.txt").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        space.assemble(root, out, tree(root), "abc")


def test_a_dockerfile_without_the_switch_is_refused() -> None:
    with pytest.raises(SystemExit, match="exactly once"):
        space.space_dockerfile("FROM x\n", real_models=True)


def test_the_real_dockerfile_has_the_switch_the_tool_needs() -> None:
    text = Path(space.DOCKERFILE).read_text("utf-8")
    assert space.space_dockerfile(text, real_models=True).count(space.ML_ON) == 1


def test_the_real_space_readme_declares_a_docker_space_on_port_7860() -> None:
    text = Path(space.SPACE_README).read_text("utf-8")
    assert "sdk: docker" in text and "app_port: 7860" in text
