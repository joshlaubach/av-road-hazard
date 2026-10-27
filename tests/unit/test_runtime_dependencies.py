"""The Docker image installs only the runtime dependencies, so every package the API imports at
startup must be declared there (the dev tools install more, which hides a missing one)."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_api_runtime_dependencies_cover_what_the_api_imports() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {d.split(">")[0].split("=")[0].strip().lower() for d in project["dependencies"]}
    assert {"qdrant-client", "anthropic", "httpx", "fastapi", "numpy", "pillow"} <= declared
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8").lower()
    for name in ("qdrant-client==", "anthropic==", "httpx=="):
        assert name in lock, f"{name} missing from requirements.lock (the Docker image)"
