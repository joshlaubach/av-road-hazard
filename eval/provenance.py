"""Provenance block stored with every evaluation report: what the numbers were measured on.

Records file hashes (checkpoints, review and probe files, model weights), the thresholds, the git
commit and whether the tree had uncommitted changes, and the UTC date. Depends only on the stdlib
and common.hashing.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.hashing import file_sha256


def _git(repo: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def provenance(
    files: Mapping[str, Path], *, thresholds: Mapping[str, float], repo: Path
) -> dict[str, Any]:
    """The provenance block; a file that does not exist is an error, not a blank hash."""
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--", ".", ":!notebooks") if commit else None
    return {
        "files": {name: file_sha256(path) for name, path in files.items()},
        "thresholds": dict(thresholds),
        "git_commit": commit or "unknown",
        "git_dirty": None if status is None else bool(status),
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
