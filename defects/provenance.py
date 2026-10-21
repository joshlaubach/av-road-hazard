"""Run provenance for training: dataset manifest checks, git state and the run directory.

Purpose: keep the bookkeeping that makes a training run reproducible (which data, which commit,
where the checkpoints go) apart from the training loop in defects/train.py.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from defects.dataset import Sample, build_dataset_manifest, manifest_identity


def manifest_path(split_file: Path) -> Path:
    """Sidecar file holding the dataset manifest persisted next to a split file."""
    return split_file.with_suffix(".manifest.json")


def _verify(samples: Sequence[Sample], split_file: Path, cfg: dict[str, Any]) -> str:
    """Hash once per run, compare the persisted manifest, and reuse its identity."""
    manifest = build_dataset_manifest(samples)
    if cfg["verify_hashes"]:
        expected = json.loads(manifest_path(split_file).read_text())
        mismatches = sorted(
            key
            for key in manifest.keys() | expected.keys()
            if manifest.get(key) != expected.get(key)
        )
        if mismatches:
            raise ValueError(f"Dataset manifest mismatch: {mismatches}")
    return manifest_identity(manifest)


def _git_sha() -> str:
    """Commit the code was run from, or "unknown" when git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool | None:
    """Include untracked files; None explicitly marks unavailable git metadata."""
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_directory(cfg: dict[str, Any], run_id: str, fold: int) -> Path:
    """Create <checkpoint_dir>/<run_id>/fold<N>; refuse bad run ids, paths in git and reruns."""
    if not run_id or run_id in (".", "..") or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single nonempty path component")
    if not cfg["checkpoint_dir"]:
        raise ValueError("Set checkpoint_dir outside git")
    root = Path(cfg["checkpoint_dir"]).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError("checkpoint_dir must be outside git")
    directory = root / run_id / f"fold{fold}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory
