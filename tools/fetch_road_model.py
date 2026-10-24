"""Download the pinned comma10k road-segmentation model into the checkpoint directory.

Usage: python -m tools.fetch_road_model [--checkpoint-dir DIR] [--dry-run]
The target is <checkpoint dir>/road_mask/comma10k_unet; the checkpoint dir defaults to
paths.checkpoint_dir from the config (env CHECKPOINT_DIR). Only the two files pinned in
defects/road_crop.py are fetched, at the pinned revision. A file already present with the right
SHA-256 is skipped. Each download lands in a fresh temporary folder next to the target, is checked
against the pinned SHA-256 and only then moved into place; a mismatch deletes the download, leaves
existing files alone and exits 1. Pickle-based file types are never fetched. On success the manifest
entries are printed as JSON so a maintainer can paste them into checkpoints/MANIFEST.json.
Exit codes: 0 ok, 1 hash mismatch or download failure, 2 usage error. --dry-run fetches nothing.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from common.config import get_config
from defects.dataset import file_sha256
from defects.road_crop import MODEL_FILES, MODEL_REPO, MODEL_REVISION, MODEL_SUBDIR

SOURCE = f"huggingface.co/{MODEL_REPO}"
LICENSE = "MIT per model card (not verified by counsel)"
# Pickle-based formats can run code when loaded, so they are never downloaded.
FORBIDDEN_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".ckpt", ".pickle")

# (repo_id, filename, revision, local_dir) -> path of the downloaded file
Downloader = Callable[[str, str, str, Path], Path]


class FetchError(Exception):
    """A download failed or did not match its pinned hash."""


def hf_download(repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
    """Fetch one file from the Hugging Face Hub (imported lazily; needs the `ml` extra)."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, local_dir=str(local_dir)
        )
    )


def check_filenames(names: Mapping[str, str]) -> None:
    for name in names:
        if name.lower().endswith(FORBIDDEN_SUFFIXES):
            raise ValueError(f"refusing to fetch pickle-style file: {name}")


def is_verified(path: Path, expected: str) -> bool:
    return path.is_file() and file_sha256(path) == expected


def fetch_one(target: Path, name: str, expected: str, download: Downloader) -> None:
    """Download name into a new temp dir beside target, verify it, then move it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".fetch_", dir=target.parent))
    try:
        try:
            got = download(MODEL_REPO, name, MODEL_REVISION, scratch)
        except Exception as exc:  # network and hub errors are many types; report them uniformly
            raise FetchError(f"download of {name} failed: {exc}") from exc
        actual = file_sha256(got)
        if actual != expected:
            raise FetchError(f"{name}: sha256 mismatch, expected {expected}, got {actual}")
        target.mkdir(parents=True, exist_ok=True)
        os.replace(got, target / name)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def manifest_entry(
    target: Path, checkpoint_dir: Path, name: str, expected: str
) -> dict[str, object]:
    path = target / name
    return {
        "path": path.relative_to(checkpoint_dir).as_posix(),
        "sha256": expected,
        "size_bytes": path.stat().st_size,
        "source": SOURCE,
        "version": MODEL_REVISION,
        "license": LICENSE,
        "retrieved": datetime.datetime.now(datetime.UTC).date().isoformat(),
    }


def fetch_model(
    checkpoint_dir: Path,
    files: Mapping[str, str] = MODEL_FILES,
    download: Downloader = hf_download,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    """Make every pinned file present and verified; return its manifest entries."""
    check_filenames(files)
    target = checkpoint_dir / MODEL_SUBDIR
    entries: list[dict[str, object]] = []
    for name, expected in files.items():
        if is_verified(target / name, expected):
            print(f"{name}: already present and verified")
        elif dry_run:
            print(f"{name}: would fetch {MODEL_REPO}@{MODEL_REVISION}")
            continue
        else:
            print(f"{name}: fetching")
            fetch_one(target, name, expected, download)
        entries.append(manifest_entry(target, checkpoint_dir, name, expected))
    return entries


def main(
    argv: list[str] | None = None,
    download: Downloader = hf_download,
    files: Mapping[str, str] = MODEL_FILES,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    checkpoint_dir = args.checkpoint_dir or Path(get_config()["paths"]["checkpoint_dir"])
    try:
        entries = fetch_model(checkpoint_dir, files, download, args.dry_run)
    except FetchError as exc:
        print(f"error: {exc}")
        return 1
    if entries and not args.dry_run:
        print(json.dumps(entries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
