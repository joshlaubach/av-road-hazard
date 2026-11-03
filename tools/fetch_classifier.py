"""Download the tile classifier (best.pt and best.json) from a private Hugging Face model repo.

Usage: python -m tools.fetch_classifier --dest DIR [--repo OWNER/NAME] [--dry-run]
The repo defaults to the environment variable CLASSIFIER_REPO; a private repo is read with the
token in HF_TOKEN. Which files, their SHA-256 and the revision are pinned in
configs/hosted_classifier.json, so a swapped or damaged file is refused. Each download lands in
a fresh temporary folder next to the destination, is checked and only then moved into place; a
mismatch deletes it, leaves existing files alone and exits 1. A file already present with the
right hash is skipped. The detector and the road model are public and have their own tools
(tools/fetch_detector.py, tools/fetch_road_model.py); this one is only for the classifier, which
was trained for this project.
Exit codes: 0 ok, 1 hash mismatch or download failure, 2 usage error. --dry-run fetches nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from defects.dataset import file_sha256
from tools.fetch_road_model import Downloader, FetchError, hf_download, is_verified

PINS_FILE = Path("configs/hosted_classifier.json")
REPO_ENV = "CLASSIFIER_REPO"


def load_pins(pins_file: Path | None = None) -> tuple[str, dict[str, str]]:
    """(revision, {file name: sha256}) from the pins file."""
    data = json.loads((pins_file or PINS_FILE).read_text("utf-8"))
    return str(data["revision"]), {str(k): str(v) for k, v in data["files"].items()}


def fetch_one(
    dest: Path, repo: str, name: str, revision: str, expected: str, download: Downloader
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".fetch_", dir=dest))
    try:
        try:
            got = download(repo, name, revision, scratch)
        except Exception as exc:  # hub and network errors come in many types; report them uniformly
            raise FetchError(f"download of {name} failed: {exc}") from exc
        actual = file_sha256(got)
        if actual != expected:
            raise FetchError(f"{name}: sha256 mismatch, expected {expected}, got {actual}")
        os.replace(got, dest / name)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(
    argv: list[str] | None = None,
    download: Downloader = hf_download,
    pins_file: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--repo", default=os.environ.get(REPO_ENV, ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.repo:
        print(f"error: give --repo or set {REPO_ENV} (a Hugging Face model repo, OWNER/NAME)")
        return 2
    revision, files = load_pins(pins_file)
    try:
        for name, expected in files.items():
            if is_verified(args.dest / name, expected):
                print(f"{name}: already present and verified")
            elif args.dry_run:
                print(f"{name}: would fetch {args.repo}@{revision}")
            else:
                print(f"{name}: fetching")
                fetch_one(args.dest, args.repo, name, revision, expected, download)
    except FetchError as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
