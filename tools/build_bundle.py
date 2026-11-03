"""List the exact files that go into the hosted bundle, check them, and write a manifest of hashes.

Usage: python -m tools.build_bundle [--out bundle_manifest.json]
The bundle is every file git tracks under the paths docker/Dockerfile.space copies. The check stops
(exit 1) when it finds, in that set: an image, weight or data file that is not on the allow list; a
nuScenes image; text that looks like a secret; or a hardcoded absolute path of someone's machine.
It also refuses to run when any bundle file differs from the last commit, so the manifest always
describes a commit, and a reviewer can check the same commit. The manifest holds the commit, one
SHA-256 per file and one hash over all of them. Nothing here is a substitute for the review itself
(docs/RELEASE_REVIEW.md).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

# What docker/Dockerfile.space copies into the image (keep in step with it).
BUNDLE_PATHS = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "requirements.lock",
    "common",
    "perception",
    "defects",
    "fusion",
    "rag",
    "api",
    "eval",
    "tools",
    "configs",
    "web",
    "docker/space",
    "docker/Dockerfile.space",
)
BINARY_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".pt", ".pth", ".safetensors", ".onnx", ".ckpt",
    ".npy", ".npz", ".parquet", ".zip", ".tar", ".gz", ".mp4", ".pkl", ".bin", ".h5",
}  # fmt: skip
# Files that may be in the bundle although they are images: the two shrunk example photos, which are
# CC0 Stellenbosch photos shown with credit (docs/DATA_AND_LICENSES.md).
ALLOWED_BINARY = ("web/public/examples/*/display.jpg",)
SECRET_PATTERNS = {
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "AWS key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "key assigned in code": re.compile(
        r"""(?i)\b(api_?key|secret|token|password)\b\s*[:=]\s*["'][A-Za-z0-9_\-]{16,}["']"""
    ),
}
PATH_PATTERN = re.compile(r"[A-Za-z]:\\Users\\|/Users/[a-z]|/home/[a-z]")
# This file lists the patterns it looks for, so it cannot be checked against them.
SELF = "tools/build_bundle.py"
# Lock files and minified output are long machine-written lines that look like tokens.
SKIP_TEXT = ("*.lock", "package-lock.json", "*.min.js")


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *BUNDLE_PATHS],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return sorted(p for p in out.split("\0") if p)


def dirty_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *BUNDLE_PATHS],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return [line[3:] for line in out.splitlines()]


def is_allowed_binary(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in ALLOWED_BINARY)


def check_file(path: str, data: bytes) -> list[str]:
    """Problems found in one file; empty when it is fine."""
    suffix = Path(path).suffix.lower()
    if suffix in BINARY_SUFFIXES:
        if is_allowed_binary(path):
            return []
        problems = [f"{path}: image, weight or data file is not on the allow list"]
        if "nuscenes" in path.lower():
            problems.append(f"{path}: looks like a nuScenes file")
        return problems
    if path == SELF or any(fnmatch.fnmatch(Path(path).name, p) for p in SKIP_TEXT):
        return []
    text = data.decode("utf-8", errors="replace")
    problems = [f"{path}: {name}" for name, rx in SECRET_PATTERNS.items() if rx.search(text)]
    if PATH_PATTERN.search(text):
        problems.append(f"{path}: hardcoded absolute path")
    return problems


def build_manifest(root: Path, files: Sequence[str], commit: str) -> dict[str, object]:
    hashes = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}
    overall = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"commit": commit, "file_count": len(files), "bundle_sha256": overall, "files": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bundle_manifest.json"))
    args = parser.parse_args(argv)
    root = Path.cwd()

    dirty = dirty_files(root)
    if dirty:
        raise SystemExit("bundle files differ from the last commit:\n" + "\n".join(dirty))
    files = tracked_files(root)
    problems = [p for f in files for p in check_file(f, (root / f).read_bytes())]
    if problems:
        print("\n".join(problems))
        return 1
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True, text=True
    ).stdout.strip()
    args.out.write_text(json.dumps(build_manifest(root, files, commit), indent=2), "utf-8")
    print(f"{len(files)} files at {commit[:10]}, manifest written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
