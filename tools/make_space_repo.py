"""Lay out the release bundle as the repository a Hugging Face Docker Space expects.

Usage: python -m tools.make_space_repo --out DIR [--stubs]
A Space builds the file named Dockerfile at the top of its repository and reads its settings from
the front matter of the top-level README.md. This repo keeps those at docker/Dockerfile.space and
docker/space/README.md, so this tool copies the bundle (the files tools/build_bundle.py lists and
checks, from a clean checkout of one commit) into DIR and makes exactly these changes:
  1. Dockerfile at the top is docker/Dockerfile.space with the line "ARG WITH_ML=0" changed to
     "ARG WITH_ML=1", so the Space installs the real models (--stubs keeps 0).
  2. README.md at the top is docker/space/README.md (the Space front matter) instead of the
     project README.
  3. BUNDLE_MANIFEST.json is added: the commit, a SHA-256 per file and one hash over them, as
     tools/build_bundle.py writes it, plus the list of the changes above.
Nothing else is added or changed, so a reviewer who checked the bundle has checked what is pushed.
DIR must not exist or must be empty. Push it with:
  hf upload OWNER/SPACE DIR . --repo-type space
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tools.build_bundle import build_manifest, check_file, dirty_files, tracked_files

DOCKERFILE = "docker/Dockerfile.space"
SPACE_README = "docker/space/README.md"
ML_OFF = "ARG WITH_ML=0"
ML_ON = "ARG WITH_ML=1"


def head_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True, text=True
    ).stdout.strip()


def space_dockerfile(source: str, real_models: bool) -> str:
    """The Dockerfile text for the Space; the ML switch must be present exactly once."""
    if source.count(ML_OFF) != 1:
        raise SystemExit(f"{DOCKERFILE} must contain the line {ML_OFF!r} exactly once")
    return source.replace(ML_OFF, ML_ON) if real_models else source


def assemble(
    root: Path, out: Path, files: list[str], commit: str, real_models: bool = True
) -> dict[str, object]:
    """Copy the bundle files into out with the three changes; return the manifest written."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new or empty folder")
    for name in files:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    (out / "Dockerfile").write_text(
        space_dockerfile((root / DOCKERFILE).read_text("utf-8"), real_models), "utf-8", newline="\n"
    )
    shutil.copyfile(root / SPACE_README, out / "README.md")
    manifest = build_manifest(root, files, commit)
    manifest["space_changes"] = [
        f"Dockerfile = {DOCKERFILE}" + (f" with {ML_OFF!r} -> {ML_ON!r}" if real_models else ""),
        f"README.md = {SPACE_README}",
        "BUNDLE_MANIFEST.json added",
    ]
    (out / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stubs", action="store_true", help="keep WITH_ML=0 (stub models only)")
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
    manifest = assemble(root, args.out, files, head_commit(root), real_models=not args.stubs)
    print(f"{manifest['file_count']} files at {str(manifest['commit'])[:10]} written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
