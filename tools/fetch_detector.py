"""Download a pinned COCO detector into the checkpoint directory, as a safetensors file.

Usage: python -m tools.fetch_detector --model NAME [--checkpoint-dir DIR] [--dry-run]
NAME is one of the models pinned in perception/detector.py. The published torchvision .pth file is
downloaded over HTTPS into a fresh temporary folder next to the target, its SHA-256 is checked
against the pinned one, and only then is it opened, with torch's weights-only loader (plain
tensors, no code), and saved as <checkpoint dir>/detector/NAME.safetensors. That file's SHA-256 must
match the pinned one too before it is moved into place. A mismatch deletes everything it fetched,
leaves existing files alone and exits 1. A file already present with the right hash is skipped.
On success the manifest entry is printed as JSON for checkpoints/MANIFEST.json.
Exit codes: 0 ok, 1 hash mismatch or download failure, 2 usage error. --dry-run fetches nothing.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from common.config import get_config
from perception.detector import DETECTOR_SUBDIR, DETECTORS, DetectorSpec

SOURCE = "download.pytorch.org (torchvision COCO_V1 weights)"
LICENSE = "BSD-3 (torchvision code); trained on COCO (not verified by counsel)"
# The largest pinned file is about 175 MB; anything far past that is not what was pinned.
MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60
_CHUNK_BYTES = 1024 * 1024

# (url, destination file) -> None; the file must exist afterwards
Downloader = Callable[[str, Path], None]


class FetchError(Exception):
    """A download failed or did not match its pinned hash."""


def http_download(url: str, dest: Path) -> None:
    """Stream url to dest, giving up past MAX_DOWNLOAD_BYTES."""
    total = 0
    with (
        urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        dest.open("wb") as out,
    ):
        while chunk := response.read(_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise OSError(f"download is larger than {MAX_DOWNLOAD_BYTES} bytes")
            out.write(chunk)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def convert_to_safetensors(source: Path, dest: Path) -> None:
    """Re-save a torch state dict as safetensors; weights_only means no pickled code is run."""
    import torch
    from safetensors.torch import save_file

    state = torch.load(source, map_location="cpu", weights_only=True)
    save_file({name: tensor.contiguous() for name, tensor in state.items()}, str(dest))


def fetch_detector(
    checkpoint_dir: Path, spec: DetectorSpec, download: Downloader = http_download
) -> dict[str, object]:
    """Make spec's safetensors file present and verified; return its manifest entry."""
    if not spec.url.startswith("https://"):
        raise ValueError(f"refusing to fetch a non-https url: {spec.url}")
    target = checkpoint_dir / DETECTOR_SUBDIR / spec.filename
    if target.is_file() and _sha256_of(target) == spec.sha256:
        print(f"{spec.filename}: already present and verified")
    else:
        print(f"{spec.filename}: fetching {spec.url}")
        _fetch_into(target, spec, download)
    return {
        "path": f"{DETECTOR_SUBDIR}/{spec.filename}",
        "sha256": spec.sha256,
        "size_bytes": target.stat().st_size,
        "source": SOURCE,
        "version": spec.url.rsplit("/", 1)[-1],
        "license": LICENSE,
        "retrieved": datetime.datetime.now(datetime.UTC).date().isoformat(),
    }


def _fetch_into(target: Path, spec: DetectorSpec, download: Downloader) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".fetch_", dir=target.parent))
    try:
        source = scratch / "source.pth"
        try:
            download(spec.url, source)
        except Exception as exc:  # network errors are many types; report them uniformly
            raise FetchError(f"download of {spec.name} failed: {exc}") from exc
        got = _sha256_of(source)
        if got != spec.source_sha256:
            raise FetchError(
                f"{spec.name}: sha256 mismatch, expected {spec.source_sha256}, got {got}"
            )
        converted = scratch / spec.filename
        convert_to_safetensors(source, converted)
        got = _sha256_of(converted)
        if got != spec.sha256:
            raise FetchError(
                f"{spec.name}: converted file sha256 mismatch, expected {spec.sha256}, got {got}"
            )
        os.replace(converted, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(
    argv: list[str] | None = None,
    download: Downloader = http_download,
    specs: Mapping[str, DetectorSpec] = DETECTORS,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.model not in specs:
        print(f"error: unknown model {args.model!r}; choose one of {sorted(specs)}")
        return 2
    spec = specs[args.model]
    checkpoint_dir = args.checkpoint_dir or Path(get_config()["paths"]["checkpoint_dir"])
    if args.dry_run:
        print(f"{spec.filename}: would fetch {spec.url}")
        return 0
    try:
        entry = fetch_detector(checkpoint_dir, spec, download)
    except FetchError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
