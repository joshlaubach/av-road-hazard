"""Manifest format for data/MANIFEST.json and checkpoints/MANIFEST.json.

The manifests are tracked in git; the files they describe are not. Download and training
scripts write entries; verify() checks files on disk against recorded hashes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path relative to the manifest's directory")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    source: str = Field(description="Dataset name, URL, or training run id")
    version: str = Field(description="Dataset version or git SHA of the producing code")
    license: str = Field(
        description="License or terms; 'unverified' is allowed and must be explicit"
    )
    retrieved: str | None = Field(default=None, description="ISO date")


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    entries: list[ManifestEntry] = []


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(manifest_path: Path) -> list[str]:
    """Return a list of problems (missing file, size or hash mismatch). Empty means OK."""
    manifest = Manifest.model_validate_json(manifest_path.read_text())
    problems: list[str] = []
    for e in manifest.entries:
        p = manifest_path.parent / e.path
        if not p.exists():
            problems.append(f"missing: {e.path}")
        elif p.stat().st_size != e.size_bytes:
            problems.append(f"size mismatch: {e.path}")
        elif sha256_file(p) != e.sha256:
            problems.append(f"hash mismatch: {e.path}")
    return problems
