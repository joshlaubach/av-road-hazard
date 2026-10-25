"""File hashing shared by dataset identity and model integrity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file, streamed so large files are never read whole into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
