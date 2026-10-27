"""File hashing shared by dataset identity and model integrity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path


def text_sha256(path: Path) -> str:
    """Hex SHA-256 of a text file with Windows line endings read as Unix ones, so a checkout on
    Windows and one on macOS or Linux give the same hash for the same text."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file, streamed so large files are never read whole into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
