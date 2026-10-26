"""load_checkpoint refuses a checkpoint whose bytes or class order differ from its sidecar."""

import json
from pathlib import Path

import pytest

from defects.dataset import file_sha256
from defects.labels import CODES
from defects.model import load_checkpoint

pytest.importorskip("torch")


def _write(tmp_path: Path, class_codes: list[str]) -> Path:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    sidecar = {"class_codes": class_codes, "checkpoint_sha256": file_sha256(checkpoint)}
    checkpoint.with_suffix(".json").write_text(json.dumps(sidecar))
    return checkpoint


def test_tampered_checkpoint_bytes_raise_provenance_mismatch(tmp_path: Path) -> None:
    checkpoint = _write(tmp_path, list(CODES))
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checkpoint provenance mismatch"):
        load_checkpoint(checkpoint, {})


def test_wrong_class_codes_raise_provenance_mismatch(tmp_path: Path) -> None:
    checkpoint = _write(tmp_path, list(reversed(CODES)))
    with pytest.raises(ValueError, match="Checkpoint provenance mismatch"):
        load_checkpoint(checkpoint, {})
