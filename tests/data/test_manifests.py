import json
from pathlib import Path

import pytest

from common.manifest import Manifest, ManifestEntry, sha256_file, verify

pytestmark = pytest.mark.data
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("rel", ["data/MANIFEST.json", "checkpoints/MANIFEST.json"])
def test_committed_manifests_are_valid(rel: str) -> None:
    Manifest.model_validate_json((ROOT / rel).read_text())


def test_baselines_placeholder_parses() -> None:
    assert json.loads((ROOT / "eval/baselines.json").read_text())["schema_version"] == 1


def test_verify_detects_tampering(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    entry = ManifestEntry(
        path="a.bin",
        sha256=sha256_file(f),
        size_bytes=3,
        source="test",
        version="0",
        license="unverified",
    )
    m = tmp_path / "MANIFEST.json"
    m.write_text(Manifest(entries=[entry]).model_dump_json())
    assert verify(m) == []
    f.write_bytes(b"abd")
    assert verify(m) == ["hash mismatch: a.bin"]
    f.unlink()
    assert verify(m) == ["missing: a.bin"]
