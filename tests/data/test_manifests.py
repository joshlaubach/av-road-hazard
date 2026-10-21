import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.data
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "rel", ["data/MANIFEST.json", "checkpoints/MANIFEST.json", "eval/baselines.json"]
)
def test_committed_json_placeholders_parse(rel: str) -> None:
    """The committed manifest and baseline placeholders are valid JSON with schema_version 1."""
    doc = json.loads((ROOT / rel).read_text())
    assert doc["schema_version"] == 1
