"""The hosted-corpus data file (web/src/lib/demo_corpus.json): decided sources, each credited."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CORPUS_FILE = Path("web/src/lib/demo_corpus.json")


def _corpus() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    return data


def test_the_decided_corpus_is_40_rdd2022_and_20_stellenbosch_images() -> None:
    sources = {s["id"]: s for s in _corpus()["sources"]}
    assert (sources["rdd2022"]["images"], sources["stellenbosch"]["images"]) == (40, 20)
    assert all(s["include"] for s in sources.values())


def test_every_source_names_its_authors_and_its_license_status() -> None:
    for source in _corpus()["sources"]:
        assert source["authors"].strip()
        assert source["license_note"].strip()
        assert source["countries"]


def test_the_notice_says_non_commercial_and_not_verified() -> None:
    notice = _corpus()["notice"]
    assert "non-commercial" in notice and "not verified" in notice


def test_no_us_images_in_the_rdd2022_source() -> None:
    rdd = next(s for s in _corpus()["sources"] if s["id"] == "rdd2022")
    assert "United States" not in rdd["countries"] and "US" not in rdd["countries"]
