"""The 20 committed Stellenbosch fixtures: listed with their source, small, and free of EXIF."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

FOLDER = Path(__file__).resolve().parent.parent / "fixtures" / "stellenbosch"


def test_every_fixture_is_listed_small_and_has_no_exif() -> None:
    listed = json.loads((FOLDER / "SOURCE.json").read_text(encoding="utf-8"))["images"]
    assert sorted(i["file"] for i in listed) == sorted(p.name for p in FOLDER.glob("*.jpg"))
    assert len(listed) == 20 and {i["shows"] for i in listed} == {"pothole", "clean"}
    for item in listed:
        with Image.open(FOLDER / item["file"]) as image:
            assert image.width == 1280 and not image.getexif()
