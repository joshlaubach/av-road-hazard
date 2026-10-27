"""tools/make_example.py: the display copy is shrunk, the trace is from the full-size image."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from api.models import TracedAnalysis
from tools.make_example import make_example


def _photo(path: Path, size: tuple[int, int]) -> Path:
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def test_the_example_holds_a_shrunk_display_photo_and_the_full_size_trace(tmp_path: Path) -> None:
    photo = _photo(tmp_path / "wide.png", (3200, 1000))
    folder = make_example(
        photo, example_id="demo", title="T", credit="C", note="N", out_dir=tmp_path / "out",
        display_width=1600,
    )  # fmt: skip
    shown = Image.open(folder / "display.jpg")
    assert shown.size == (1600, 500) and not shown.getexif()
    data = json.loads((folder / "example.json").read_text(encoding="utf-8"))
    assert data["analyzed_size"] == [3200, 1000] and data["display_size"] == [1600, 500]
    analysis = TracedAnalysis.model_validate(data["analysis"])
    assert analysis.record.image_size == (3200, 1000)  # the trace is of the full-size image
    assert (data["title"], data["credit"], data["note"]) == ("T", "C", "N")


def test_a_photo_smaller_than_the_display_width_is_not_enlarged(tmp_path: Path) -> None:
    photo = _photo(tmp_path / "small.png", (800, 400))
    folder = make_example(
        photo, example_id="s", title="T", credit="C", note="", out_dir=tmp_path, display_width=1600
    )
    assert Image.open(folder / "display.jpg").size == (800, 400)
