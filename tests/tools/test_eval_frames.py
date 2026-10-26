"""Frame-level alert collection and the export of never-reviewed alerts."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval.classifier import Alert
from fusion.schema import Defect, Object
from tools.eval_frames import FrameContext, collect_alerts, export_unreviewed, pack_mask


def _frame(root: Path, name: str, size: tuple[int, int] = (16, 8)) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(path)


def _context(name: str, night: bool = False) -> FrameContext:
    mask = np.ones((8, 16), dtype=bool)
    mask[:, 8:] = False
    return FrameContext(
        name, night, (Object(label="car", score=0.9, box=(0, 0, 2, 2)),), *pack_mask(mask)
    )


def test_mask_round_trips_through_packing() -> None:
    mask = np.random.default_rng(0).random((7, 13)) > 0.5

    context = FrameContext("f.jpg", False, (), *pack_mask(mask))

    assert np.array_equal(context.mask(), mask)


def test_collect_alerts_passes_objects_and_mask_to_the_finder_and_adds_road_cover(
    tmp_path: Path,
) -> None:
    _frame(tmp_path, "s/a.jpg")
    seen: dict[str, object] = {}

    def finder(frame: np.ndarray, **kw: object) -> list[Defect]:
        seen.update(kw)
        return [
            Defect(cls=1, name="D00", confidence=0.95, box=(0, 0, 8, 8), patch_row=0, patch_col=0)
        ]

    alerts = collect_alerts([_context("s/a.jpg", night=True)], tmp_path, finder=finder)

    assert alerts == [Alert("s/a.jpg", (0, 0, 8, 8), "D00", 0.95, True, 1.0)]
    assert len(seen["objects"]) == 1  # type: ignore[arg-type]
    assert seen["road_mask"].shape == (8, 16)  # type: ignore[attr-defined]


def test_export_writes_each_unreviewed_tile_once_with_a_blank_sheet_and_a_key(
    tmp_path: Path,
) -> None:
    _frame(tmp_path / "raw", "s/a.jpg")
    alerts = [
        Alert("s/a.jpg", (0, 0, 8, 8), "D00", 0.95, False, 1.0),
        Alert("s/a.jpg", (0, 0, 8, 8), "D10", 0.80, False, 1.0),  # same tile, second class
        Alert("s/a.jpg", (8, 0, 16, 8), "D40", 0.90, False, 0.5),  # reviewed: skipped
    ]
    reviewed = {("s/a.jpg", (8, 0, 16, 8)): "No crack"}

    written = export_unreviewed(
        alerts, tmp_path / "raw", tmp_path / "out", tmp_path / "key", reviewed, prefix="n2"
    )

    assert written == 1
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["annotations.csv", "n2_001.png"]
    rows = list(csv.DictReader((tmp_path / "out" / "annotations.csv").open()))
    assert [r["tile_id"] for r in rows] == ["n2_001"]
    assert rows[0]["call"] == ""
    key = json.loads((tmp_path / "key" / "key.json").read_text())
    assert key["n2_001"]["frame"] == "s/a.jpg"
    assert key["n2_001"]["classes"] == {"D00": 0.95, "D10": 0.8}
