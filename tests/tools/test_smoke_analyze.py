"""The /analyze smoke check: valid records, failures and unknown road state are counted apart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api.pipeline import Analysis, Pipeline, decode_image
from defects.infer import DefectsUnavailable
from fusion.schema import Defect, HazardRecord, Object
from perception.detector import DetectorUnavailable
from tools.smoke_analyze import run_smoke

DEFECT = Defect(cls=1, name="x", confidence=0.9, box=(1, 1, 5, 5), patch_row=0, patch_col=0)


class FakePipeline(Pipeline):
    """Returns a scripted outcome per image name: 'ok', 'no_road', 'detector' or 'defects'."""

    def __init__(self, script: dict[str, str]) -> None:
        super().__init__()
        self.script = script

    def analyze_checked(self, data: bytes, source: str = "upload") -> Analysis:
        decode_image(data)  # a bad file raises InvalidImage, as in the real pipeline
        name = self.script[str(len(data))]
        if name == "detector":
            raise DetectorUnavailable("down")
        if name == "defects":
            raise DefectsUnavailable("down")
        road = name == "ok"
        record = HazardRecord(
            frame_id=f"f{len(data)}",
            source="upload",
            image_size=(16, 8),
            objects=[Object(label="car", score=0.9, box=(0, 0, 3, 3))],
            defects=[
                {
                    "name": "x",
                    "confidence": 0.9,
                    "box": DEFECT.box,
                    "severity": "unknown",
                    "in_road": True if road else None,
                    "near_objects": [],
                }
            ],
            hazard_score=0.5,
            meta={"time_of_day": "unknown", "location": None},
        )
        return Analysis(record, road)


def _write(folder: Path, name: str, pad: int) -> int:
    path = folder / name
    Image.fromarray(np.full((8, 16, 3), 100, dtype=np.uint8)).save(path)
    with path.open("ab") as handle:  # trailing bytes change the length, which keys the script
        handle.write(b"\0" * pad)
    return path.stat().st_size


def test_counts_each_outcome_separately_and_writes_overlays(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    sizes = {
        "ok": _write(folder, "a.png", 0),
        "no_road": _write(folder, "b.png", 7),
        "detector": _write(folder, "c.png", 13),
        "defects": _write(folder, "d.png", 21),
    }
    (folder / "bad.png").write_bytes(b"not an image")
    script = {str(size): outcome for outcome, size in sizes.items()}
    out = tmp_path / "out"
    summary = run_smoke(FakePipeline(script), sorted(folder.glob("*.png")), out)
    assert summary["images"] == 5
    assert summary["valid"] == 2
    assert summary["road_unknown"] == 1
    assert summary["detector_unavailable"] == 1
    assert summary["defects_unavailable"] == 1
    assert summary["invalid_image"] == 1
    assert summary["failures"] == 3
    assert (out / "a.png.overlay.png").is_file()
    assert (out / "a.png.record.json").is_file()
    assert not (out / "c.png.overlay.png").exists()
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["valid"] == 2
    assert saved["per_image"]["b.png"] == "road_unknown"


def test_a_pipeline_with_stubs_on_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.smoke_analyze import main

    monkeypatch.setenv("USE_STUBS", "1")
    from common import config

    config.get_config.cache_clear()
    try:
        assert main([str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    finally:
        config.get_config.cache_clear()


def test_images_that_share_a_stem_keep_separate_outputs(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    size = _write(folder, "frame.png", 0)
    Image.open(folder / "frame.png").convert("RGB").save(folder / "frame.jpg")
    jpg_size = (folder / "frame.jpg").stat().st_size
    script = {str(size): "ok", str(jpg_size): "ok"}
    out = tmp_path / "out"
    images = sorted(folder.iterdir())
    run_smoke(FakePipeline(script), images, out)
    for name in ("frame.png", "frame.jpg"):
        assert (out / f"{name}.overlay.png").is_file()
        assert (out / f"{name}.record.json").is_file()
