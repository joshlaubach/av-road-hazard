"""tools/eval_nonroad.py: scores only the tiles that are not road, with the shipped rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from defects.infer import Classifier
from tools.eval_frames import FrameContext, pack_mask
from tools.eval_nonroad import score_frames

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torchvision") is None, reason="torch and torchvision not installed"
)

THRESHOLDS = (0.9, 0.75, 0.65, 0.75)
CFG = {
    "input_size": 16, "batch_size": 4, "num_workers": 0, "amp": "none", "channels_last": False,
    "tile_size": 100, "tile_overlap": 0.0,
}  # fmt: skip


def _red_detector() -> Any:
    torch = pytest.importorskip("torch")  # untyped here, as in the other model tests
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
    )
    with torch.no_grad():
        model[2].weight.zero_()
        model[2].weight[0, 0] = 10.0  # class 0 (D00) fires on a bright red channel
        model[2].bias.zero_()
    return model


def _context(tmp_path: Path, white: tuple[int, int], road_left: bool) -> FrameContext:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)  # (height, width, channels)
    frame[:, white[0] : white[1]] = 255
    Image.fromarray(frame).save(tmp_path / "f.png")
    mask = np.zeros((100, 200), dtype=bool)
    mask[:, : 100 if road_left else 0] = True
    return FrameContext("f.png", False, (), *pack_mask(mask))


@needs_torch
def test_only_tiles_without_road_are_scored_and_a_flag_on_one_is_counted(tmp_path: Path) -> None:
    context = _context(tmp_path, white=(0, 200), road_left=True)  # both tiles bright
    report = score_frames([context], tmp_path, Classifier(_red_detector(), CFG), THRESHOLDS)
    assert report["frames"] == 1 and report["non_road_tiles_scored"] == 1  # the right tile only
    band = report["clearly_not_road"]
    assert band["tiles"] == 1 and band["flagged"]["count"] == 1
    assert band["per_class"]["D00"]["count"] == 1


@needs_torch
def test_a_quiet_non_road_tile_is_not_a_false_positive(tmp_path: Path) -> None:
    context = _context(tmp_path, white=(0, 100), road_left=True)  # road tile bright, rest dark
    report = score_frames([context], tmp_path, Classifier(_red_detector(), CFG), THRESHOLDS)
    assert report["clearly_not_road"]["flagged"]["count"] == 0


@needs_torch
def test_a_frame_that_is_all_road_gives_empty_bands(tmp_path: Path) -> None:
    context = _context(tmp_path, white=(0, 200), road_left=True)
    mask = np.ones((100, 200), dtype=bool)
    full = FrameContext(context.path, False, (), *pack_mask(mask))
    report = score_frames([full], tmp_path, Classifier(_red_detector(), CFG), THRESHOLDS)
    assert report["non_road_tiles_scored"] == 0 and report["clearly_not_road"]["tiles"] == 0
