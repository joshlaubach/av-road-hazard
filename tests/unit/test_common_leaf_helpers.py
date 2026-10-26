"""The shared leaf helpers in common/ and the names that still re-export them."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from common.device import select_device
from common.geometry import box_iou
from common.hashing import file_sha256

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("a", "b", "want"),
    [
        ((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0), 1.0),
        ((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0), 0.0),
        ((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0), 1 / 3),
        ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.0),
    ],
)
def test_box_iou_and_old_names_agree(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], want: float
) -> None:
    from eval.models import iou
    from fusion.fuse import box_iou as fuse_box_iou

    assert box_iou(a, b) == pytest.approx(want)
    assert iou(a, b) == box_iou(a, b) == fuse_box_iou(a, b)


def test_file_sha256_streams_a_multi_megabyte_file(tmp_path: Path) -> None:
    from defects.dataset import file_sha256 as dataset_file_sha256

    payload = (b"road-hazard" * 1000) * 500  # about 5.5 MB
    path = tmp_path / "big.bin"
    path.write_bytes(payload)

    want = hashlib.sha256(payload).hexdigest()
    assert file_sha256(path) == want
    assert dataset_file_sha256(path) == want


def test_select_device_returns_a_known_device() -> None:
    pytest.importorskip("torch")
    assert select_device() in {"cuda", "mps", "cpu"}


def test_old_select_device_name_is_the_same_function() -> None:
    from defects import runtime

    assert runtime.select_device is select_device


def test_all_constant_is_importable_from_both_homes() -> None:
    from defects import labels, voc

    assert voc.ALL is labels.ALL
    assert labels.ALL == "ALL"


def test_common_helpers_import_with_torch_blocked() -> None:
    script = (
        "import sys\n"
        "for name in ('torch', 'torchvision'):\n"
        "    sys.modules[name] = None\n"
        "import common.geometry, common.hashing, common.device\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
