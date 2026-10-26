"""Inference must not depend on the training module."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    "import sys\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "for name in ('torch', 'torchvision', 'timm', 'tqdm', 'mlflow', 'nuscenes'):\n"
    "    sys.modules[name] = None\n"
    "import defects.infer\n"
    "assert 'defects.train' not in sys.modules, 'defects.infer pulled in defects.train'\n"
)


def test_importing_infer_does_not_import_train() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-800:]
