"""Covered modules must import in the base install; the ml extra is only needed when they run.

CI installs the base and dev dependencies only. Test modules import `defects.train` before they
skip themselves for a missing torch, so an eager `import tqdm` or `import mlflow` at module level
breaks collection there while passing on a machine that has everything installed.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    "common.progress",
    "common.tracking",
    "defects.train",
    "defects.tile_dataset",
    "defects.tile_predict",
    "defects.runtime",
    "defects.infer",
    "defects.road_crop",
    "defects.tile_defects",
    "defects.patches",
    "perception.infer",
    "perception.detector",
    "perception.labels",
    "api.pipeline",
    "tools.predict_images",
    "tools.box_sizes",
)
# Block the optional packages as if they were not installed, then import everything.
SCRIPT = (
    "import sys\n"
    "for name in ('torch', 'torchvision', 'timm', 'tqdm', 'mlflow', 'nuscenes'):\n"
    "    sys.modules[name] = None\n"
    f"import {', '.join(MODULES)}\n"
)


def test_training_modules_import_without_the_optional_packages() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-800:]
