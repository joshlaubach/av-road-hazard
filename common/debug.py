"""Optional dumps of pipeline intermediates to DEBUG_DUMP_DIR, keyed by request_id."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.config import get_config
from common.logging import request_id_var


def dump_dir() -> Path | None:
    raw = get_config()["paths"]["debug_dump_dir"]
    if not raw:
        return None
    path = Path(raw) / (request_id_var.get() or "no-request")
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump(name: str, obj: Any) -> Path | None:
    """Write obj under DEBUG_DUMP_DIR/<request_id>/. Arrays become PNG, everything else JSON.

    Returns the written path, or None when dumping is disabled.
    """
    target = dump_dir()
    if target is None:
        return None
    if isinstance(obj, np.ndarray):
        out = target / f"{name}.png"
        Image.fromarray(obj).save(out)
    else:
        out = target / f"{name}.json"
        data = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
        out.write_text(json.dumps(data, indent=2, default=str))
    return out
