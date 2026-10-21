"""Load configs/default.yaml and apply environment overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

_TRUE = {"1", "true", "yes", "on"}


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in _TRUE


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read the YAML config, then apply environment overrides. Not cached; see get_config."""
    with (path or DEFAULT_CONFIG).open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    stubs = _env_bool("USE_STUBS")
    if stubs is not None:
        cfg["pipeline"]["use_stubs"] = stubs
    for env, (section, key) in {
        "DATA_DIR": ("paths", "data_dir"),
        "CHECKPOINT_DIR": ("paths", "checkpoint_dir"),
        "DEBUG_DUMP_DIR": ("paths", "debug_dump_dir"),
        "LLM_BACKEND": ("llm", "backend"),
        "OLLAMA_HOST": ("llm", "ollama_host"),
        "OLLAMA_MODEL": ("llm", "ollama_model"),
        "QDRANT_URL": ("qdrant", "url"),
        "LOG_LEVEL": ("logging", "level"),
    }.items():
        value = os.environ.get(env)
        if value:
            cfg[section][key] = value
    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    return load_config()
