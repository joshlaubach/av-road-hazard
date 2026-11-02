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
    """Read an environment variable as a boolean: 1/true/yes/on (any case) is True, anything else
    False. Returns None when the variable is unset or empty, meaning "leave the YAML value"."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in _TRUE


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read the YAML config, then apply environment overrides. Not cached; see get_config."""
    # safe_load only builds plain data (dicts, lists, strings, numbers); it never runs code.
    with (path or DEFAULT_CONFIG).open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Environment variables win over the YAML file, so a deploy or a .env can change a setting
    # without editing the repo. USE_STUBS is a boolean, so it gets its own parsing.
    stubs = _env_bool("USE_STUBS")
    if stubs is not None:
        cfg["pipeline"]["use_stubs"] = stubs
    # env var name -> (config section, key). Only non-empty values override the YAML default.
    for env, (section, key) in {
        "DATA_DIR": ("paths", "data_dir"),
        "CHECKPOINT_DIR": ("paths", "checkpoint_dir"),
        "DEBUG_DUMP_DIR": ("paths", "debug_dump_dir"),
        "DETECTOR_MODEL": ("perception", "model"),
        "DETECTOR_CHECKPOINT": ("perception", "finetuned_checkpoint"),
        "DEFECT_CHECKPOINT": ("defects", "classifier_checkpoint"),
        "LLM_BACKEND": ("llm", "backend"),
        "OLLAMA_HOST": ("llm", "ollama_host"),
        "OLLAMA_MODEL": ("llm", "ollama_model"),
        "CLAUDE_MODEL": ("llm", "claude_model"),
        "CLAUDE_ESCALATION_MODEL": ("llm", "claude_escalation_model"),
        "QDRANT_URL": ("qdrant", "url"),
        "QDRANT_API_KEY": ("qdrant", "api_key"),
        "ANTHROPIC_DAILY_SPEND_CAP_USD": ("rag", "daily_usd"),
        "ANTHROPIC_MONTHLY_SPEND_CAP_USD": ("rag", "monthly_usd"),
        "LOG_LEVEL": ("logging", "level"),
    }.items():
        value = os.environ.get(env)
        if value:
            cfg[section][key] = value
    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """The shared config, loaded once per process (lru_cache). Use this in application code.

    Tests that change environment variables need `get_config.cache_clear()` or load_config().
    """
    return load_config()
