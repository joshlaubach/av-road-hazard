import json
import logging
from pathlib import Path

from common.config import load_config
from common.debug import dump
from common.logging import JsonFormatter, request_id_var


def test_config_defaults_and_seed() -> None:
    """The shipped config turns stubs on and pins the seed to 42."""
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["pipeline"]["use_stubs"] is True


def test_env_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Environment variables override YAML values, including the boolean USE_STUBS."""
    monkeypatch.setenv("USE_STUBS", "0")
    monkeypatch.setenv("LLM_BACKEND", "claude")
    cfg = load_config()
    assert cfg["pipeline"]["use_stubs"] is False
    assert cfg["llm"]["backend"] == "claude"


def test_json_log_carries_request_id() -> None:
    """A formatted log line is JSON and includes the current request id and allowed extra fields."""
    token = request_id_var.set("req-123")
    try:
        rec = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None)
        rec.frame_id = "f1"
        out = json.loads(JsonFormatter().format(rec))
    finally:
        request_id_var.reset(token)
    assert out["request_id"] == "req-123"
    assert out["msg"] == "hello"
    assert out["frame_id"] == "f1"


def test_dump_disabled_by_default() -> None:
    """Without DEBUG_DUMP_DIR set, dump() writes nothing and returns None."""
    assert dump("x", {"a": 1}) is None


def test_dump_writes_json(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """With DEBUG_DUMP_DIR set, dump() writes the object as JSON (cache cleared)."""
    from common import config

    monkeypatch.setenv("DEBUG_DUMP_DIR", str(tmp_path))
    config.get_config.cache_clear()
    try:
        out = dump("thing", {"a": 1})
    finally:
        monkeypatch.delenv("DEBUG_DUMP_DIR")
        config.get_config.cache_clear()
    assert out is not None
    assert json.loads(out.read_text()) == {"a": 1}
