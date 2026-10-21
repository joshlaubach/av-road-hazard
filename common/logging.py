"""Structured JSON logging with a request_id carried in a context variable."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# Holds the current request's id. A ContextVar is per-request even with concurrent async requests,
# which a plain global would not be.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Extra fields callers may pass via logger.info(..., extra={...}).
_EXTRA_FIELDS = ("frame_id", "stage", "latency_ms", "count", "route", "status")


class JsonFormatter(logging.Formatter):
    """Format each log record as one JSON object per line, easy to filter by request_id."""

    def format(self, record: logging.LogRecord) -> str:
        """Build the JSON line: timestamp, level, logger name, message, request id, plus any of the
        allowed extra fields the caller attached and the traceback if there was an exception."""
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        # Only the allowed names are copied from `extra={...}`. An allowlist keeps accidental
        # large or sensitive values (image bytes, keys) out of the logs.
        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # default=str: anything json cannot serialize is written as text instead of raising.
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Replace root handlers with one JSON handler on stdout. Never log image bytes or secrets."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
