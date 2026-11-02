"""Structured logging: one JSON object per line on stderr (journald and docker logs keep it as is).

Every record carries ``ts``, ``level``, ``component`` and ``event``; callers add fields as keyword arguments.
Nothing secret is ever passed in: tokens stay inside auth.py, request bodies are never logged.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_ROOT = "kvhandoff"
_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "component": record.name.removeprefix(_ROOT + ".") if record.name != _ROOT else "core",
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            doc.update(fields)  # pyright: ignore[reportUnknownArgumentType]
        if record.exc_info:
            doc["exception"] = self.formatException(record.exc_info)
        return json.dumps(doc, default=str, separators=(",", ":"))


def configure(level: str = "info", stream: Any = None) -> None:
    """Attach the JSON handler to the ``kvhandoff`` logger once. Safe to call repeatedly."""
    global _configured
    root = logging.getLogger(_ROOT)
    root.setLevel(level.upper())
    if _configured:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False
    _configured = True


class EventLogger:
    """``log.info("event_name", tag=..., ms=...)``: the message is the event name, keywords become JSON fields."""

    def __init__(self, component: str):
        self._log = logging.getLogger(f"{_ROOT}.{component}")

    def _emit(self, level: int, event: str, exc_info: bool, fields: dict[str, Any]) -> None:
        if self._log.isEnabledFor(level):
            self._log.log(level, event, extra={"fields": fields}, exc_info=exc_info)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, False, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, False, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, False, fields)

    def error(self, event: str, exc_info: bool = False, **fields: Any) -> None:
        self._emit(logging.ERROR, event, exc_info, fields)


def get_logger(component: str) -> EventLogger:
    return EventLogger(component)


def ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)
