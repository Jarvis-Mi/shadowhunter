"""Console logging with the project palette, plus an in-process event bus."""
from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

_ANSI = {
    "solar": "\033[38;2;255;176;32m",
    "shadow": "\033[38;2;63;211;228m",
    "signal": "\033[38;2;105;227;140m",
    "alert": "\033[38;2;255;77;94m",
    "faint": "\033[38;2;84;99;111m",
    "reset": "\033[0m",
}

_LEVEL_COLOR = {
    logging.DEBUG: "faint",
    logging.INFO: "shadow",
    logging.WARNING: "solar",
    logging.ERROR: "alert",
    logging.CRITICAL: "alert",
}


class DeckFormatter(logging.Formatter):
    """Telemetry-line formatter: ``HH:MM:SS.mmm  LEVEL  logger  message``."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _ANSI[_LEVEL_COLOR.get(record.levelno, "faint")]
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        name = record.name.replace("shadowhunter.", "")
        return (
            f"{_ANSI['faint']}{ts}{_ANSI['reset']} "
            f"{colour}{record.levelname[:4]:<4}{_ANSI['reset']} "
            f"{_ANSI['faint']}{name:<22}{_ANSI['reset']} "
            f"{record.getMessage()}"
        )


def get_logger(name: str = "shadowhunter") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Windows consoles still default to cp1252; degrees and check marks
        # travel through log lines, so never let an encode error kill a run.
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(DeckFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class EventBus:
    """Tiny thread-safe pub/sub used to fan telemetry out to every view.

    The API layer publishes; websocket clients, the DearPyGui scope and the
    Qt deck all subscribe. Keeps the model layer free of UI imports.
    """

    def __init__(self, history: int = 400) -> None:
        self._subs: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)

    def publish(self, topic: str, **payload: Any) -> dict[str, Any]:
        event = {"topic": topic, "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **payload}
        with self._lock:
            self._history.append(event)
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(event)
            except Exception:  # a broken subscriber must never stop training
                get_logger().debug("subscriber raised", exc_info=True)
        return event

    def subscribe(self, fn: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)
        return lambda: self.unsubscribe(fn)

    def unsubscribe(self, fn: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if fn in self._subs:
                self._subs.remove(fn)

    def history(self, topic: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._history)
        if topic:
            items = [e for e in items if e["topic"] == topic]
        return items[-limit:]


BUS = EventBus()
