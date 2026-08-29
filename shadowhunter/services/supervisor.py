"""Start the FastAPI backend inside the current process.

Every desktop front-end can therefore be launched as a single executable
without asking the operator to run a server first - but they still talk HTTP,
so the same binary works pointed at a remote deck.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from ..core.config import SETTINGS
from ..core.logging import get_logger
from .client import DeckClient

log = get_logger(__name__)

_server: Any = None
_thread: threading.Thread | None = None


def serve_in_thread(host: str | None = None, port: int | None = None,
                    wait: bool = True) -> DeckClient:
    """Launch uvicorn on a daemon thread and return a client bound to it."""
    global _server, _thread

    import uvicorn

    host = host or SETTINGS.server.host
    port = int(port or SETTINGS.server.port)
    client = DeckClient(base_url=f"http://{host}:{port}")

    # Someone already has the deck up - reuse it rather than fighting for the port.
    try:
        client.health()
        log.info("attached to existing deck at %s", client.base_url)
        return client
    except Exception:
        pass

    from ..views.api.app import app

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    _server = uvicorn.Server(config)
    # Load the trained policy while we are still on the main thread - SB3
    # cannot be deserialised from the server thread without corrupting the heap.
    from ..models.pipeline import REGISTRY
    REGISTRY.warmup()

    _thread = threading.Thread(target=_server.run, daemon=True, name="deck-api")
    _thread.start()

    if wait and not client.wait_until_ready():
        raise RuntimeError(f"backend failed to come up on {client.base_url}")
    log.info("deck started at %s", client.base_url)
    return client


def shutdown(timeout: float = 5.0) -> None:
    global _server, _thread
    if _server is not None:
        _server.should_exit = True
        deadline = time.time() + timeout
        while _thread and _thread.is_alive() and time.time() < deadline:
            time.sleep(0.05)
    _server, _thread = None, None
