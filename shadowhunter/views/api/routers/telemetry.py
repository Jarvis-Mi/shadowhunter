"""Live telemetry: a websocket fan-out of the in-process event bus."""
from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ....core.logging import BUS, get_logger

log = get_logger(__name__)
router = APIRouter(tags=["telemetry"])


@router.get("/api/telemetry", summary="Recent telemetry events")
async def recent(topic: str | None = None, limit: int = 100):
    return {"items": BUS.history(topic, limit)}


@router.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket) -> None:
    """Every training step, every finished analysis, pushed as it happens."""
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=512)

    def on_event(event: dict) -> None:
        # Called from worker threads - hop back onto the event loop safely.
        loop.call_soon_threadsafe(lambda: queue.put_nowait(event) if not queue.full() else None)

    unsubscribe = BUS.subscribe(on_event)
    try:
        for past in BUS.history(limit=25):
            await ws.send_json(past)
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20.0)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                await ws.send_json({"topic": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("telemetry socket closed: %s", exc)
    finally:
        unsubscribe()
        with contextlib.suppress(Exception):
            await ws.close()
