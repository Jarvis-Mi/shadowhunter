"""``DeckClient`` - the one transport every front-end uses.

CustomTkinter, PySide6, Flet, NiceGUI and DearPyGui all speak to the backend
through this class. That is what keeps five UIs from drifting into five
slightly different applications.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np

from ..core.config import SETTINGS
from ..core.logging import get_logger

log = get_logger(__name__)


class DeckError(RuntimeError):
    """Any failure talking to the backend, already human-readable."""


@dataclass
class DeckClient:
    """Minimal synchronous HTTP client - stdlib only, no hard dependency."""

    base_url: str = SETTINGS.api_base
    timeout: float = 600.0

    # ------------------------------------------------------------------ #
    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None,
              timeout: float | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urlrequest.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"})
        try:
            with urlrequest.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read()
                return json.loads(body) if body else {}
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except json.JSONDecodeError:
                pass
            raise DeckError(f"HTTP {exc.code}: {detail}") from exc
        except urlerror.URLError as exc:
            raise DeckError(f"backend unreachable at {self.base_url} ({exc.reason})") from exc

    # ------------------------------------------------------------ endpoints
    def health(self) -> dict[str, Any]:
        return self._call("GET", "/api/health", timeout=8.0)

    def wait_until_ready(self, attempts: int = 150, delay: float = 0.4) -> bool:
        for _ in range(attempts):
            try:
                self.health()
                return True
            except DeckError:
                time.sleep(delay)
        return False

    def analyze(self, *, size: int = 512, buildings: int = 14, density: float = 1.0,
                seed: int | None = None, sun: dict[str, float] | None = None,
                policy: str = "auto", max_steps: int = 48,
                start: tuple[int, int] | None = None,
                overlay: bool = True) -> dict[str, Any]:
        return self._call("POST", "/api/analyze", {
            "scene": {"synthesize": True, "size": size, "buildings": buildings,
                      "density": density, "seed": seed, "sun": sun},
            "start": list(start) if start else None,
            "policy": policy, "max_steps": max_steps,
            "return_overlay": overlay, "return_trajectory": True,
        })

    def sweep(self, *, size: int = 512, buildings: int = 14, seed: int | None = None,
              sun: dict[str, float] | None = None, policy: str = "auto",
              limit: int = 12) -> dict[str, Any]:
        return self._call("POST", "/api/sweep", {
            "scene": {"synthesize": True, "size": size, "buildings": buildings,
                      "seed": seed, "sun": sun},
            "policy": policy, "limit": limit,
        })

    def train_rl(self, **kwargs: Any) -> dict[str, Any]:
        return self._call("POST", "/api/train/rl", kwargs, timeout=30.0)

    def train_cnn(self, **kwargs: Any) -> dict[str, Any]:
        return self._call("POST", "/api/train/cnn", kwargs, timeout=30.0)

    def jobs(self, limit: int = 40) -> list[dict[str, Any]]:
        return self._call("GET", f"/api/train/jobs?limit={limit}", timeout=15.0)["items"]

    def job(self, job_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/train/jobs/{job_id}", timeout=15.0)

    def abort(self, job_id: str) -> dict[str, Any]:
        return self._call("POST", f"/api/train/jobs/{job_id}/abort", {}, timeout=15.0)

    def artifacts(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/train/artifacts", timeout=15.0)["items"]

    def load_artifact(self, name: str) -> dict[str, Any]:
        return self._call("POST", f"/api/train/artifacts/{name}/load", {}, timeout=60.0)

    def telemetry(self, topic: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        q = f"?limit={limit}" + (f"&topic={topic}" if topic else "")
        return self._call("GET", f"/api/telemetry{q}", timeout=15.0)["items"]

    def history(self, limit: int = 25) -> list[dict[str, Any]]:
        return self._call("GET", f"/api/history?limit={limit}", timeout=15.0)["items"]

    def scenes(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/scenes", timeout=15.0)["items"]

    # ------------------------------------------------------- map / real AOI
    def providers(self) -> dict[str, Any]:
        return self._call("GET", "/api/map/providers", timeout=15.0)

    def map_plan(self, bbox: tuple[float, float, float, float],
                 max_tiles: int = 36) -> dict[str, Any]:
        west, south, east, north = bbox
        return self._call("GET", f"/api/map/plan?west={west}&south={south}"
                                 f"&east={east}&north={north}&max_tiles={max_tiles}",
                          timeout=15.0)

    def sun(self, lat: float, lon: float, when: str | None = None,
            best_hour: bool = True) -> dict[str, Any]:
        return self._call("POST", "/api/sun",
                          {"lat": lat, "lon": lon, "when": when, "best_hour": best_hour},
                          timeout=20.0)

    def suggest(self, lat: float, lon: float, *, span_m: float = 200.0,
                when: str | None = None) -> dict[str, Any]:
        q = f"/api/aoi/suggest?lat={lat}&lon={lon}&span_m={span_m}"
        if when:
            from urllib.parse import quote
            q += f"&when={quote(when)}"
        return self._call("POST", q, {}, timeout=180.0)

    def survey(self, bbox: tuple[float, float, float, float], *, provider: str = "esri",
               zoom: int | None = None, max_tiles: int = 36, when: str | None = None,
               auto_sun: bool = False, min_size_m: float = 10.0,
               max_structures: int = 80) -> dict[str, Any]:
        west, south, east, north = bbox
        return self._call("POST", "/api/aoi/survey", {
            "bbox": {"west": west, "south": south, "east": east, "north": north},
            "provider": provider, "zoom": zoom, "max_tiles": max_tiles, "when": when,
            "auto_sun": auto_sun, "min_size_m": min_size_m,
            "max_structures": max_structures, "detect": True,
        })

    def aoi_analyze(self, aoi_id: str, *, indices: list[int] | None = None,
                    policy: str = "auto", max_steps: int = 40,
                    limit: int = 24) -> dict[str, Any]:
        return self._call("POST", "/api/aoi/analyze", {
            "aoi_id": aoi_id, "indices": indices, "policy": policy,
            "max_steps": max_steps, "limit": limit,
        })

    def geocode(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        from urllib.parse import quote

        return self._call("GET", f"/api/geocode?q={quote(query)}&limit={limit}",
                          timeout=25.0)["items"]

    def brain_health(self) -> dict[str, Any]:
        return self._call("GET", "/api/brain/health", timeout=8.0)

    def brief(self, aoi_id: str | None = None, locale: str = "fa") -> dict[str, Any]:
        payload: dict[str, Any] = {"locale": locale}
        if aoi_id:
            payload["aoi_id"] = aoi_id
        return self._call("POST", "/api/brain/brief", payload, timeout=180.0)

    def construct(self, aoi_id: str | None = None, locale: str = "fa") -> dict[str, Any]:
        payload: dict[str, Any] = {"locale": locale}
        if aoi_id:
            payload["aoi_id"] = aoi_id
        return self._call("POST", "/api/brain/construct", payload, timeout=180.0)

    def intel(self, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
        west, south, east, north = bbox
        return self._call("GET", f"/api/intel?west={west}&south={south}"
                                 f"&east={east}&north={north}", timeout=90.0)

    def inspect_aoi(self, aoi_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/inspect/{aoi_id}", timeout=30.0)

    def timeline(self, lat: float, lon: float, when: str | None = None,
                 step_minutes: int = 60) -> dict[str, Any]:
        q = f"/api/timeline?lat={lat}&lon={lon}&step_minutes={step_minutes}"
        if when:
            from urllib.parse import quote
            q += f"&when={quote(when)}"
        return self._call("GET", q, timeout=20.0)

    def scene3d(self, aoi_id: str, measures: list[dict[str, Any]] | None = None,
                open: bool = True) -> dict[str, Any]:
        """Export a 3D scene. ``measures`` may be positional (Qt worker) or omitted."""
        return self._call("POST", "/api/scene3d",
                          {"aoi_id": aoi_id, "open": open, "measures": measures or []},
                          timeout=60.0)

    def plan_run(self, lat: float, lon: float, span_m: float = 400.0) -> dict[str, Any]:
        return self._call("POST", "/api/brain/plan",
                          {"lat": lat, "lon": lon, "span_m": span_m}, timeout=30.0)

    def tools(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/brain/tools", timeout=8.0).get("items") or []

    def favorites(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/places/favorites", timeout=8.0).get("items") or []

    def add_favorite(self, name: str, lat: float, lon: float,
                     bbox: list[float] | None = None, note: str = "") -> dict[str, Any]:
        return self._call("POST", "/api/places/favorites",
                          {"name": name, "lat": lat, "lon": lon, "bbox": bbox, "note": note})

    def place_history(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/places/history?limit=40", timeout=8.0).get("items") or []

    def run_field(self, bbox: tuple[float, float, float, float], *,
                  locale: str = "fa", query: str | None = None,
                  when: str | None = None) -> dict[str, Any]:
        west, south, east, north = bbox
        payload: dict[str, Any] = {
            "bbox": {"west": west, "south": south, "east": east, "north": north},
            "locale": locale,
        }
        if query:
            payload["query"] = query
        if when:
            payload["when"] = when
        return self._call("POST", "/api/field/run", payload, timeout=600.0)


# --------------------------------------------------------------------------- #
# Image helpers shared by every view
# --------------------------------------------------------------------------- #
def decode_png(b64: str | None) -> np.ndarray | None:
    """base64 PNG -> BGR ndarray."""
    if not b64:
        return None
    import cv2

    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def png_bytes(b64: str | None) -> bytes | None:
    return base64.b64decode(b64) if b64 else None


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
class TelemetryStream:
    """Background follower of ``/api/telemetry``.

    Polls rather than holding a websocket, because Tk, Qt, Flet and DearPyGui
    each have a different event loop and a 300 ms poll is indistinguishable
    from a push at human timescales. The websocket at ``/ws/telemetry``
    remains available for the browser client.
    """

    def __init__(self, client: DeckClient, on_event: Callable[[dict[str, Any]], None],
                 interval: float = 0.35) -> None:
        self.client = client
        self.on_event = on_event
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: set[str] = set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="telemetry")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for event in self.client.telemetry(limit=60):
                    key = f"{event.get('ts')}|{event.get('topic')}|{event.get('timesteps', '')}"
                    if key in self._seen:
                        continue
                    self._seen.add(key)
                    self.on_event(event)
                if len(self._seen) > 4000:
                    self._seen.clear()
            except DeckError:
                pass  # backend restarting; keep trying quietly
            except Exception:
                log.debug("telemetry poll failed", exc_info=True)
            self._stop.wait(self.interval)
