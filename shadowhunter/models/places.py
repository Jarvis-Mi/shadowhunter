"""Saved places and search / survey history.

JSON on disk under ``SETTINGS.data_dir / "places"``. Fail-soft: a corrupt
file or a full disk never raises into the survey path.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.config import SETTINGS
from ..core.logging import get_logger

log = get_logger(__name__)

_LOCK = threading.Lock()
_FAV_NAME = "favorites.json"
_HIST_NAME = "history.jsonl"
_HIST_CAP = 400


def _root() -> Path:
    folder = Path(SETTINGS.data_dir) / "places"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        log.debug("places read failed %s: %s", path, exc)
        return default


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _favorites_unlocked() -> list[dict[str, Any]]:
    raw = _read_json(_root() / _FAV_NAME, [])
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def list_favorites() -> list[dict[str, Any]]:
    with _LOCK:
        return _favorites_unlocked()


def add_favorite(name: str, lat: float, lon: float, bbox: list[float] | tuple[float, ...] | None = None,
                 note: str = "") -> dict[str, Any]:
    item = {
        "id": uuid.uuid4().hex[:8],
        "name": (name or "place").strip()[:120],
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "bbox": [float(v) for v in bbox] if bbox and len(bbox) == 4 else None,
        "note": (note or "")[:240],
        "saved_at": _now(),
    }
    with _LOCK:
        items = _favorites_unlocked()
        items = [f for f in items if not (
            abs(float(f.get("lat") or 0) - item["lat"]) < 1e-5
            and abs(float(f.get("lon") or 0) - item["lon"]) < 1e-5
            and (f.get("name") or "") == item["name"]
        )]
        items.insert(0, item)
        _write_json(_root() / _FAV_NAME, items[:80])
    return item


def remove_favorite(fav_id: str) -> bool:
    ident = (fav_id or "").strip()
    with _LOCK:
        items = _favorites_unlocked()
        kept = [f for f in items if str(f.get("id")) != ident]
        if len(kept) == len(items):
            return False
        _write_json(_root() / _FAV_NAME, kept)
    return True


def list_history(limit: int = 40) -> list[dict[str, Any]]:
    cap = max(1, min(int(limit), 200))
    path = _root() / _HIST_NAME
    lines: list[str] = []
    with _LOCK:
        try:
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.debug("history read failed: %s", exc)
            return []
    out: list[dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
        if len(out) >= cap:
            break
    return out


def record_event(kind: str, **fields: Any) -> None:
    """Append one history row. Never raises."""
    try:
        event: dict[str, Any] = {"kind": str(kind or "event"), "when": _now()}
        for key, value in fields.items():
            if value is None:
                continue
            event[key] = value
        path = _root() / _HIST_NAME
        blob = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with _LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(blob)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                if len(lines) > _HIST_CAP:
                    path.write_text("\n".join(lines[-_HIST_CAP:]) + "\n", encoding="utf-8")
            except OSError:
                pass
    except Exception as exc:
        log.debug("record_event skipped: %s", exc)
