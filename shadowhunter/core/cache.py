"""Two-tier cache: process-local LRU plus a namespaced disk store.

Network-facing modules (intel, the copilot brief) go through here so a
timeout never has to be paid twice in the same day. Memory is the hot path;
disk is how a restart still remembers Nominatim.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .config import SETTINGS
from .logging import get_logger

log = get_logger(__name__)

_CAPACITY = 256
_NS_OK = re.compile(r"^[A-Za-z0-9_.-]+$")


def hash_key(*parts: Any) -> str:
    """Stable 24-char key from arbitrary parts (sha256 hex prefix)."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            h.update(part)
        else:
            h.update(repr(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def _safe_namespace(namespace: str) -> str:
    text = (namespace or "default").strip().replace("\\", "/").split("/")[-1]
    if not text or not _NS_OK.match(text):
        text = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text


class Cache:
    """Memory LRU (256) in front of ``SETTINGS.data_dir / cache / namespace``."""

    _instance: Cache | None = None
    _instance_lock = threading.Lock()

    def __init__(self, capacity: int = _CAPACITY, root: Path | None = None) -> None:
        self.capacity = capacity
        self.root = Path(root) if root is not None else Path(SETTINGS.data_dir) / "cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem: OrderedDict[tuple[str, str], tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()

    @classmethod
    def instance(cls) -> Cache:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------ #
    def _paths(self, namespace: str, key: str) -> tuple[Path, Path]:
        ns = _safe_namespace(namespace)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        folder = self.root / ns
        return folder / f"{digest}.json", folder / f"{digest}.bin"

    def _remember(self, namespace: str, key: str, value: Any, exp: float) -> None:
        slot = (namespace, key)
        self._mem[slot] = (exp, value)
        self._mem.move_to_end(slot)
        while len(self._mem) > self.capacity:
            self._mem.popitem(last=False)

    def _purge_disk(self, json_path: Path, bin_path: Path) -> None:
        for path in (json_path, bin_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _read_disk(self, namespace: str, key: str) -> Any | None:
        json_path, bin_path = self._paths(namespace, key)
        now = time.time()
        envelope: dict[str, Any] | None = None
        if json_path.exists():
            try:
                envelope = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.debug("cache json unreadable %s: %s", json_path, exc)
                envelope = None
        if envelope is not None:
            exp = float(envelope.get("exp") or 0.0)
            if exp and exp < now:
                self._purge_disk(json_path, bin_path)
                return None
            if envelope.get("bin"):
                if not bin_path.exists():
                    return None
                try:
                    blob = bin_path.read_bytes()
                except OSError as exc:
                    log.debug("cache bin unreadable %s: %s", bin_path, exc)
                    return None
                self._remember(namespace, key, blob, exp if exp else now + 86400)
                return blob
            value = envelope.get("v")
            self._remember(namespace, key, value, exp if exp else now + 86400)
            return value
        if bin_path.exists():
            # Bytes with no envelope: treat as expired (unknown TTL).
            self._purge_disk(json_path, bin_path)
            return None
        return None

    def get(self, namespace: str, key: str) -> Any | None:
        slot = (namespace, key)
        now = time.time()
        with self._lock:
            hit = self._mem.get(slot)
            if hit is not None:
                exp, value = hit
                if exp >= now:
                    self._mem.move_to_end(slot)
                    return value
                del self._mem[slot]
            return self._read_disk(namespace, key)

    def set(self, namespace: str, key: str, value: Any, *,
            ttl_s: float = 86400, raw: bytes | None = None) -> None:
        exp = time.time() + max(float(ttl_s), 0.0)
        json_path, bin_path = self._paths(namespace, key)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        blob = raw if raw is not None else (value if isinstance(value, (bytes, bytearray)) else None)
        with self._lock:
            self._remember(namespace, key, value, exp)
            try:
                if blob is not None:
                    bin_path.write_bytes(bytes(blob))
                    envelope = {"exp": exp, "bin": True}
                    json_path.write_text(json.dumps(envelope), encoding="utf-8")
                else:
                    try:
                        payload = json.dumps({"exp": exp, "v": value}, ensure_ascii=False, default=str)
                    except (TypeError, ValueError) as exc:
                        log.debug("cache skip disk for %s/%s: %s", namespace, key, exc)
                        return
                    json_path.write_text(payload, encoding="utf-8")
                    try:
                        bin_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            except OSError as exc:
                log.debug("cache disk write failed %s/%s: %s", namespace, key, exc)

    def get_json(self, namespace: str, key: str) -> Any | None:
        return self.get(namespace, key)

    def set_json(self, namespace: str, key: str, value: Any, *, ttl_s: float = 86400) -> None:
        self.set(namespace, key, value, ttl_s=ttl_s)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            entries_mem = len(self._mem)
        bytes_disk = 0
        namespaces: list[str] = []
        if self.root.exists():
            for child in sorted(self.root.iterdir()):
                if child.is_dir():
                    namespaces.append(child.name)
                    for path in child.rglob("*"):
                        if path.is_file():
                            try:
                                bytes_disk += path.stat().st_size
                            except OSError:
                                pass
        return {"entries_mem": entries_mem, "bytes_disk": bytes_disk, "namespaces": namespaces}


def instance() -> Cache:
    """Module-level alias for ``Cache.instance()``."""
    return Cache.instance()
