"""Job bookkeeping - background training runs, persisted to SQLite.

Kept deliberately small: a training run is a row, its live progress is an
event on the bus. Restarting the API loses no history.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.config import SETTINGS

_JOB_COLUMNS = frozenset({
    "id", "kind", "state", "progress", "message", "params", "metrics",
    "artifact", "started_at", "finished_at",
})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    state       TEXT NOT NULL,
    progress    REAL NOT NULL DEFAULT 0,
    message     TEXT NOT NULL DEFAULT '',
    params      TEXT NOT NULL DEFAULT '{}',
    metrics     TEXT NOT NULL DEFAULT '{}',
    artifact    TEXT,
    started_at  TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    scene       TEXT NOT NULL,
    box         TEXT NOT NULL,
    height_m    REAL NOT NULL,
    sigma_m     REAL NOT NULL,
    score       REAL NOT NULL,
    policy      TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or SETTINGS.data_dir / "shadowhunter.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._flags: dict[str, dict[str, bool]] = {}
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    # ---------------------------------------------------------------- jobs
    def create(self, kind: str, params: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO jobs (id, kind, state, params, started_at) VALUES (?,?,?,?,?)",
                (job_id, kind, "queued", json.dumps(params), _now()),
            )
        self._flags[job_id] = {"stop": False}
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        unknown = [k for k in fields if k not in _JOB_COLUMNS]
        if unknown:
            raise ValueError(f"unknown job column(s): {unknown}")
        for key in ("metrics", "params"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key])
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._connect() as con:
            con.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def finish(self, job_id: str, state: str, **fields: Any) -> None:
        self.update(job_id, state=state, finished_at=_now(), **fields)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def stop_flag(self, job_id: str) -> dict[str, bool]:
        return self._flags.setdefault(job_id, {"stop": False})

    def request_stop(self, job_id: str) -> bool:
        if job_id not in self._flags:
            return False
        self._flags[job_id]["stop"] = True
        self.update(job_id, message="abort requested")
        return True

    # ------------------------------------------------------------ analyses
    def record_analysis(self, payload: dict[str, Any]) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO analyses (id, created_at, scene, box, height_m, sigma_m, score, policy)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (rid, _now(), payload["scene"].get("name", "scene"), json.dumps(payload["box"]),
                 float(payload["height"]["fused_m"]), float(payload["height"]["sigma_m"]),
                 float(payload["score"]), payload["policy"]),
            )
        return rid

    def recent_analyses(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["box"] = json.loads(d["box"])
            out.append(d)
        return out

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("params", "metrics"):
            try:
                d[key] = json.loads(d.get(key) or "{}")
            except json.JSONDecodeError:
                d[key] = {}
        return d


STORE = JobStore()
