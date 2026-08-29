"""JSON that FastAPI can actually emit.

Python's ``json.dumps`` (and Starlette's JSONResponse) reject ``inf`` / ``nan``.
Solar geometry at the horizon produces those; we map them to ``null`` so a
timeline request never takes the ASGI stack down.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from fastapi.responses import JSONResponse


def finite(obj: Any) -> Any:
    """Walk a payload and replace non-finite numbers with ``None``."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {str(k): finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return finite(obj.item())
    return obj


class FiniteJSONResponse(JSONResponse):
    """Default response class: never raise on inf/nan."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            finite(content),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
