"""Published / operator-stated heights for known monuments.

Shadow metres on an undated mosaic stay INDICATIVE. When the AOI sits on a
named landmark the 3D extrusion may use the stated height so the volume is
not a squat stub from a failed hunt.
"""
from __future__ import annotations

import math
from typing import Any

# Operator-stated Azadi height (this survey): 43 m. Identity is the plaza
# around 35.70 N, 51.338 E — match within 250 m of the monument.
LANDMARKS: tuple[dict[str, Any], ...] = (
    {
        "id": "azadi_tower",
        "name": "Azadi Tower",
        "name_fa": "برج آزادی",
        "lat": 35.69974,
        "lon": 51.33810,
        "height_m": 43.0,
        # Operator plan extents for massing (nadir inverted-Y), not cadastral.
        "footprint_width_m": 63.0,
        "footprint_depth_m": 52.0,
        "radius_m": 280.0,
        "source": "operator",
        "massing": "azadi_arch",
    },
)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def lookup(lat: float, lon: float) -> dict[str, Any] | None:
    """Nearest landmark within its radius, or None."""
    best: dict[str, Any] | None = None
    best_d = float("inf")
    for row in LANDMARKS:
        dist = _haversine_m(lat, lon, float(row["lat"]), float(row["lon"]))
        if dist <= float(row["radius_m"]) and dist < best_d:
            best = {**row, "dist_m": round(dist, 1)}
            best_d = dist
    return best
