"""Shadow-hour timeline: where the sun is, and how long a metre of height casts.

The strip renderer is OpenCV-only so every view (API, Qt, Flet) can show the
same day-curve without importing a UI toolkit.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import cv2
import numpy as np

from ..core.geo import SunGeometry, quality_of_geometry
from ..core.sun import SolarPosition, best_hour_of_day, daylight_window, solar_position

# Local civil hours sampled every day so shadow geometry can be tabulated
# without re-photographing an undated mosaic (ephemeris only).
CAPTURE_HOURS_LOCAL: tuple[int, ...] = (10, 12, 13, 14, 15, 16)

_SEASON_MARKERS: tuple[tuple[str, int, int], ...] = (
    ("spring", 3, 21),
    ("summer", 6, 21),
    ("autumn", 9, 22),
    ("winter", 12, 21),
)


def _aware(day: datetime) -> datetime:
    if day.tzinfo is None:
        return day.replace(tzinfo=timezone.utc)
    return day.astimezone(timezone.utc)


def _utc_day(day: datetime) -> datetime:
    return _aware(day).replace(hour=0, minute=0, second=0, microsecond=0)


def _as_sun(sun: SunGeometry | SolarPosition | dict[str, Any],
            gsd_m: float = 0.5) -> SunGeometry:
    if isinstance(sun, SunGeometry):
        return sun
    if isinstance(sun, SolarPosition):
        return sun.to_geometry(gsd_m)
    return SunGeometry(
        azimuth_deg=float(sun.get("azimuth_deg", 148.0)),
        elevation_deg=float(sun.get("elevation_deg", 41.0)),
        gsd_m=float(sun.get("gsd_m", gsd_m)),
    )


def _shadow_length_factor(elevation_deg: float) -> float | None:
    """Metres of shadow per metre of height. None when the sun is at/below the horizon."""
    if elevation_deg <= 0.25:
        return None
    tan_e = math.tan(math.radians(max(elevation_deg, 0.5)))
    if tan_e <= 1e-8:
        return None
    return round(min(1.0 / tan_e, 1.0e6), 4)


def civil_timezone(lon_deg: float) -> timezone:
    """Fixed civil offset for this longitude.

    Iran (lon 44..63.5) is UTC+03:30 year-round. Everywhere else uses the
    nearest hour from longitude/15.
    """
    lon = float(lon_deg)
    if 44.0 <= lon <= 63.5:
        return timezone(timedelta(hours=3, minutes=30))
    return timezone(timedelta(hours=int(round(lon / 15.0))))


def capture_slots(lat: float, lon: float, day: datetime,
                  gsd_m: float = 0.5) -> list[dict[str, Any]]:
    """One ephemeris sample per local capture hour on this civil day."""
    tz = civil_timezone(lon)
    local_date = _aware(day).astimezone(tz).date()
    slots: list[dict[str, Any]] = []
    for hour in CAPTURE_HOURS_LOCAL:
        local_moment = datetime(
            local_date.year, local_date.month, local_date.day,
            int(hour), 0, 0, tzinfo=tz,
        )
        moment = local_moment.astimezone(timezone.utc)
        pos = solar_position(lat, lon, moment)
        geom = pos.to_geometry(gsd_m)
        slots.append({
            "local_hour": int(hour),
            "when": moment.isoformat(),
            "elevation_deg": round(pos.elevation_deg, 3),
            "azimuth_deg": round(pos.azimuth_deg, 3),
            "quality": round(quality_of_geometry(geom), 4),
            "is_daylight": bool(pos.is_daylight),
            "shadow_bearing_deg": round(geom.shadow_bearing_deg, 3),
            "shadow_length_factor": _shadow_length_factor(pos.elevation_deg),
        })
    return slots


def capture_calendar(lat: float, lon: float, year: int,
                     gsd_m: float = 0.5) -> dict[str, Any]:
    """Four seasonal marker days, each with the local capture-hour clock."""
    tz = civil_timezone(lon)
    seasons: list[dict[str, Any]] = []
    for season, month, month_day in _SEASON_MARKERS:
        stamp = datetime(int(year), month, month_day, tzinfo=tz)
        seasons.append({
            "season": season,
            "day": stamp.date().isoformat(),
            "slots": capture_slots(lat, lon, stamp, gsd_m=gsd_m),
        })
    return {"seasons": seasons, "tz": str(tz)}


def hour_samples(lat: float, lon: float, day: datetime,
                 step_minutes: int = 60, gsd_m: float = 0.5) -> list[dict[str, Any]]:
    """One solar sample per step across the UTC day."""
    origin = _utc_day(day)
    step = max(1, int(step_minutes))
    samples: list[dict[str, Any]] = []
    for minute in range(0, 24 * 60, step):
        moment = origin + timedelta(minutes=minute)
        pos = solar_position(lat, lon, moment)
        geom = pos.to_geometry(gsd_m)
        samples.append({
            "when": moment.isoformat(),
            "azimuth_deg": round(pos.azimuth_deg, 3),
            "elevation_deg": round(pos.elevation_deg, 3),
            "is_daylight": bool(pos.is_daylight),
            "quality": round(quality_of_geometry(geom), 4),
            "shadow_bearing_deg": round(geom.shadow_bearing_deg, 3),
            "shadow_length_factor": _shadow_length_factor(pos.elevation_deg),
        })
    return samples


def build_timeline(lat: float, lon: float, when: datetime,
                   gsd_m: float = 0.5, step_minutes: int = 60) -> dict[str, Any]:
    """Day curve plus sunrise, sunset, and the best metrology hour."""
    when = _aware(when)
    samples = hour_samples(lat, lon, when, step_minutes=step_minutes, gsd_m=gsd_m)
    rise, set_ = daylight_window(lat, lon, when)
    best_when, best_pos = best_hour_of_day(lat, lon, when)
    best_geom = best_pos.to_geometry(gsd_m)
    tz = civil_timezone(lon)
    return {
        "lat": float(lat),
        "lon": float(lon),
        "day": _utc_day(when).date().isoformat(),
        "gsd_m": float(gsd_m),
        "sunrise": rise.isoformat() if rise else None,
        "sunset": set_.isoformat() if set_ else None,
        "best_when": best_when.isoformat(),
        "best_elevation": round(best_pos.elevation_deg, 3),
        "best_elevation_deg": round(best_pos.elevation_deg, 3),
        "best_azimuth": round(best_pos.azimuth_deg, 3),
        "best_quality": round(quality_of_geometry(best_geom), 4),
        "samples": samples,
        "captures": capture_slots(lat, lon, when, gsd_m=gsd_m),
        "calendar": capture_calendar(lat, lon, int(when.year), gsd_m=gsd_m),
        "tz": str(tz),
    }


def render_timeline_strip(samples: list[dict[str, Any]],
                          width: int = 720, height: int = 88) -> np.ndarray:
    """BGR uint8 strip: amber elevation curve, darker night, cyan best hour."""
    width = max(64, int(width))
    height = max(32, int(height))
    void = np.array([11, 8, 6], dtype=np.uint8)          # #06080B
    night_c = np.array([8, 6, 4], dtype=np.uint8)
    solar = (32, 176, 255)                               # #FFB020 BGR
    cyan = (228, 211, 63)                                # #3FD3E4 BGR
    hairline = (54, 44, 31)

    img = np.full((height, width, 3), void, dtype=np.uint8)
    if not samples:
        return img

    n = len(samples)
    xs = np.linspace(0, width - 1, n).astype(np.int32)
    daylight = np.array([1.0 if s.get("is_daylight") else 0.0 for s in samples], np.float32)
    # Night bands as a 1-D mask expanded across height — no per-pixel Python loops.
    col = np.where(daylight[:, None] < 0.5, night_c[None, :], void[None, :]).astype(np.uint8)
    # Map each sample onto its x-span by painting columns.
    edges = np.linspace(0, width, n + 1).astype(np.int32)
    for i in range(n):
        img[:, edges[i]:max(edges[i + 1], edges[i] + 1)] = col[i]

    pad = max(8, height // 10)
    usable = max(1, height - 2 * pad)
    elevations = np.array([float(s.get("elevation_deg", 0.0)) for s in samples], dtype=np.float64)
    ys = (height - pad - np.clip(elevations, 0.0, 90.0) / 90.0 * usable).astype(np.int32)
    ys = np.clip(ys, 0, height - 1)
    pts = np.stack([xs, ys], axis=1)

    baseline_y = int(height - pad)
    cv2.line(img, (0, baseline_y), (width - 1, baseline_y), hairline, 1, cv2.LINE_AA)

    if n >= 2:
        fill = np.vstack([
            pts,
            [[int(pts[-1, 0]), baseline_y]],
            [[int(pts[0, 0]), baseline_y]],
        ]).astype(np.int32)
        wash = img.copy()
        cv2.fillPoly(wash, [fill], (18, 72, 130), cv2.LINE_AA)
        img = cv2.addWeighted(wash, 0.38, img, 0.62, 0)
        cv2.polylines(img, [pts], False, solar, 2, cv2.LINE_AA)

    qualities = np.array([float(s.get("quality", 0.0)) for s in samples], dtype=np.float64)
    best_i = int(np.argmax(qualities))
    bx, by = int(xs[best_i]), int(ys[best_i])
    cv2.line(img, (bx, pad // 2), (bx, height - pad // 2), cyan, 1, cv2.LINE_AA)
    cv2.circle(img, (bx, by), 5, cyan, -1, cv2.LINE_AA)
    cv2.circle(img, (bx, by), 8, cyan, 1, cv2.LINE_AA)
    return img


def project_shadow_field(
    bgr: np.ndarray,
    sun: SunGeometry | SolarPosition | dict[str, Any],
    heights_m: list[float],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw predicted shadow volumes for footprints under a given sun."""
    out = np.ascontiguousarray(bgr).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    geom = _as_sun(sun)
    dx, dy = geom.shadow_vector
    elev = max(geom.elevation_deg, 0.5)
    length_per_m = (1.0 / math.tan(math.radians(elev))) / max(geom.gsd_m, 1e-6)
    tint = (228, 211, 63)
    overlay = out.copy()

    for height_m, box in zip(heights_m, boxes):
        x, y, w, h = (int(v) for v in box)
        length_px = float(height_m) * length_per_m
        if length_px < 1.0:
            continue
        roof = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        shadow = roof + np.array([dx, dy], dtype=np.float32) * length_px
        hull = cv2.convexHull(np.vstack([roof, shadow])).astype(np.int32)
        cv2.fillConvexPoly(overlay, hull, tint, cv2.LINE_AA)
        cv2.polylines(out, [shadow.astype(np.int32)], True, tint, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.28, out, 0.72, 0)
