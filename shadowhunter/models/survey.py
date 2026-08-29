"""AOI survey: real imagery in, counted structures and measured heights out.

This is the orchestration the operator actually drives:

    draw a rectangle on the map
        -> stitch the satellite tiles for it            (tiles.py)
        -> place the sun from lat/lon and a timestamp   (core/sun.py)
        -> count the structures inside it               (detect.py)
        -> hunt each one's shadow and measure it        (pipeline.py)

The mosaic is kept in a small server-side cache keyed by ``aoi_id`` so the
follow-up analysis never re-downloads the tiles the survey already paid for.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from ..core.geo import SunGeometry, quality_of_geometry
from ..core.logging import get_logger
from ..core.sun import resolve_illumination
from .honesty import imagery_honesty
from .pipeline import analyze, png_b64
from .landmarks import lookup as lookup_landmark
from .vision.detect import (compact_footprint, detect_structures, draw_structures,
                            seed_frame_structure)
from .vision.preprocess import Scene
from .vision.tiles import Mosaic, fetch_mosaic

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# AOI cache
# --------------------------------------------------------------------------- #
class AOICache:
    """Last few surveyed areas, so analysis is one click and no download."""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, entry: dict[str, Any]) -> str:
        aoi_id = uuid.uuid4().hex[:10]
        with self._lock:
            self._items[aoi_id] = entry
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return aoi_id

    def get(self, aoi_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._items.get(aoi_id)
            if entry is not None:
                self._items.move_to_end(aoi_id)
        return entry

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items.keys())


AOI_CACHE = AOICache()


def _curve_json(curve: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in curve or []:
        if not isinstance(row, dict):
            continue
        try:
            rows.append({
                "season": row.get("season"),
                "when": row.get("when"),
                "elevation_deg": round(float(row.get("elevation_deg") or 0), 2),
                "azimuth_deg": round(float(row.get("azimuth_deg") or 0), 2),
                "quality": round(float(row.get("quality") or 0), 3),
                "is_daylight": bool(row.get("is_daylight")),
            })
        except (TypeError, ValueError):
            continue
    return rows


def _annotate_geo(mosaic: Mosaic, structures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pixel box -> lat/lon so the table is filled before MEASURE."""
    located: list[dict[str, Any]] = []
    for i, raw in enumerate(structures):
        item = dict(raw)
        item["index"] = int(item.get("index", i))
        box = item.get("box") or [0, 0, 1, 1]
        center = item.get("center")
        if not (isinstance(center, (list, tuple)) and len(center) >= 2):
            center = [box[0] + box[2] / 2.0, box[1] + box[3] / 2.0]
        cx, cy = float(center[0]), float(center[1])
        lon, lat = mosaic.pixel_to_lonlat(cx, cy)
        item["center"] = [int(round(cx)), int(round(cy))]
        item["lon"] = round(lon, 6)
        item["lat"] = round(lat, 6)
        located.append(item)
    return located


def _covering(structure: dict[str, Any], frame_px: int) -> bool:
    box = structure.get("box") or [0, 0, 1, 1]
    frac = (float(box[2]) * float(box[3])) / max(int(frame_px), 1)
    return frac >= 0.45


def _expand_landmark_footprint(structure: dict[str, Any], landmark: dict[str, Any] | None,
                               gsd: float, frame_w: int, frame_h: int) -> dict[str, Any]:
    """Grow a vault-only crop to the operator plan so massing/label match Azadi."""
    item = dict(structure)
    if not isinstance(landmark, dict):
        return item
    try:
        fw = float(landmark.get("footprint_width_m") or 0)
        fd = float(landmark.get("footprint_depth_m") or 0)
    except (TypeError, ValueError):
        return item
    if fw < 20.0 or fd < 20.0 or gsd <= 0:
        return item
    box = list(item.get("box") or [0, 0, 1, 1])
    if len(box) < 4:
        return item
    x, y, bw, bh = (float(v) for v in box[:4])
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    need_w = fw / gsd
    need_h = fd / gsd
    # Only expand — never shrink a good silhouette.
    if bw >= 0.85 * need_w and bh >= 0.85 * need_h:
        item["width_m"] = max(float(item.get("width_m") or 0), fw)
        item["height_m"] = max(float(item.get("height_m") or 0), fd)
        return item
    nw, nh = max(bw, need_w), max(bh, need_h)
    nx = max(0.0, min(cx - nw / 2.0, frame_w - nw))
    ny = max(0.0, min(cy - nh / 2.0, frame_h - nh))
    nw = min(nw, frame_w - nx)
    nh = min(nh, frame_h - ny)
    item["box"] = [int(round(nx)), int(round(ny)), int(round(max(nw, 4))), int(round(max(nh, 4)))]
    item["center"] = [int(round(nx + nw / 2.0)), int(round(ny + nh / 2.0))]
    item["width_m"] = float(item["box"][2]) * gsd
    item["height_m"] = float(item["box"][3]) * gsd
    item["landmark_footprint"] = True
    return item


def _resolve_footprints(bgr: np.ndarray, sun: SunGeometry,
                        detected: list[dict[str, Any]],
                        lat: float, lon: float, min_size_m: float) -> list[dict[str, Any]]:
    """Prefer a compact central box over a whole-AOI seed or a tiny false hit."""
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    frame = w * h
    landmark = lookup_landmark(lat, lon)
    compact = compact_footprint(bgr, sun, min_size_m=min_size_m,
                                allow_large=bool(landmark))
    items = [dict(s) for s in detected if not _covering(s, frame)]
    gsd = max(float(sun.gsd_m), 1e-6)

    if compact is not None:
        compact = _expand_landmark_footprint(compact, landmark, gsd, w, h)
        compact_span = max(float(compact.get("width_m") or 0), float(compact.get("height_m") or 0))
        if landmark and 8.0 <= compact_span <= 120.0:
            return [compact]
        if not items:
            return [compact]
        best = max(items, key=lambda s: float(s.get("score") or 0))
        best_span = max(float(best.get("width_m") or 0), float(best.get("height_m") or 0))
        if compact_span >= min_size_m and compact_span < 0.70 * max(best_span, 1.0):
            return [compact]
    if items:
        return [_expand_landmark_footprint(s, landmark, gsd, w, h) for s in items]
    seed = seed_frame_structure(bgr, sun)
    if landmark:
        retry = compact_footprint(bgr, sun, min_size_m=max(6.0, min_size_m * 0.6),
                                  allow_large=True)
        if retry is not None:
            return [_expand_landmark_footprint(retry, landmark, gsd, w, h)]
        seed = _expand_landmark_footprint(seed, landmark, gsd, w, h)
        seed["tight_aoi"] = True
        seed["compact"] = True
    return [seed]


def parse_when(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = value.replace("Z", "+00:00")
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Survey
# --------------------------------------------------------------------------- #
def survey(bbox: tuple[float, float, float, float], *, provider: str = "esri",
           zoom: int | None = None, max_tiles: int = 36, when: str | None = None,
           auto_sun: bool = False, min_size_m: float = 10.0,
           max_structures: int = 80, detect: bool = True,
           return_images: bool = True) -> dict[str, Any]:
    """Fetch, illuminate and count an area of interest."""
    t0 = time.perf_counter()

    mosaic: Mosaic = fetch_mosaic(bbox, zoom=zoom, provider=provider, max_tiles=max_tiles)
    lon, lat = mosaic.center
    requested = parse_when(when)
    choice = resolve_illumination(lat, lon, requested, mosaic.gsd_m)
    moment, position, sun, snapped = choice
    season = getattr(choice, "season", None)
    year_curve = _curve_json(getattr(choice, "year_curve", None))
    if snapped:
        auto_sun = True

    result: dict[str, Any] = {
        "structures": [], "count": 0, "with_shadow": 0, "mean_score": 0.0,
        "shadow_coverage": 0.0, "water_coverage": 0.0, "sun_estimate": {},
    }
    if detect:
        result = detect_structures(mosaic.image, sun, min_size_m=min_size_m,
                                   max_results=max_structures, auto_sun=auto_sun)
        if result["sun_estimate"].get("applied"):
            sun = SunGeometry(result["sun"]["azimuth_deg"], sun.elevation_deg, sun.gsd_m)
        result["structures"] = _resolve_footprints(
            mosaic.image, sun, result["structures"], lat, lon, min_size_m)
        result["count"] = len(result["structures"])
        result["with_shadow"] = sum(
            1 for s in result["structures"]
            if float(s.get("shadow_support") or 0) > 0.2
            or float(s.get("shadow_len_px") or 0) > 2)
        scores = [float(s.get("score") or 0) for s in result["structures"]]
        result["mean_score"] = round(sum(scores) / len(scores), 3) if scores else 0.0
    result["structures"] = _annotate_geo(mosaic, result["structures"])
    landmark = lookup_landmark(lat, lon)
    for item in result["structures"]:
        hit = lookup_landmark(float(item["lat"]), float(item["lon"]))
        if not hit:
            continue
        item["stated_height_m"] = float(hit["height_m"])
        item["landmark"] = hit["name"]
        item["height_indicative"] = True
    masks = result.pop("_masks", None)

    scene = Scene(image=mosaic.image, sun=sun, name=f"aoi_{provider}_{mosaic.zoom}",
                  source="basemap")

    aoi_id = AOI_CACHE.put({
        "mosaic": mosaic, "scene": scene, "sun": sun,
        "structures": result["structures"], "when": moment.isoformat(),
        "masks": masks, "season": season, "year_curve": year_curve,
        "landmark": landmark,
    })

    scene_meta = mosaic.meta()
    scene_meta["name"] = scene.name
    scene_meta["center"] = [round(lon, 6), round(lat, 6)]
    scene_meta["center_lat"] = round(lat, 6)
    scene_meta["center_lon"] = round(lon, 6)

    payload: dict[str, Any] = {
        "aoi_id": aoi_id,
        "scene": scene_meta,
        "sun": {
            "azimuth_deg": round(sun.azimuth_deg, 2),
            "elevation_deg": round(sun.elevation_deg, 2),
            "gsd_m": round(sun.gsd_m, 4),
            "shadow_bearing_deg": round(sun.shadow_bearing_deg, 2),
            "quality": round(quality_of_geometry(sun), 3),
            "when": moment.isoformat(timespec="minutes"),
            "when_requested": requested.isoformat(timespec="minutes"),
            "snapped_to_best_hour": snapped,
            "is_daylight": position.is_daylight,
            "season": season,
            "year_curve": year_curve,
        },
        "season": season,
        "year_curve": year_curve,
        "sun_estimate": result["sun_estimate"],
        "structures": result["structures"],
        "count": result["count"],
        "with_shadow": result["with_shadow"],
        "mean_score": result["mean_score"],
        "shadow_coverage": result["shadow_coverage"],
        "water_coverage": result["water_coverage"],
        "honesty": imagery_honesty(
            provider=provider, source="basemap",
            is_daylight=position.is_daylight,
            quality=quality_of_geometry(sun),
            sun_estimate=result.get("sun_estimate"),
        ),
        "landmark": landmark,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    try:
        from .places import record_event
        record_event(
            "survey", lat=lat, lon=lon, bbox=list(bbox), aoi_id=aoi_id,
            name=scene_meta.get("name"), extra={"count": payload["count"], "season": season},
        )
    except Exception:
        pass
    if return_images:
        payload["image_png"] = png_b64(mosaic.image)
        payload["overlay_png"] = png_b64(draw_structures(mosaic.image, result["structures"]))
    return payload


# --------------------------------------------------------------------------- #
# Analysis over a survey
# --------------------------------------------------------------------------- #
def analyze_survey(aoi_id: str, *, indices: list[int] | None = None,
                   policy: str = "auto", max_steps: int = 40,
                   limit: int = 24, return_overlay: bool = True) -> dict[str, Any]:
    """Hunt each detected structure's shadow and turn it into metres."""
    t0 = time.perf_counter()
    entry = AOI_CACHE.get(aoi_id)
    if entry is None:
        raise KeyError(f"unknown or expired aoi: {aoi_id}")

    mosaic: Mosaic = entry["mosaic"]
    scene: Scene = entry["scene"]
    structures: list[dict[str, Any]] = entry["structures"]

    chosen = list(range(len(structures))) if indices is None else \
        [i for i in indices if 0 <= i < len(structures)]
    # Work the most confident detections first - a partial run is then still
    # the most useful partial run.
    chosen.sort(key=lambda i: -structures[i]["score"])
    chosen = chosen[:limit]

    items: list[dict[str, Any]] = []
    boxes: list[tuple[int, int, int, int]] = []
    labels: list[str] = []

    for index in chosen:
        structure = structures[index]
        cx, cy = structure["center"]
        result = analyze(scene=scene, start=(cx, cy), policy=policy,
                         max_steps=max_steps, return_overlay=False,
                         return_trajectory=True)
        height = result["height"]
        lon, lat = mosaic.pixel_to_lonlat(cx, cy)
        items.append({
            "index": index,
            "box": result["box"],
            "detect_box": list(structure.get("box") or []),
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "height_m": height["fused_m"],
            "geometric_m": height.get("geometric_m"),
            "cnn_m": height.get("cnn_m"),
            "fused_m": height.get("fused_m"),
            "stated_height_m": structure.get("stated_height_m"),
            "sigma_m": height["sigma_m"],
            "floors": height["floors"],
            "confidence": height["confidence"],
            "score": result["score"],
            "policy": result.get("policy"),
            "steps": result.get("steps"),
            "trajectory": result.get("trajectory") or [],
            "occlusion": round(result["metrics"]["occlusion"], 3),
            "shadow_len_px": round(result["metrics"]["shadow_len_px"], 1),
            "truncation": round(result["metrics"]["truncation"], 3),
            "box_geom": {
                "px": list(structure.get("box") or []),
                "width_m": round(float(structure.get("width_m") or 0), 2),
                "depth_m": round(float(structure.get("height_m") or 0), 2),
            },
        })
        boxes.append(tuple(structure["box"]))
        labels.append(f"{height['fused_m']:.0f}m")

    heights = [i["height_m"] for i in items]
    tallest = max(items, key=lambda i: i["height_m"]) if items else None

    payload: dict[str, Any] = {
        "aoi_id": aoi_id,
        "items": items,
        "tallest": tallest,
        "mean_height_m": round(float(np.mean(heights)), 2) if heights else 0.0,
        "total_floors": int(sum(i["floors"] for i in items)),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    if return_overlay:
        payload["overlay_png"] = png_b64(_render_measured(mosaic.image, structures, items))
    return payload


def _render_measured(bgr: np.ndarray, structures: list[dict[str, Any]],
                     items: list[dict[str, Any]]) -> np.ndarray:
    """Green boxes for every detection, amber height tags for the measured ones."""
    out = draw_structures(bgr, structures, label=False)
    for item in items:
        x, y, w, h = structures[item["index"]]["box"]
        text = f"{item['height_m']:.0f}m"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
        top = max(th + 5, y)
        cv2.rectangle(out, (x, top - th - 6), (x + tw + 9, top - 1), (32, 176, 255), -1)
        cv2.putText(out, text, (x + 4, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.44,
                    (8, 10, 12), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Geocoding
# --------------------------------------------------------------------------- #
def bbox_from_center(lat: float, lon: float, width_m: float, height_m: float
                     ) -> tuple[float, float, float, float]:
    """Axis-aligned lon/lat box around a point, metres on the ground."""
    import math
    dlat = (height_m / 2.0) / 111_320.0
    cos_lat = max(math.cos(math.radians(lat)), 0.15)
    dlon = (width_m / 2.0) / (111_320.0 * cos_lat)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def suggest_structures(lat: float, lon: float, *, span_m: float = 200.0,
                       provider: str = "esri", max_tiles: int = 36,
                       min_size_m: float = 8.0, when: str | None = None
                       ) -> dict[str, Any]:
    """Propose structure AOIs before the operator draws a rectangle.

    Order of preference:
      1. Named landmark within radius (Azadi → operator plan + pad)
      2. Compact / detect hits inside a scan window around the view centre
    """
    import math
    t0 = time.perf_counter()
    lat, lon = float(lat), float(lon)
    span = float(max(60.0, min(span_m, 600.0)))
    scan = bbox_from_center(lat, lon, span, span)
    items: list[dict[str, Any]] = []

    landmark = lookup_landmark(lat, lon)
    if landmark:
        fw = float(landmark.get("footprint_width_m") or 80.0)
        fd = float(landmark.get("footprint_depth_m") or 80.0)
        pad = 1.40
        lb = bbox_from_center(float(landmark["lat"]), float(landmark["lon"]),
                              fw * pad, fd * pad)
        items.append({
            "id": str(landmark.get("id") or "landmark"),
            "name": landmark.get("name") or "landmark",
            "name_fa": landmark.get("name_fa"),
            "kind": "landmark",
            "bbox": [round(v, 6) for v in lb],
            "lat": float(landmark["lat"]),
            "lon": float(landmark["lon"]),
            "width_m": round(fw * pad, 1),
            "depth_m": round(fd * pad, 1),
            "stated_height_m": float(landmark.get("height_m") or 0) or None,
            "score": 0.96,
            "reason": "named landmark near view centre",
            "reason_fa": "بنای شناخته‌شده نزدیک مرکز نقشه",
        })

    survey_payload: dict[str, Any] | None = None
    try:
        survey_payload = survey(
            scan, provider=provider, max_tiles=max_tiles, when=when,
            auto_sun=True, min_size_m=min_size_m, max_structures=12,
            detect=True, return_images=False,
        )
        entry = AOI_CACHE.get(survey_payload["aoi_id"])
        mosaic = entry["mosaic"] if entry else None
        for s in (survey_payload.get("structures") or [])[:6]:
            box = s.get("box") or [0, 0, 1, 1]
            if mosaic is None or len(box) < 4:
                continue
            x, y, bw, bh = (float(v) for v in box[:4])
            # Pad detect box ~18% so RUN FIELD has shadow margin.
            pad_x, pad_y = bw * 0.18, bh * 0.18
            corners = (
                mosaic.pixel_to_lonlat(x - pad_x, y - pad_y),
                mosaic.pixel_to_lonlat(x + bw + pad_x, y - pad_y),
                mosaic.pixel_to_lonlat(x + bw + pad_x, y + bh + pad_y),
                mosaic.pixel_to_lonlat(x - pad_x, y + bh + pad_y),
            )
            lons = [c[0] for c in corners]
            lats = [c[1] for c in corners]
            sb = (min(lons), min(lats), max(lons), max(lats))
            # Skip if almost the same as an existing landmark suggestion.
            skip = False
            for existing in items:
                if existing.get("kind") != "landmark":
                    continue
                elat, elon = float(existing["lat"]), float(existing["lon"])
                if _haversine_m_local(elat, elon, float(s.get("lat") or lat),
                                      float(s.get("lon") or lon)) < 40.0:
                    skip = True
                    break
            if skip:
                continue
            w_m = float(s.get("width_m") or bw * mosaic.gsd_m)
            d_m = float(s.get("height_m") or bh * mosaic.gsd_m)
            items.append({
                "id": f"detect_{int(s.get('index', 0))}",
                "name": f"structure {int(s.get('index', 0)) + 1}",
                "name_fa": f"سازه {int(s.get('index', 0)) + 1}",
                "kind": "detect",
                "bbox": [round(v, 6) for v in sb],
                "lat": float(s.get("lat") or lat),
                "lon": float(s.get("lon") or lon),
                "width_m": round(w_m * 1.18, 1),
                "depth_m": round(d_m * 1.18, 1),
                "stated_height_m": s.get("stated_height_m"),
                "score": round(float(s.get("score") or 0.4), 3),
                "reason": "detected in scan window",
                "reason_fa": "تشخیص خودکار در پنجرهٔ اسکن",
                "compact": bool(s.get("compact")),
            })
    except Exception as exc:
        log.warning("suggest scan failed: %s", exc)

    items.sort(key=lambda row: (-float(row.get("score") or 0),
                                0 if row.get("kind") == "landmark" else 1))
    return {
        "lat": lat,
        "lon": lon,
        "span_m": span,
        "scan_bbox": [round(v, 6) for v in scan],
        "count": len(items),
        "items": items,
        "best": items[0] if items else None,
        "aoi_id": (survey_payload or {}).get("aoi_id"),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def _haversine_m_local(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def geocode(query: str, limit: int = 5, timeout: float = 12.0) -> list[dict[str, Any]]:
    """Place name -> coordinates, via OpenStreetMap Nominatim (free, no key).

    Failure here is never fatal: the operator can always type coordinates, so
    a timeout returns an empty list rather than an error.
    """
    import json
    import urllib.parse
    import urllib.request

    url = ("https://nominatim.openstreetmap.org/search?"
           + urllib.parse.urlencode({"q": query, "format": "json", "limit": limit}))
    request = urllib.request.Request(url, headers={
        "User-Agent": "ShadowHunter/1.0 (open-source building-height research)",
        "Accept-Language": "en",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        log.warning("geocode failed for %r: %s", query, exc)
        return []

    out: list[dict[str, Any]] = []
    for hit in raw:
        box = hit.get("boundingbox")
        out.append({
            "name": hit.get("display_name", query),
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
            # Nominatim orders it south, north, west, east
            "bbox": [float(box[2]), float(box[0]), float(box[3]), float(box[1])] if box else None,
            "kind": hit.get("type"),
        })
    if out:
        try:
            from .places import record_event
            hit = out[0]
            record_event("search", query=query, lat=hit["lat"], lon=hit["lon"],
                         name=hit.get("name"), bbox=hit.get("bbox"))
        except Exception:
            pass
    return out
