"""One-click field run: plan → survey → spectra → measure → 3D → brief."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from ..core import brain
from ..core.config import SETTINGS
from ..core.logging import get_logger
from .inspect import inspect_raster, inspect_to_jsonable
from .pipeline import png_b64
from .survey import AOI_CACHE, analyze_survey, parse_when, survey

log = get_logger(__name__)


def _span_m(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    try:
        from .vision.tiles import bbox_span_m
        return bbox_span_m(bbox)
    except Exception:
        west, south, east, north = bbox
        return abs(east - west) * 111_000.0, abs(north - south) * 111_000.0


def _save_spectra(aoi_id: str, raw: dict[str, Any]) -> dict[str, str]:
    folder = Path(SETTINGS.workspace_dir) / Path(aoi_id).name
    folder.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    mapping = {
        "_depth_u8": ("depth.jpg", cv2.COLORMAP_INFERNO),
        "_false_u8": ("false.jpg", cv2.COLORMAP_TURBO),
        "_lab_u8": ("lab.jpg", None),
        "_boost_u8": ("boost.jpg", None),
        "_hsv_u8": ("hsv.jpg", "hsv"),
    }
    for key, (name, cmap) in mapping.items():
        arr = raw.get(key)
        if arr is None:
            continue
        try:
            if cmap == "hsv" and getattr(arr, "ndim", 0) == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)
            elif cmap is not None and getattr(arr, "ndim", 0) == 2:
                bgr = cv2.applyColorMap(arr, cmap)
            elif getattr(arr, "ndim", 0) == 2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                bgr = arr
            path = folder / name
            cv2.imwrite(str(path), bgr)
            saved[name] = str(path)
        except Exception as exc:
            log.debug("spectra write %s skipped: %s", name, exc)
    return saved


def _spectrum_pngs(raw: dict[str, Any]) -> dict[str, str | None]:
    pngs: dict[str, str | None] = {}
    pairs = (
        ("_depth_u8", "depth_png", cv2.COLORMAP_INFERNO),
        ("_false_u8", "false_png", cv2.COLORMAP_TURBO),
        ("_lab_u8", "lab_png", None),
        ("_boost_u8", "boost_png", None),
        ("_hsv_u8", "hsv_png", "hsv"),
    )
    for key, out_key, cmap in pairs:
        arr = raw.get(key)
        if arr is None:
            pngs[out_key] = None
            continue
        try:
            if cmap == "hsv" and getattr(arr, "ndim", 0) == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)
            elif cmap is not None and getattr(arr, "ndim", 0) == 2:
                bgr = cv2.applyColorMap(arr, cmap)
            elif getattr(arr, "ndim", 0) == 2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                bgr = arr
            pngs[out_key] = png_b64(bgr)
        except Exception:
            pngs[out_key] = None
    return pngs


def run_field(bbox: tuple[float, float, float, float], *, locale: str = "fa",
              query: str | None = None, when: str | None = None) -> dict[str, Any]:
    """Operator path: few buttons, LLM (or heuristic) chooses the rest."""
    west, south, east, north = bbox
    lat = (south + north) / 2.0
    lon = (west + east) / 2.0
    width_m, height_m = _span_m(bbox)
    span = max(width_m, height_m)
    plan = brain.plan_run({
        "lat": lat, "lon": lon, "span_m": span,
        "width_m": width_m, "height_m": height_m, "query": query,
    })

    survey_payload = survey(
        bbox, provider="esri", max_tiles=int(plan["max_tiles"]),
        when=when, auto_sun=bool(plan.get("auto_sun", True)),
        min_size_m=float(plan["min_size_m"]), max_structures=80,
    )
    aoi_id = survey_payload["aoi_id"]
    entry = AOI_CACHE.get(aoi_id)
    mosaic = entry["mosaic"] if entry else None

    inspect_info: dict[str, Any] = {}
    pngs: dict[str, str | None] = {}
    saved: dict[str, str] = {}
    if mosaic is not None:
        raw = inspect_raster(mosaic.image)
        saved = _save_spectra(aoi_id, raw)
        pngs = _spectrum_pngs(raw)
        inspect_info = inspect_to_jsonable(raw)

    measures: dict[str, Any] = {"items": []}
    try:
        measures = analyze_survey(
            aoi_id, policy=str(plan.get("policy") or "auto"),
            limit=int(plan.get("measure_limit") or 16), max_steps=40,
        )
    except Exception as exc:
        log.warning("field measure failed: %s", exc)
        measures = {"items": [], "error": str(exc)}

    intel: dict[str, Any] | None = None
    try:
        from . import intel as intel_model
        intel = intel_model.enrich_aoi(bbox, lat, lon)
    except Exception as exc:
        log.debug("field intel skipped: %s", exc)

    if mosaic is not None:
        try:
            vision = brain.see_image(mosaic.image, cache_key=aoi_id)
            if vision.get("text"):
                inspect_info["vlm"] = {
                    "text": vision["text"][:4000],
                    "model": vision.get("model"),
                    "used_vlm": bool(vision.get("used_vlm")),
                }
                intel = brain.rerank_intel(intel, vision["text"], extra=query)
        except Exception as exc:
            log.debug("field vlm skipped: %s", exc)

    timeline: dict[str, Any] | None = None
    try:
        from . import timeline as timeline_mod
        from .timeline import build_timeline, render_timeline_strip
        used = (survey_payload.get("sun") or {}).get("when")
        when_dt = parse_when(used)
        gsd_m = mosaic.gsd_m if mosaic else 0.5
        timeline = build_timeline(lat, lon, when_dt, gsd_m)
        capture_slots = getattr(timeline_mod, "capture_slots", None)
        capture_calendar = getattr(timeline_mod, "capture_calendar", None)
        if callable(capture_slots) and not timeline.get("captures"):
            try:
                timeline["captures"] = capture_slots(lat, lon, when_dt, gsd_m=gsd_m)
            except Exception as exc:
                log.debug("capture_slots skipped: %s", exc)
        if callable(capture_calendar) and timeline.get("calendar") is None:
            try:
                timeline["calendar"] = capture_calendar(
                    lat, lon, int(when_dt.year), gsd_m=gsd_m)
            except Exception as exc:
                log.debug("capture_calendar skipped: %s", exc)
        samples = list(timeline.get("samples") or [])
        try:
            timeline["strip_png"] = png_b64(render_timeline_strip(samples))
        except Exception as exc:
            log.debug("timeline strip skipped: %s", exc)
    except Exception as exc:
        log.debug("field timeline skipped: %s", exc)

    brief = brain.decide(
        survey_payload, inspect=inspect_info, timeline=timeline,
        intel=intel, measures=measures, locale=locale,
    )

    construct_payload = None
    if hasattr(brain, "construct"):
        try:
            construct_payload = brain.construct(
                survey_payload, inspect=inspect_info, timeline=timeline,
                intel=intel, measures=measures, locale=locale,
            )
        except Exception as exc:
            log.debug("field construct skipped: %s", exc)

    scene3d: dict[str, Any] = {}
    try:
        from . import reconstruct
        if mosaic is not None:
            meta = {**(mosaic.meta() if mosaic else {})}
            if entry and entry.get("landmark"):
                meta["landmark"] = entry["landmark"]
            if inspect_info.get("vlm"):
                meta["vlm"] = inspect_info["vlm"]
            if construct_payload:
                meta["construct"] = construct_payload
            scene3d = reconstruct.export_scene(
                Path(SETTINGS.workspace_dir) / aoi_id,
                mosaic_bgr=mosaic.image,
                structures=(entry or {}).get("structures") or [],
                measures=measures.get("items"),
                gsd_m=mosaic.gsd_m,
                sun={"azimuth_deg": entry["sun"].azimuth_deg,
                     "elevation_deg": entry["sun"].elevation_deg,
                     "gsd_m": entry["sun"].gsd_m} if entry else None,
                meta=meta,
            )
    except Exception as exc:
        log.warning("field 3d failed: %s", exc)
        scene3d = {"error": str(exc)}

    try:
        from .places import record_event
        record_event(
            "run", lat=lat, lon=lon, bbox=list(bbox), aoi_id=aoi_id,
            name=query or (survey_payload.get("scene") or {}).get("name"),
            extra={"season": survey_payload.get("season"), "plan": plan.get("source")},
        )
    except Exception:
        pass

    slim_timeline = None
    if isinstance(timeline, dict):
        samples = list(timeline.get("samples") or [])
        slim_timeline = dict(timeline)
        slim_timeline["samples"] = samples
        try:
            from .timeline import render_timeline_strip
            slim_timeline["strip_png"] = png_b64(render_timeline_strip(samples))
        except Exception as exc:
            log.debug("timeline strip skipped: %s", exc)
            slim_timeline.setdefault("strip_png", None)
        captures = slim_timeline.get("captures")
        if not captures:
            try:
                from . import timeline as timeline_mod
                used = (survey_payload.get("sun") or {}).get("when")
                when_dt = parse_when(used)
                gsd_m = mosaic.gsd_m if mosaic else 0.5
                capture_slots = getattr(timeline_mod, "capture_slots", None)
                capture_calendar = getattr(timeline_mod, "capture_calendar", None)
                if callable(capture_slots):
                    captures = capture_slots(lat, lon, when_dt, gsd_m=gsd_m)
                elif callable(capture_calendar):
                    cal = capture_calendar(lat, lon, int(when_dt.year), gsd_m=gsd_m)
                    slim_timeline.setdefault("calendar", cal)
                    captures = []
                    for season in (cal.get("seasons") or []) if isinstance(cal, dict) else []:
                        captures.extend(season.get("slots") or [])
            except Exception as exc:
                log.debug("timeline captures skipped: %s", exc)
        if captures:
            slim_timeline["captures"] = captures

    result: dict[str, Any] = {
        "aoi_id": aoi_id,
        "plan": plan,
        "survey": survey_payload,
        "inspect": {**inspect_info, **pngs, "saved": saved},
        "measures": measures,
        "scene3d": scene3d,
        "intel": intel,
        "timeline": slim_timeline,
        "brief": brief,
    }
    if construct_payload is not None:
        result["construct"] = construct_payload
    return result
