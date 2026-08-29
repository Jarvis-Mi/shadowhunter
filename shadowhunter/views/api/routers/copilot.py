"""Copilot, raster inspect, shadow timeline, web intel, 3D workspace."""
from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ....core import brain
from ....core.cache import hash_key, instance as cache
from ....core.config import SETTINGS
from ....core.geo import quality_of_geometry
from ....core.logging import get_logger
from ....models import intel as intel_model
from ....models import survey as survey_model
from ....models.schemas import BriefRequest, FavoriteIn, FieldRunRequest, PlanRequest, Scene3DRequest
from ....models.survey import parse_when

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["copilot"])

_DEPTH_PIXEL_CAP = 640 * 640


def _aoi(aoi_id: str) -> dict[str, Any]:
    entry = survey_model.AOI_CACHE.get(aoi_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown or expired aoi: {aoi_id}")
    return entry


def _workspace(aoi_id: str) -> Path:
    safe = Path(aoi_id).name
    if not safe or safe in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid aoi id")
    return Path(SETTINGS.workspace_dir) / safe


def _survey_from_entry(aoi_id: str, entry: dict[str, Any],
                       inspect: dict[str, Any] | None = None) -> dict[str, Any]:
    mosaic = entry["mosaic"]
    scene = entry.get("scene")
    sun = entry["sun"]
    structures = list(entry.get("structures") or [])
    lon, lat = mosaic.center
    meta = mosaic.meta()
    meta["name"] = getattr(scene, "name", meta.get("provider") or aoi_id)
    meta["source"] = getattr(scene, "source", "basemap")
    meta["center"] = [round(lon, 6), round(lat, 6)]
    scores = [float(s.get("score") or 0) for s in structures]
    payload: dict[str, Any] = {
        "aoi_id": aoi_id,
        "scene": meta,
        "sun": {
            "azimuth_deg": round(float(sun.azimuth_deg), 2),
            "elevation_deg": round(float(sun.elevation_deg), 2),
            "gsd_m": round(float(sun.gsd_m), 4),
            "shadow_bearing_deg": round(float(sun.shadow_bearing_deg), 2),
            "quality": round(quality_of_geometry(sun), 3),
            "when": entry.get("when"),
            "is_daylight": float(sun.elevation_deg) > 0.0,
            "season": entry.get("season"),
            "year_curve": entry.get("year_curve") or [],
        },
        "sun_estimate": entry.get("sun_estimate") or {},
        "structures": structures,
        "count": len(structures),
        "with_shadow": sum(
            1 for s in structures
            if float(s.get("shadow_len_px") or 0) > 2
            or float(s.get("shadow_support") or 0) > 0.15
        ),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "shadow_coverage": (inspect or {}).get("shadow_fraction", 0.0),
        "water_coverage": 0.0,
    }
    return payload


@router.get("/brain/health")
async def brain_health():
    models = brain.list_local_models()
    return {
        "llm_up": bool(models) or brain.llm_up(),
        "models": models,
        "configured": brain.configured_models(),
        "cache": cache().stats(),
        "tools": brain.list_tools(),
    }


@router.post("/brain/plan")
async def brain_plan(body: PlanRequest):
    return brain.plan_run(body.model_dump())


@router.get("/brain/tools")
async def brain_tools():
    return {"items": brain.list_tools()}


@router.post("/brain/brief")
async def brief(body: BriefRequest | None = None):
    body = body or BriefRequest()
    aoi_id = body.aoi_id
    locale = body.locale
    if aoi_id:
        ckey = hash_key("brief", aoi_id, locale)
        hit = cache().get("brief", ckey)
        if isinstance(hit, dict):
            return {**hit, "cached": True}

    inspect_info: dict[str, Any] | None = None
    timeline: dict[str, Any] | None = None
    intel: dict[str, Any] | None = None
    survey_payload: dict[str, Any] = {"locale": locale}

    if aoi_id:
        entry = _aoi(aoi_id)
        mosaic = entry["mosaic"]
        try:
            from ....models.inspect import inspect_raster, inspect_to_jsonable
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=f"inspect module unavailable: {exc}") from exc
        inspect_info = inspect_to_jsonable(inspect_raster(mosaic.image))
        lon, lat = mosaic.center
        try:
            from ....models.timeline import build_timeline
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=f"timeline module unavailable: {exc}") from exc
        timeline = build_timeline(lat, lon, parse_when(entry.get("when")), mosaic.gsd_m)
        intel = intel_model.enrich_aoi(mosaic.bbox, lat, lon)
        survey_payload = _survey_from_entry(aoi_id, entry, inspect_info)
        try:
            vision = brain.see_image(mosaic.image, cache_key=aoi_id)
            if vision.get("text") and isinstance(inspect_info, dict):
                inspect_info["vlm"] = {
                    "text": vision["text"], "model": vision.get("model"),
                    "used_vlm": bool(vision.get("used_vlm")),
                }
            if vision.get("text"):
                intel = brain.rerank_intel(intel, vision["text"])
        except Exception as exc:
            log.debug("brief vlm skipped: %s", exc)

    decision = brain.decide(
        survey_payload, inspect=inspect_info, timeline=timeline,
        intel=intel, locale=locale,
    )
    construction = brain.construct(
        survey_payload, inspect=inspect_info, timeline=timeline,
        intel=intel, locale=locale,
    )
    result = {
        "aoi_id": aoi_id,
        "inspect": inspect_info,
        "timeline": {k: v for k, v in (timeline or {}).items() if k != "samples"} if timeline else None,
        "intel": intel,
        **decision,
        "construct": construction,
        "build_fa": construction.get("build_fa"),
        "build_en": construction.get("build_en"),
    }
    if aoi_id:
        cache().set("brief", hash_key("brief", aoi_id, locale), result, ttl_s=3600)
    return result


@router.post("/brain/construct")
async def brain_construct(body: BriefRequest | None = None):
    body = body or BriefRequest()
    aoi_id = body.aoi_id
    locale = body.locale
    inspect_info: dict[str, Any] | None = None
    timeline: dict[str, Any] | None = None
    intel: dict[str, Any] | None = None
    survey_payload: dict[str, Any] = {"locale": locale}
    if aoi_id:
        entry = _aoi(aoi_id)
        mosaic = entry["mosaic"]
        try:
            from ....models.inspect import inspect_raster, inspect_to_jsonable
            from ....models.timeline import build_timeline
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=f"construct modules unavailable: {exc}") from exc
        inspect_info = inspect_to_jsonable(inspect_raster(mosaic.image))
        lon, lat = mosaic.center
        timeline = build_timeline(lat, lon, parse_when(entry.get("when")), mosaic.gsd_m)
        intel = intel_model.enrich_aoi(mosaic.bbox, lat, lon)
        survey_payload = _survey_from_entry(aoi_id, entry, inspect_info)
        try:
            vision = brain.see_image(mosaic.image, cache_key=aoi_id)
            if vision.get("text") and isinstance(inspect_info, dict):
                inspect_info["vlm"] = {
                    "text": vision["text"], "model": vision.get("model"),
                    "used_vlm": bool(vision.get("used_vlm")),
                }
            if vision.get("text"):
                intel = brain.rerank_intel(intel, vision["text"])
        except Exception as exc:
            log.debug("construct vlm skipped: %s", exc)
    return brain.construct(
        survey_payload, inspect=inspect_info, timeline=timeline,
        intel=intel, locale=locale,
    )


@router.get("/intel")
async def intel(west: float, south: float, east: float, north: float):
    bbox = (min(west, east), min(south, north), max(west, east), max(south, north))
    lat = (bbox[1] + bbox[3]) / 2
    lon = (bbox[0] + bbox[2]) / 2
    return intel_model.enrich_aoi(bbox, lat, lon)


@router.get("/inspect/{aoi_id}")
async def inspect_aoi(aoi_id: str):
    try:
        from ....models.inspect import inspect_raster, inspect_to_jsonable
        from ....models.pipeline import png_b64
        import cv2
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"inspect module unavailable: {exc}") from exc

    entry = _aoi(aoi_id)
    raw = inspect_raster(entry["mosaic"].image)
    from ....models.field import _spectrum_pngs
    pngs = _spectrum_pngs(raw)
    stats = inspect_to_jsonable(raw)
    return {**stats, **pngs}


@router.get("/timeline")
async def timeline(lat: float, lon: float, when: str | None = None,
                   step_minutes: int = Query(60, ge=10, le=180)):
    try:
        from ....models.pipeline import png_b64
        from ....models.timeline import build_timeline, render_timeline_strip
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"timeline module unavailable: {exc}") from exc

    moment = parse_when(when)
    payload = build_timeline(lat, lon, moment, step_minutes=step_minutes)
    strip = render_timeline_strip(payload.get("samples") or [])
    payload["strip_png"] = png_b64(strip)
    return payload


@router.post("/scene3d")
async def scene3d(body: Scene3DRequest):
    try:
        from ....models import reconstruct
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"reconstruct module unavailable: {exc}") from exc

    entry = _aoi(body.aoi_id)
    mosaic = entry["mosaic"]
    out_dir = _workspace(body.aoi_id)
    try:
        result = reconstruct.export_scene(
            out_dir, mosaic_bgr=mosaic.image,
            structures=entry.get("structures") or [],
            measures=body.measures,
            gsd_m=mosaic.gsd_m,
            sun={"azimuth_deg": entry["sun"].azimuth_deg,
                 "elevation_deg": entry["sun"].elevation_deg,
                 "gsd_m": entry["sun"].gsd_m},
            meta=mosaic.meta(),
        )
    except Exception as exc:
        log.exception("scene3d export failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    if body.open:
        html = result.get("html")
        if html:
            try:
                webbrowser.open(Path(html).as_uri())
            except Exception as exc:
                log.debug("scene3d open skipped: %s", exc)
    return result


@router.get("/workspace/{aoi_id}/{name}")
async def workspace_file(aoi_id: str, name: str):
    from fastapi.responses import FileResponse

    folder = _workspace(aoi_id).resolve()
    path = (folder / Path(name).name).resolve()
    if not str(path).startswith(str(folder)) or not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    media = {".html": "text/html", ".json": "application/json",
             ".gltf": "model/gltf+json", ".jpg": "image/jpeg"}.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=media)


@router.get("/places/favorites")
async def places_favorites():
    from ....models import places
    return {"items": places.list_favorites()}


@router.post("/places/favorites")
async def places_add(body: FavoriteIn):
    from ....models import places
    return places.add_favorite(body.name, body.lat, body.lon, body.bbox, body.note)


@router.delete("/places/favorites/{fav_id}")
async def places_remove(fav_id: str):
    from ....models import places
    if not places.remove_favorite(fav_id):
        raise HTTPException(status_code=404, detail="favorite not found")
    return {"ok": True}


@router.get("/places/history")
async def places_history(limit: int = Query(40, ge=1, le=200)):
    from ....models import places
    return {"items": places.list_history(limit)}


@router.post("/field/run")
async def field_run(body: FieldRunRequest):
    from ....models.field import run_field
    bbox = body.bbox.as_tuple()
    try:
        return run_field(bbox, locale=body.locale, query=body.query, when=body.when)
    except Exception as exc:
        log.exception("field run failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
