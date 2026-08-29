"""Map, sun and AOI endpoints - the real-imagery half of the API.

The tile route is a caching proxy rather than a redirect on purpose: every
front-end then gets identical imagery from one disk cache, the providers see
one polite client instead of five, and the desktop views need no network code
of their own.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ....core.geo import quality_of_geometry
from ....core.logging import get_logger
from ....core.sun import best_hour_of_day, daylight_window, solar_position, year_seasons
from ....models import survey as survey_model
from ....models.schemas import (GeocodeResult, SunQuery, SunResponse,
                                SurveyAnalyzeRequest, SurveyAnalyzeResponse,
                                SurveyRequest, SurveyResponse)
from ....models.vision import tiles as tile_model

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["atlas"])


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #
@router.get("/map/providers", summary="Available basemap and imagery sources")
async def providers():
    return {"items": list(tile_model.iter_providers()),
            "default_satellite": tile_model.DEFAULT_SATELLITE,
            "default_basemap": tile_model.DEFAULT_BASEMAP,
            "cache": tile_model.cache_stats()}


@router.get("/map/tile/{provider}/{z}/{x}/{y}", summary="Cached XYZ tile proxy")
async def tile(provider: str, z: int, x: int, y: int):
    if provider not in tile_model.PROVIDERS:
        raise HTTPException(status_code=404, detail=f"unknown provider: {provider}")
    if not 0 <= z <= tile_model.PROVIDERS[provider].max_zoom:
        raise HTTPException(status_code=400, detail=f"zoom out of range: {z}")

    blob = tile_model.tile_png_bytes(provider, z, x, y)
    if blob is None:
        raise HTTPException(status_code=502, detail="tile unavailable upstream")
    media = "image/jpeg" if tile_model.PROVIDERS[provider].ext == "jpg" else "image/png"
    return Response(content=blob, media_type=media,
                    headers={"Cache-Control": "public, max-age=604800"})


@router.get("/map/plan", summary="Cost of an AOI before downloading it")
async def plan(west: float, south: float, east: float, north: float,
               zoom: int | None = None, max_tiles: int = 36):
    bbox = (min(west, east), min(south, north), max(west, east), max(south, north))
    chosen = zoom or tile_model.choose_zoom(bbox, max_tiles)
    detail = tile_model.plan_mosaic(bbox, chosen)
    width_m, height_m = tile_model.bbox_span_m(bbox)
    return {**detail, "zoom": chosen, "width_m": round(width_m, 1),
            "height_m": round(height_m, 1),
            "affordable": detail["tiles"] <= max_tiles}


# --------------------------------------------------------------------------- #
# Sun
# --------------------------------------------------------------------------- #
@router.post("/sun", response_model=SunResponse, summary="Solar position for a place and time")
async def sun(query: SunQuery) -> SunResponse:
    moment = survey_model.parse_when(query.when)
    position = solar_position(query.lat, query.lon, moment)
    geometry = position.to_geometry(0.5)

    best_when = best_elevation = None
    if query.best_hour:
        when, best = best_hour_of_day(query.lat, query.lon, moment)
        best_when = when.isoformat(timespec="minutes")
        best_elevation = round(best.elevation_deg, 2)

    rise, set_ = daylight_window(query.lat, query.lon, moment)
    curve = year_seasons(query.lat, query.lon, moment.year)
    year_best = max(curve, key=lambda row: float(row.get("quality") or 0)) if curve else None
    return SunResponse(
        azimuth_deg=round(position.azimuth_deg, 2),
        elevation_deg=round(position.elevation_deg, 2),
        declination_deg=round(position.declination_deg, 2),
        is_daylight=position.is_daylight,
        when=moment.isoformat(timespec="minutes"),
        quality=round(quality_of_geometry(geometry), 3),
        shadow_bearing_deg=round(geometry.shadow_bearing_deg, 2),
        best_when=best_when, best_elevation_deg=best_elevation,
        sunrise=rise.isoformat(timespec="minutes") if rise else None,
        sunset=set_.isoformat(timespec="minutes") if set_ else None,
        season=(year_best or {}).get("season") if year_best else None,
        seasons=curve,
    )


# --------------------------------------------------------------------------- #
# AOI
# --------------------------------------------------------------------------- #
@router.post("/aoi/suggest", summary="Propose structure boxes before the operator draws an AOI")
async def aoi_suggest(lat: float = Query(..., ge=-90, le=90),
                      lon: float = Query(..., ge=-180, le=180),
                      span_m: float = Query(200.0, gt=40, le=800),
                      when: str | None = None):
    try:
        return survey_model.suggest_structures(
            lat, lon, span_m=span_m, when=when, provider="esri")
    except Exception as exc:
        log.exception("suggest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/aoi/survey", response_model=SurveyResponse,
             summary="Fetch an area's imagery, place the sun, count the structures")
async def aoi_survey(request: SurveyRequest) -> SurveyResponse:
    bbox = request.bbox.as_tuple()
    spec = tile_model.PROVIDERS.get(request.provider)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"unknown provider: {request.provider}")
    if spec.kind != "satellite":
        raise HTTPException(
            status_code=400,
            detail="survey needs satellite imagery — OSM/Carto have no building shadows",
        )
    width_m, height_m = tile_model.bbox_span_m(bbox)
    if width_m < 20 or height_m < 20:
        raise HTTPException(status_code=400,
                            detail="area too small - draw a rectangle at least 20 m across")
    try:
        payload = survey_model.survey(
            bbox, provider=request.provider, zoom=request.zoom,
            max_tiles=request.max_tiles, when=request.when, auto_sun=request.auto_sun,
            min_size_m=request.min_size_m, max_structures=request.max_structures,
            detect=request.detect,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("survey failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    if payload["scene"]["tiles"].split("/")[0] == "0":
        raise HTTPException(status_code=502,
                            detail="no imagery returned - check the network or try another provider")
    return SurveyResponse(**payload)


@router.post("/aoi/analyze", response_model=SurveyAnalyzeResponse,
             summary="Measure the height of structures found by a survey")
async def aoi_analyze(request: SurveyAnalyzeRequest) -> SurveyAnalyzeResponse:
    try:
        payload = survey_model.analyze_survey(
            request.aoi_id, indices=request.indices, policy=request.policy,
            max_steps=request.max_steps, limit=request.limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("aoi analysis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return SurveyAnalyzeResponse(**payload)


@router.get("/aoi", summary="AOIs still held in the server-side cache")
async def aoi_list():
    return {"items": survey_model.AOI_CACHE.keys()}


# --------------------------------------------------------------------------- #
# Geocoding
# --------------------------------------------------------------------------- #
@router.get("/geocode", summary="Place name to coordinates (OpenStreetMap Nominatim)")
async def geocode(q: str = Query(..., min_length=2), limit: int = 5) -> dict[str, list[GeocodeResult]]:
    return {"items": [GeocodeResult(**hit) for hit in survey_model.geocode(q, limit)]}
