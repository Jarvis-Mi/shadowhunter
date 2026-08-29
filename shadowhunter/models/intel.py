"""Internet enrichment for an AOI: place names, OSM footprints, Wikipedia.

Every call is fail-soft and cached for 24 hours. A Nominatim timeout must
never take the survey down - the operator can still work from the mosaic.
Stdlib ``urllib`` only; no extra packages.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..core.cache import hash_key, instance as cache
from ..core.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "ShadowHunter/1.1 (open-source research)"
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
OVERPASS = "https://overpass-api.de/api/interpreter"
TTL_S = 86400
_HEIGHT_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _http_json(url: str, *, data: bytes | None = None, timeout: float = 12.0,
               headers: dict[str, str] | None = None, method: str | None = None) -> Any | None:
    hdrs = {**_headers(), **(headers or {})}
    request = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        log.warning("intel request failed %s: %s", url.split("?", 1)[0], exc)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("intel JSON decode failed: %s", exc)
        return None


def _parse_height_m(tag: str | None) -> float | None:
    if not tag:
        return None
    text = str(tag).strip().lower().replace(",", ".")
    match = _HEIGHT_NUM.search(text)
    if not match:
        return None
    try:
        value = float(match.group())
    except ValueError:
        return None
    if any(tok in text for tok in ("ft", "feet", "foot")) or "'" in str(tag):
        value *= 0.3048
    return round(value, 2)


def _parse_levels(tag: str | None) -> int | None:
    if not tag:
        return None
    match = _HEIGHT_NUM.search(str(tag).replace(",", "."))
    if not match:
        return None
    try:
        return max(0, int(round(float(match.group()))))
    except ValueError:
        return None


def _round_bbox(bbox: tuple[float, float, float, float], ndigits: int = 4
                ) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    west, east = min(west, east), max(west, east)
    south, north = min(south, north), max(south, north)
    return (round(west, ndigits), round(south, ndigits),
            round(east, ndigits), round(north, ndigits))


# --------------------------------------------------------------------------- #
# Nominatim
# --------------------------------------------------------------------------- #
def reverse_geocode(lat: float, lon: float, timeout: float = 12.0) -> dict[str, Any] | None:
    """Nominatim reverse geocode. ``None`` on any failure."""
    key = hash_key("reverse", round(float(lat), 5), round(float(lon), 5))
    hit = cache().get("intel", key)
    if isinstance(hit, dict):
        return hit

    query = urllib.parse.urlencode({
        "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
        "format": "json", "addressdetails": 1, "zoom": 18,
    })
    raw = _http_json(f"{NOMINATIM}?{query}", timeout=timeout)
    if not isinstance(raw, dict) or not raw.get("display_name"):
        return None
    place = {
        "name": raw.get("display_name"),
        "lat": float(raw.get("lat", lat)),
        "lon": float(raw.get("lon", lon)),
        "address": raw.get("address") or {},
        "kind": raw.get("type") or raw.get("class") or raw.get("category"),
        "osm_type": raw.get("osm_type"),
        "osm_id": raw.get("osm_id"),
    }
    cache().set("intel", key, place, ttl_s=TTL_S)
    return place


# --------------------------------------------------------------------------- #
# Overpass
# --------------------------------------------------------------------------- #
def overpass_buildings(bbox: tuple[float, float, float, float], timeout: float = 25.0
                       ) -> dict[str, Any]:
    """OSM building footprints inside ``bbox`` (west, south, east, north)."""
    west, south, east, north = _round_bbox(bbox)
    key = hash_key("overpass", west, south, east, north)
    hit = cache().get("intel", key)
    if isinstance(hit, dict) and "items" in hit:
        return {**hit, "cached": True}

    query = (
        f"[out:json][timeout:{int(timeout)}];"
        f'(way["building"]({south},{west},{north},{east});'
        f'relation["building"]({south},{west},{north},{east}););'
        "out tags center 80;"
    )
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    raw = _http_json(
        OVERPASS, data=body, timeout=float(timeout) + 8.0,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    if not isinstance(raw, dict):
        return {"count": 0, "items": [], "source": "overpass", "cached": False,
                "error": "overpass unreachable"}

    items: list[dict[str, Any]] = []
    for element in (raw.get("elements") or [])[:80]:
        tags = element.get("tags") or {}
        center = element.get("center") or {}
        lat = center.get("lat", element.get("lat"))
        lon = center.get("lon", element.get("lon"))
        try:
            lat_f = float(lat) if lat is not None else None
            lon_f = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            lat_f = lon_f = None
        items.append({
            "name": tags.get("name") or tags.get("name:en") or tags.get("name:fa") or "",
            "type": tags.get("building") or "yes",
            "levels": _parse_levels(tags.get("building:levels")),
            "height_tag_m": _parse_height_m(tags.get("height") or tags.get("building:height")),
            "lon": round(lon_f, 6) if lon_f is not None else None,
            "lat": round(lat_f, 6) if lat_f is not None else None,
        })
    payload = {"count": len(items), "items": items, "source": "overpass",
               "cached": False, "error": None}
    cache().set("intel", key, payload, ttl_s=TTL_S)
    return payload


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #
def _wiki_geosearch(lang: str, lat: float, lon: float, radius_m: int, limit: int,
                    timeout: float) -> list[dict[str, Any]]:
    radius = max(10, min(int(radius_m), 10_000))
    query = urllib.parse.urlencode({
        "action": "query", "list": "geosearch",
        "gscoord": f"{lat}|{lon}", "gsradius": radius,
        "gslimit": max(1, min(int(limit), 50)), "format": "json",
    })
    url = f"https://{lang}.wikipedia.org/w/api.php?{query}"
    raw = _http_json(url, timeout=timeout)
    if not isinstance(raw, dict):
        return []
    out: list[dict[str, Any]] = []
    for hit in (raw.get("query") or {}).get("geosearch") or []:
        title = hit.get("title")
        if not title:
            continue
        dist = hit.get("dist")
        try:
            dist_m = float(dist) if dist is not None else None
        except (TypeError, ValueError):
            dist_m = None
        out.append({
            "title": title,
            "dist_m": round(dist_m, 1) if dist_m is not None else None,
            "lang": lang,
            "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
        })
    return out


def _wiki_extracts(lang: str, titles: list[str], timeout: float) -> dict[str, str]:
    if not titles:
        return {}
    query = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
        "redirects": 1, "format": "json", "titles": "|".join(titles[:8]),
        "exlimit": min(len(titles), 8),
    })
    url = f"https://{lang}.wikipedia.org/w/api.php?{query}"
    raw = _http_json(url, timeout=timeout)
    if not isinstance(raw, dict):
        return {}
    pages = (raw.get("query") or {}).get("pages") or {}
    out: dict[str, str] = {}
    for page in pages.values():
        title = page.get("title")
        extract = (page.get("extract") or "").strip()
        if title and extract:
            out[title] = extract[:800]
    return out


def wikipedia_nearby(lat: float, lon: float, radius_m: int = 800
                     ) -> list[dict[str, Any]]:
    """Nearby Wikipedia pages (English, plus Farsi when it answers). Fail-soft."""
    key = hash_key("wiki", round(float(lat), 5), round(float(lon), 5), int(radius_m))
    hit = cache().get("intel", key)
    if isinstance(hit, list):
        return hit

    by_title: dict[str, dict[str, Any]] = {}
    for lang in ("en", "fa"):
        for item in _wiki_geosearch(lang, lat, lon, radius_m, limit=8, timeout=10.0):
            stamp = item["title"].casefold()
            previous = by_title.get(stamp)
            if previous is None or (item.get("dist_m") or 1e9) < (previous.get("dist_m") or 1e9):
                by_title[stamp] = item
    ranked = sorted(by_title.values(),
                    key=lambda i: i.get("dist_m") if i.get("dist_m") is not None else 1e9)

    for lang in ("en", "fa"):
        subset = [i for i in ranked[:3] if i.get("lang") == lang]
        extracts = _wiki_extracts(lang, [i["title"] for i in subset], timeout=10.0)
        for item in subset:
            summary = extracts.get(item["title"])
            if summary:
                item["summary"] = summary

    out: list[dict[str, Any]] = []
    for item in ranked[:8]:
        out.append({
            "title": item["title"],
            "summary": item.get("summary") or "",
            "url": item["url"],
            "dist_m": item.get("dist_m"),
        })
    cache().set("intel", key, out, ttl_s=TTL_S)
    return out


# --------------------------------------------------------------------------- #
# Combined AOI
# --------------------------------------------------------------------------- #
def enrich_aoi(bbox: tuple[float, float, float, float],
               center_lat: float, center_lon: float) -> dict[str, Any]:
    """Reverse + Overpass + Wikipedia, cached on a rounded bbox."""
    rounded = _round_bbox(bbox)
    key = hash_key("enrich", *rounded)
    hit = cache().get("intel", key)
    if isinstance(hit, dict) and ("overpass" in hit or "buildings" in hit):
        return {**hit, "cached": True}

    reverse = reverse_geocode(center_lat, center_lon)
    overpass = overpass_buildings(bbox)
    wiki = wikipedia_nearby(center_lat, center_lon)
    payload = {
        "reverse": reverse,
        "overpass": overpass,
        "wikipedia": wiki,
        "place": reverse,
        "buildings": overpass,
        "center": [round(center_lat, 6), round(center_lon, 6)],
        "bbox": list(rounded),
        "cached": False,
    }
    cache().set("intel", key, payload, ttl_s=TTL_S)
    return payload
