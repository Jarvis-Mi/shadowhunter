"""Slippy-map tiles: Web-Mercator maths, cached fetching, and AOI mosaicking.

This is what turns the project from a demo on synthetic tiles into something
you point at a real city. Pick a rectangle anywhere on Earth, and this module
returns a georeferenced RGB mosaic with a **real** ground sample distance -
which is the number ``h = L*tan(theta)`` needs to produce metres.

All providers here are free to use. Esri World Imagery and OpenStreetMap both
require attribution, which every view displays.
"""
from __future__ import annotations

import math
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from ...core.config import SETTINGS
from ...core.logging import get_logger

log = get_logger(__name__)

TILE_PX = 256
EARTH_CIRCUMFERENCE_M = 40075016.686
CACHE_DIR = Path(SETTINGS.data_dir) / "tiles"
USER_AGENT = "ShadowHunter/1.0 (open-source building-height research)"


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    url: str                  # {z}/{x}/{y} template
    kind: str                 # "satellite" | "basemap"
    max_zoom: int
    attribution: str
    ext: str = "png"


PROVIDERS: dict[str, Provider] = {
    "esri": Provider(
        key="esri", label="Esri World Imagery", kind="satellite", max_zoom=19, ext="jpg",
        url="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="Esri, Maxar, Earthstar Geographics",
    ),
    "carto_dark": Provider(
        key="carto_dark", label="Carto Dark Matter", kind="basemap", max_zoom=19,
        url="https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attribution="OpenStreetMap contributors, CARTO",
    ),
    "osm": Provider(
        key="osm", label="OpenStreetMap", kind="basemap", max_zoom=19,
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="OpenStreetMap contributors",
    ),
}

DEFAULT_SATELLITE = "esri"
DEFAULT_BASEMAP = "carto_dark"


# --------------------------------------------------------------------------- #
# Web-Mercator maths
# --------------------------------------------------------------------------- #
def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Fractional tile coordinates - keep the fraction for sub-tile precision."""
    lat = max(-85.05112878, min(85.05112878, lat))
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def ground_resolution(lat: float, zoom: int) -> float:
    """Metres per pixel at this latitude and zoom - the GSD the physics uses."""
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (TILE_PX * 2.0 ** zoom)


def zoom_for_resolution(lat: float, target_gsd_m: float) -> int:
    """Smallest zoom whose GSD is at least as fine as requested."""
    for z in range(0, 20):
        if ground_resolution(lat, z) <= target_gsd_m:
            return z
    return 19


def bbox_span_m(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """(width_m, height_m) of a (west, south, east, north) box."""
    west, south, east, north = bbox
    lat_mid = (south + north) / 2
    width = abs(east - west) * 111_320.0 * math.cos(math.radians(lat_mid))
    height = abs(north - south) * 110_540.0
    return width, height


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(path), threading.Lock())


def fetch_tile(provider_key: str, z: int, x: int, y: int, *,
               timeout: float = 12.0, use_cache: bool = True) -> np.ndarray | None:
    """One 256x256 BGR tile, from the disk cache when possible."""
    provider = PROVIDERS.get(provider_key)
    if provider is None:
        raise KeyError(f"unknown tile provider: {provider_key}")

    n = 2 ** z
    if not (0 <= x < n and 0 <= y < n):
        return None

    path = CACHE_DIR / provider.key / str(z) / str(x) / f"{y}.{provider.ext}"
    if use_cache and path.exists() and path.stat().st_size > 0:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img

    url = provider.url.format(z=z, x=x, y=y)
    with _lock_for(path):
        if use_cache and path.exists() and path.stat().st_size > 0:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                blob = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.debug("tile %s/%d/%d/%d failed: %s", provider.key, z, x, y, exc)
            return None

        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        return img


def tile_png_bytes(provider_key: str, z: int, x: int, y: int) -> bytes | None:
    """Raw bytes for the HTTP tile proxy - avoids a decode/encode round trip."""
    provider = PROVIDERS.get(provider_key)
    if provider is None:
        return None
    path = CACHE_DIR / provider.key / str(z) / str(x) / f"{y}.{provider.ext}"
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    if fetch_tile(provider_key, z, x, y) is None:
        return None
    return path.read_bytes() if path.exists() else None


# --------------------------------------------------------------------------- #
# Mosaic
# --------------------------------------------------------------------------- #
@dataclass
class Mosaic:
    """A stitched AOI image plus everything needed to georeference a pixel."""
    image: np.ndarray                      # BGR uint8
    bbox: tuple[float, float, float, float]   # west, south, east, north (actual, snapped)
    zoom: int
    gsd_m: float
    provider: str
    attribution: str
    origin_px: tuple[float, float]         # world-pixel coords of the image's top-left
    tiles_ok: int = 0
    tiles_total: int = 0

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    @property
    def center(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2

    def pixel_to_lonlat(self, px: float, py: float) -> tuple[float, float]:
        world_x = (self.origin_px[0] + px) / TILE_PX
        world_y = (self.origin_px[1] + py) / TILE_PX
        return tile_to_lonlat(world_x, world_y, self.zoom)

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        tx, ty = lonlat_to_tile(lon, lat, self.zoom)
        return tx * TILE_PX - self.origin_px[0], ty * TILE_PX - self.origin_px[1]

    def meta(self) -> dict[str, Any]:
        west, south, east, north = self.bbox
        width_m, height_m = bbox_span_m(self.bbox)
        return {
            "bbox": [west, south, east, north],
            "zoom": self.zoom,
            "gsd_m": round(self.gsd_m, 4),
            "provider": self.provider,
            "attribution": self.attribution,
            "width_px": self.image.shape[1],
            "height_px": self.image.shape[0],
            "width_m": round(width_m, 1),
            "height_m": round(height_m, 1),
            "tiles": f"{self.tiles_ok}/{self.tiles_total}",
            "complete": self.tiles_ok == self.tiles_total,
        }


def _tile_range(bbox: tuple[float, float, float, float], zoom: int) -> tuple[int, int, int, int]:
    west, south, east, north = bbox
    x0f, y0f = lonlat_to_tile(west, north, zoom)     # north-west corner
    x1f, y1f = lonlat_to_tile(east, south, zoom)     # south-east corner
    return math.floor(x0f), math.floor(y0f), math.floor(x1f), math.floor(y1f)


def plan_mosaic(bbox: tuple[float, float, float, float], zoom: int) -> dict[str, Any]:
    """How many tiles and pixels an AOI would cost, without downloading."""
    x0, y0, x1, y1 = _tile_range(bbox, zoom)
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    lat_mid = (bbox[1] + bbox[3]) / 2
    return {
        "tiles": cols * rows,
        "cols": cols,
        "rows": rows,
        "width_px": cols * TILE_PX,
        "height_px": rows * TILE_PX,
        "gsd_m": round(ground_resolution(lat_mid, zoom), 4),
    }


def choose_zoom(bbox: tuple[float, float, float, float], max_tiles: int = 36,
                max_zoom: int = 19) -> int:
    """Finest zoom whose mosaic still fits the tile budget.

    Detail is the whole point - a 10 m Sentinel pixel cannot resolve a shadow
    edge - so this always reaches for the sharpest level that stays affordable
    rather than a fixed zoom.
    """
    for zoom in range(max_zoom, 0, -1):
        if plan_mosaic(bbox, zoom)["tiles"] <= max_tiles:
            return zoom
    return 1


def fetch_mosaic(bbox: tuple[float, float, float, float], *, zoom: int | None = None,
                 provider: str = DEFAULT_SATELLITE, max_tiles: int = 36,
                 crop_to_bbox: bool = True, workers: int = 8) -> Mosaic:
    """Download and stitch the tiles covering ``bbox`` (west, south, east, north)."""
    spec = PROVIDERS[provider]
    zoom = min(zoom or choose_zoom(bbox, max_tiles, spec.max_zoom), spec.max_zoom)

    x0, y0, x1, y1 = _tile_range(bbox, zoom)
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    if cols * rows > max_tiles * 4:
        raise ValueError(f"area too large: {cols * rows} tiles at zoom {zoom}")

    canvas = np.zeros((rows * TILE_PX, cols * TILE_PX, 3), np.uint8)
    canvas[:] = (18, 14, 10)                      # void-ish, so gaps read as "no data"

    jobs = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    ok = 0

    def load(job: tuple[int, int]):
        x, y = job
        return job, fetch_tile(provider, zoom, x, y)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (x, y), tile in pool.map(load, jobs):
            if tile is None:
                continue
            if tile.shape[0] != TILE_PX or tile.shape[1] != TILE_PX:
                tile = cv2.resize(tile, (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA)
            cy, cx = (y - y0) * TILE_PX, (x - x0) * TILE_PX
            canvas[cy:cy + TILE_PX, cx:cx + TILE_PX] = tile
            ok += 1

    origin = (x0 * TILE_PX, y0 * TILE_PX)
    actual_bbox = (
        tile_to_lonlat(x0, y0, zoom)[0], tile_to_lonlat(x1 + 1, y1 + 1, zoom)[1],
        tile_to_lonlat(x1 + 1, y1 + 1, zoom)[0], tile_to_lonlat(x0, y0, zoom)[1],
    )

    if crop_to_bbox:
        west, south, east, north = bbox
        left, top = lonlat_to_tile(west, north, zoom)
        right, bottom = lonlat_to_tile(east, south, zoom)
        px0 = int(round(left * TILE_PX - origin[0]))
        py0 = int(round(top * TILE_PX - origin[1]))
        px1 = int(round(right * TILE_PX - origin[0]))
        py1 = int(round(bottom * TILE_PX - origin[1]))
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(canvas.shape[1], px1), min(canvas.shape[0], py1)
        if px1 - px0 >= 32 and py1 - py0 >= 32:
            canvas = canvas[py0:py1, px0:px1]
            origin = (origin[0] + px0, origin[1] + py0)
            actual_bbox = bbox

    lat_mid = (actual_bbox[1] + actual_bbox[3]) / 2
    return Mosaic(
        image=np.ascontiguousarray(canvas), bbox=actual_bbox, zoom=zoom,
        gsd_m=ground_resolution(lat_mid, zoom), provider=provider,
        attribution=spec.attribution, origin_px=origin,
        tiles_ok=ok, tiles_total=len(jobs),
    )


def cache_stats() -> dict[str, Any]:
    if not CACHE_DIR.exists():
        return {"tiles": 0, "bytes": 0, "providers": {}}
    per_provider: dict[str, int] = {}
    total = count = 0
    for path in CACHE_DIR.glob("*/*/*/*"):
        if path.is_file():
            size = path.stat().st_size
            total += size
            count += 1
            per_provider[path.parts[-4]] = per_provider.get(path.parts[-4], 0) + 1
    return {"tiles": count, "bytes": total, "providers": per_provider}


def iter_providers() -> Iterator[dict[str, Any]]:
    for p in PROVIDERS.values():
        yield {"key": p.key, "label": p.label, "kind": p.kind,
               "max_zoom": p.max_zoom, "attribution": p.attribution}
