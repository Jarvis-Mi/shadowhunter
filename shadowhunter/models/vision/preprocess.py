"""Scene I/O: GeoTIFF via rasterio, ordinary rasters via OpenCV, and a
physically-consistent synthetic city so the pipeline runs with zero downloads.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...core.geo import SunGeometry, shadow_from_height

try:  # rasterio is heavy and optional at import time
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except Exception:  # pragma: no cover
    HAS_RASTERIO = False


@dataclass
class Building:
    """Ground truth for one structure (synthetic scenes, or a labelled dataset)."""
    x: int
    y: int
    w: int
    h: int
    height_m: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scene:
    """A working tile plus everything needed to reason about its shadows."""
    image: np.ndarray                 # BGR uint8
    sun: SunGeometry
    name: str = "scene"
    buildings: list[Building] | None = None
    source: str = "synthetic"
    crs: str | None = None
    transform: Any | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "width": self.image.shape[1],
            "height": self.image.shape[0],
            "crs": self.crs,
            "sun": {
                "azimuth_deg": self.sun.azimuth_deg,
                "elevation_deg": self.sun.elevation_deg,
                "gsd_m": self.sun.gsd_m,
            },
            "buildings": len(self.buildings or []),
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_scene(path: str | Path, sun: SunGeometry | None = None,
               max_side: int = 1536) -> Scene:
    """Load a GeoTIFF (rasterio) or a plain image (OpenCV) into a Scene."""
    path = Path(path)
    sun = sun or SunGeometry()

    if path.suffix.lower() in {".tif", ".tiff"} and HAS_RASTERIO:
        with rasterio.open(path) as ds:
            count = min(3, ds.count)
            scale = min(1.0, max_side / max(ds.width, ds.height))
            out_w, out_h = int(ds.width * scale), int(ds.height * scale)
            arr = ds.read(
                indexes=list(range(1, count + 1)),
                out_shape=(count, out_h, out_w),
                resampling=Resampling.bilinear,
            )
            img = _to_uint8_bgr(arr)
            tags = ds.tags()
            sun = SunGeometry(
                azimuth_deg=float(tags.get("SUN_AZIMUTH", sun.azimuth_deg)),
                elevation_deg=float(tags.get("SUN_ELEVATION", sun.elevation_deg)),
                gsd_m=abs(ds.transform.a) / max(scale, 1e-9) if ds.transform else sun.gsd_m,
            )
            return Scene(image=img, sun=sun, name=path.stem, source="geotiff",
                         crs=str(ds.crs) if ds.crs else None, transform=ds.transform)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read scene: {path}")
    img = _fit(img, max_side)

    sidecar = path.with_suffix(".json")
    buildings = None
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        s = meta.get("sun", {})
        sun = SunGeometry(
            azimuth_deg=float(s.get("azimuth_deg", sun.azimuth_deg)),
            elevation_deg=float(s.get("elevation_deg", sun.elevation_deg)),
            gsd_m=float(s.get("gsd_m", sun.gsd_m)),
        )
        buildings = [Building(**b) for b in meta.get("buildings", [])] or None

    return Scene(image=img, sun=sun, name=path.stem, buildings=buildings, source="raster")


def _to_uint8_bgr(arr: np.ndarray) -> np.ndarray:
    """Percentile-stretch an arbitrary-dtype band stack into display BGR."""
    bands = []
    for b in arr:
        b = b.astype(np.float32)
        lo, hi = np.percentile(b, (2, 98))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        bands.append(np.clip((b - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8))
    while len(bands) < 3:
        bands.append(bands[-1])
    rgb = np.stack(bands[:3], axis=-1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _fit(img: np.ndarray, max_side: int) -> np.ndarray:
    s = max(img.shape[:2])
    if s <= max_side:
        return img
    k = max_side / s
    return cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Crops / observations
# --------------------------------------------------------------------------- #
def safe_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), np.uint8)
    return img[y0:y1, x0:x1]


def to_observation(crop_bgr: np.ndarray, mask: np.ndarray, size: int = 84) -> np.ndarray:
    """(4, size, size) uint8 tensor: RGB + shadow mask, channel-first for SB3."""
    rgb = cv2.cvtColor(cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([rgb.transpose(2, 0, 1), m[None, ...]], axis=0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Synthetic city - the zero-dependency dataset
# --------------------------------------------------------------------------- #
def synthesize_scene(size: int = 512, n_buildings: int = 14, sun: SunGeometry | None = None,
                     seed: int | None = None, density: float = 1.0, name: str = "synthetic") -> Scene:
    """Render an overhead urban tile whose shadows obey ``h = L*tan(theta)``.

    Buildings are drawn back-to-front along the shadow direction so that, in
    dense configurations, a taller neighbour's shadow genuinely falls across a
    shorter roof. That occlusion is the exact failure mode the RL agent exists
    to route around, so the simulator must produce it honestly.
    """
    rng = np.random.default_rng(seed)
    sun = sun or SunGeometry(
        azimuth_deg=float(rng.uniform(110, 210)),
        elevation_deg=float(rng.uniform(22, 55)),
        gsd_m=0.5,
    )
    dx, dy = sun.shadow_vector

    # --- ground: soil-toned noise, then a road grid ------------------------
    base = rng.normal(96, 9, (size, size, 3)).astype(np.float32)
    base[:, :, 0] *= 0.92   # B
    base[:, :, 2] *= 1.06   # R  -> warm dusty ground
    grain = cv2.GaussianBlur(rng.normal(0, 26, (size, size)).astype(np.float32), (0, 0), 7)
    base += grain[..., None]

    img = np.clip(base, 0, 255).astype(np.uint8)
    for pos in range(60, size, 150):
        cv2.line(img, (pos, 0), (pos, size), (78, 78, 82), int(rng.integers(9, 17)), cv2.LINE_AA)
        cv2.line(img, (0, pos), (size, pos), (78, 78, 82), int(rng.integers(9, 17)), cv2.LINE_AA)

    # --- buildings ---------------------------------------------------------
    n = max(1, int(n_buildings * density))
    specs: list[Building] = []
    margin = 24
    for _ in range(n):
        w = int(rng.integers(26, 74))
        h = int(rng.integers(26, 74))
        x = int(rng.integers(margin, max(margin + 1, size - w - margin)))
        y = int(rng.integers(margin, max(margin + 1, size - h - margin)))
        height_m = float(rng.uniform(6, 58))
        specs.append(Building(x, y, w, h, height_m))

    # Paint far-from-sun first so shadows layer in the physically right order.
    specs.sort(key=lambda b: -(b.center[0] * dx + b.center[1] * dy))

    shadow_layer = np.zeros((size, size), np.uint8)
    for b in specs:
        L = shadow_from_height(b.height_m, sun)
        ox, oy = int(round(dx * L)), int(round(dy * L))
        quad = np.array([
            [b.x, b.y], [b.x + b.w, b.y], [b.x + b.w, b.y + b.h], [b.x, b.y + b.h],
            [b.x + ox, b.y + oy], [b.x + b.w + ox, b.y + oy],
            [b.x + b.w + ox, b.y + b.h + oy], [b.x + ox, b.y + b.h + oy],
        ], np.int32)
        cv2.fillConvexPoly(shadow_layer, cv2.convexHull(quad), 255)

    shadow_soft = cv2.GaussianBlur(shadow_layer, (0, 0), 1.6).astype(np.float32) / 255.0
    img = (img.astype(np.float32) * (1.0 - 0.62 * shadow_soft[..., None])).astype(np.uint8)
    # Shadows are cooler, not just darker - that is what the HSV ratio picks up.
    img[:, :, 0] = np.clip(img[:, :, 0].astype(np.float32) + 26 * shadow_soft, 0, 255).astype(np.uint8)

    for b in specs:
        tone = int(rng.integers(150, 205))
        roof = (tone - int(rng.integers(0, 16)), tone, tone + int(rng.integers(0, 14)))
        cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), roof, -1)
        cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h),
                      tuple(int(v * 0.72) for v in roof), 1, cv2.LINE_AA)
        # roof clutter so the CNN cannot cheat on flat texture alone
        for _ in range(int(rng.integers(1, 4))):
            rx = int(rng.integers(b.x + 3, max(b.x + 4, b.x + b.w - 6)))
            ry = int(rng.integers(b.y + 3, max(b.y + 4, b.y + b.h - 6)))
            cv2.rectangle(img, (rx, ry), (rx + 5, ry + 5),
                          tuple(int(v * 0.85) for v in roof), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    noise = rng.normal(0, 3.4, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    specs.sort(key=lambda b: (b.y, b.x))
    return Scene(image=img, sun=sun, name=name, buildings=specs, source="synthetic")


def save_scene(scene: Scene, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{scene.name}.png"
    cv2.imwrite(str(img_path), scene.image)
    meta = scene.meta()
    meta["buildings"] = [b.to_dict() for b in (scene.buildings or [])]
    img_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return img_path


def crop_dataset(scene: Scene, crop: int = 96, jitter: int = 10,
                 seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build a (N,H,W,3) / (N,) supervised set for the CNN from a labelled scene.

    Each sample is centred on a building and extended along the shadow
    direction, so the crop contains the roof *and* the cast shadow.
    """
    from .shadow_ops import shadow_mask  # local import keeps module import cheap

    rng = np.random.default_rng(seed)
    dx, dy = scene.sun.shadow_vector
    xs, ys = [], []
    for b in scene.buildings or []:
        L = shadow_from_height(b.height_m, scene.sun)
        cx = b.center[0] + int(dx * L * 0.45) + int(rng.integers(-jitter, jitter + 1))
        cy = b.center[1] + int(dy * L * 0.45) + int(rng.integers(-jitter, jitter + 1))
        box = (cx - crop // 2, cy - crop // 2, crop, crop)
        patch = safe_crop(scene.image, box)
        if patch.shape[0] < crop // 2 or patch.shape[1] < crop // 2:
            continue
        patch = cv2.resize(patch, (crop, crop), interpolation=cv2.INTER_AREA)
        m = shadow_mask(patch)
        xs.append(np.dstack([patch, m]))
        ys.append(b.height_m)
    if not xs:
        return np.zeros((0, crop, crop, 4), np.uint8), np.zeros((0,), np.float32)
    return np.stack(xs).astype(np.uint8), np.asarray(ys, np.float32)


def shadow_mask_preview(bgr: np.ndarray) -> np.ndarray:
    """Convenience for the API: scene tinted with its own shadow mask."""
    from .shadow_ops import overlay_shadow, shadow_mask

    return overlay_shadow(bgr, shadow_mask(bgr))
