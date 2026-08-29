"""Raster inspection: colour spectra, histogram, shadows, and a pseudo-depth preview.

Nothing here is a sensor reading. The depth map is a relative 0..1 field
derived from luminance with shadows pulled toward the ground, so roofs read
near and shade reads far. Colour spectra (RGB, HSV, LAB, vegetation-ish
false colour, shadow-boosted luma) are summary stats only. Callers that
need JSON should run :func:`inspect_to_jsonable` (it drops ``_``-prefixed
arrays used as UI previews).
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .vision.shadow_ops import shadow_mask


def _bit_depth(dtype: np.dtype) -> int:
    if dtype in (np.uint16, np.int16) or dtype.itemsize == 2:
        return 16
    return 8


def _to_bgr8(bgr: np.ndarray) -> np.ndarray:
    """Normalise any 2-D / 3-D raster to contiguous BGR uint8 (alpha dropped)."""
    img = np.ascontiguousarray(bgr)
    plane = img if img.ndim == 2 else img[:, :, :3]

    if plane.dtype == np.uint8:
        u8 = plane
    elif np.issubdtype(plane.dtype, np.integer):
        info = np.iinfo(plane.dtype)
        scale = 255.0 / max(int(info.max), 1)
        u8 = np.clip(plane.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    else:
        f = plane.astype(np.float32)
        mx = float(np.nanmax(f)) if f.size else 1.0
        if mx <= 1.0 + 1e-5:
            u8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        elif mx <= 255.0 + 1e-2:
            u8 = np.clip(f, 0, 255).astype(np.uint8)
        else:
            u8 = np.clip(f * (255.0 / mx), 0, 255).astype(np.uint8)

    if u8.ndim == 2:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(u8)


def _hist32(channel: np.ndarray) -> list[int]:
    hist = cv2.calcHist([channel], [0], None, [32], [0, 256]).ravel()
    return [int(v) for v in hist]


def inspect_raster(bgr: np.ndarray) -> dict[str, Any]:
    """Analyse a BGR (or gray / BGRA) raster. Fast: numpy and OpenCV only."""
    if bgr is None or bgr.size == 0:
        raise ValueError("inspect_raster: empty image")

    src = np.ascontiguousarray(bgr)
    height, width = src.shape[:2]
    ndim = int(src.ndim)
    channels = 1 if ndim == 2 else int(src.shape[2])
    if channels not in (1, 3, 4):
        channels = min(max(channels, 1), 4)

    bgr8 = _to_bgr8(src)
    r, g, bch = bgr8[:, :, 2], bgr8[:, :, 1], bgr8[:, :, 0]

    rgb_mean = [float(r.mean()), float(g.mean()), float(bch.mean())]
    rgb_std = [float(r.std()), float(g.std()), float(bch.std())]
    rgb_min = [int(r.min()), int(g.min()), int(bch.min())]
    rgb_max = [int(r.max()), int(g.max()), int(bch.max())]

    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
    hsv_mean = [float(hsv[:, :, 0].mean()), float(hsv[:, :, 1].mean()),
                float(hsv[:, :, 2].mean())]

    lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]
    lab_mean = [float(lab_l.mean()), float(lab[:, :, 1].mean()),
                float(lab[:, :, 2].mean())]
    lab_u8 = np.ascontiguousarray(lab_l)

    lo = float(min(rgb_min))
    hi = float(max(rgb_max))
    # Per-pixel clip: a pixel counts if any of B,G,R is 0 / 255 (voids & highlights).
    at_0 = float((bgr8.min(axis=2) == 0).mean())
    at_255 = float((bgr8.max(axis=2) == 255).mean())

    mask = shadow_mask(bgr8)
    shadow_fraction = float((mask > 0).mean())
    in_shadow = (mask > 0).astype(np.float32)

    luma = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Bright roofs -> near (255). Shadows are ground, pulled toward far (0).
    # "Inverse luma" here means closeness is the inverse of darkness.
    depth = luma * (1.0 - 0.62 * in_shadow)
    depth = np.clip(depth, 0.0, 1.0)
    depth_u8 = (depth * 255.0 + 0.5).astype(np.uint8)

    # Vegetation-ish: (G-R)/(G+R) in [-1, 1], then stretch to 0-1.
    rf = r.astype(np.float32)
    gf = g.astype(np.float32)
    denom = gf + rf
    veg = np.zeros_like(denom, dtype=np.float32)
    np.divide(gf - rf, denom, out=veg, where=denom > 1e-6)
    false = np.clip((veg + 1.0) * 0.5, 0.0, 1.0)
    false_u8 = (false * 255.0 + 0.5).astype(np.uint8)

    shadow_boost = np.clip(luma * (1.0 - 0.78 * in_shadow), 0.0, 1.0)

    return {
        "width": int(width),
        "height": int(height),
        "ndim": ndim,
        "dtype": str(src.dtype),
        "channels": int(channels),
        "layout": "BGR",
        "bit_depth": _bit_depth(src.dtype),
        "bytes_per_pixel": int(src.dtype.itemsize * (1 if ndim == 2 else src.shape[2])),
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "rgb_min": rgb_min,
        "rgb_max": rgb_max,
        "hsv_mean": hsv_mean,
        "dynamic_range": (hi - lo) / 255.0,
        "clipping": {"at_0": at_0, "at_255": at_255},
        "histogram": {"r": _hist32(r), "g": _hist32(g), "b": _hist32(bch)},
        "shadow_fraction": shadow_fraction,
        "spectra": {
            "rgb": {
                "mean": rgb_mean,
                "note": "sRGB channel means (R, G, B) in 0-255",
            },
            "hsv": {
                "mean": hsv_mean,
                "note": "OpenCV HSV means (H 0-179, S 0-255, V 0-255)",
            },
            "lab": {
                "mean": lab_mean,
                "note": "OpenCV CIE LAB means; L is lightness (0-255)",
            },
            "false_color": {
                "mean": float(false.mean()),
                "note": "vegetation-ish (G-R)/(G+R) stretched to 0-1",
            },
            "shadow_boost": {
                "mean": float(shadow_boost.mean()),
                "note": "luma with shadows darkened further",
            },
        },
        "depth_proxy": {
            "method": "shadow_luma",
            "note": "not a true depth sensor; relative 0-1 from inverse luma with shadows boosted",
            "mean": float(depth.mean()),
            "std": float(depth.std()),
        },
        "_depth_u8": depth_u8,
        "_false_u8": false_u8,
        "_lab_u8": lab_u8,
        "_boost_u8": (shadow_boost * 255.0 + 0.5).astype(np.uint8),
        "_hsv_u8": hsv,
    }


def inspect_to_jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Copy *d*, drop ``_`` keys, round floats, promote numpy scalars."""

    def conv(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: conv(v) for k, v in obj.items() if not str(k).startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [conv(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return conv(obj.tolist())
        if isinstance(obj, (np.floating, float)):
            x = float(obj)
            if math.isnan(x) or math.isinf(x):
                return None
            return round(x, 6)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    return conv(dict(d))
