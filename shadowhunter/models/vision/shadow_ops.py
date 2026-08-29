"""OpenCV shadow extraction and the image-processing metrics the reward uses.

Every metric here is *label-free*: it can be computed from the pixels alone,
which is what lets the RL agent explore without ground-truth heights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

from ...core.geo import SunGeometry


# --------------------------------------------------------------------------- #
# Shadow segmentation
# --------------------------------------------------------------------------- #
def _stretch(channel: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Percentile-stretch a float map into 0..255 uint8."""
    lo, hi = np.percentile(channel, (lo_pct, hi_pct))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((channel - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def shadow_mask(bgr: np.ndarray, method: str = "auto") -> np.ndarray:
    """Binary shadow mask (uint8, 0/255) for a BGR uint8 image.

    ``auto``   - darkness *and* blue-shift combined, then Otsu, then a
                 coverage sanity check. This is the one to use on real
                 imagery: a satellite shadow is both darker and cooler than
                 the sunlit ground beside it, and neither cue alone survives
                 the range of scenes you get from a live basemap.
    ``ratio``  - the classic HSV ratio map (H+1)/(V+1) + Otsu. Cheap; good on
                 clean scenes with a single illumination.
    ``lab``    - plain CIE-Lab luminance thresholding, for hazy scenes.

    The coverage guard matters more than the cue. Otsu happily returns 0.5%
    or 95% coverage on a scene that violates its bimodal assumption - a
    harbour, a sand flat, a tile that is half cloud - and a shadow mask that
    wrong poisons every metric downstream. When Otsu lands outside a
    plausible band the threshold is redrawn at a percentile instead.
    """
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    bgr = np.ascontiguousarray(bgr)

    if method == "lab":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lum = lab[:, :, 0]
        thr = max(1, int(np.mean(lum) - 0.6 * np.std(lum)))
        mask = (lum < thr).astype(np.uint8) * 255
    elif method == "ratio":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        ratio = _stretch(cv2.GaussianBlur((h + 1.0) / (v + 1.0), (5, 5), 0))
        _, mask = cv2.threshold(ratio, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lum = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0).astype(np.float32)
        b, g, r = (bgr[:, :, i].astype(np.float32) for i in range(3))
        # Blue-over-red: sky illumination survives in shadow, direct sun does not.
        coolness = (b + 1.0) / (r + 1.0)

        darkness = 255.0 - _stretch(lum).astype(np.float32)
        cool = _stretch(cv2.GaussianBlur(coolness, (5, 5), 0)).astype(np.float32)
        score = _stretch(0.68 * darkness + 0.32 * cool)

        _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coverage = float((mask > 0).mean())
        if not 0.03 <= coverage <= 0.62:
            target = 22.0 if coverage > 0.62 else 30.0
            cut = float(np.percentile(score, 100.0 - target))
            mask = (score >= cut).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def largest_blob(mask: np.ndarray, min_area: int = 24) -> tuple[np.ndarray, dict[str, Any]]:
    """Isolate the dominant connected component and describe it."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask), {"area": 0, "index": -1, "components": 0, "centroid": (0.0, 0.0)}

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < min_area:
        return np.zeros_like(mask), {"area": 0, "index": -1, "components": int(n - 1), "centroid": (0.0, 0.0)}

    blob = ((labels == idx).astype(np.uint8)) * 255
    info = {
        "area": area,
        "index": idx,
        "components": int((areas >= min_area).sum()),
        "centroid": (float(centroids[idx][0]), float(centroids[idx][1])),
        "bbox": (
            int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP]),
            int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT]),
        ),
        "all_areas": [int(a) for a in np.sort(areas)[::-1][:8]],
    }
    return blob, info


def blob_axis(blob: np.ndarray) -> tuple[float, float, float]:
    """PCA of a blob -> (axis_deg in [0,180), major_len_px, elongation)."""
    ys, xs = np.nonzero(blob)
    if xs.size < 8:
        return 0.0, 0.0, 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    major = vecs[:, 0]
    # Image y grows downward; negate to reason in compass-like terms.
    angle = math.degrees(math.atan2(-major[1], major[0])) % 180.0
    major_len = float(4.0 * math.sqrt(max(vals[0], 1e-6)))
    elong = float(math.sqrt(max(vals[0], 1e-6) / max(vals[1], 1e-6)))
    return angle, major_len, elong


def shadow_extent_along(blob: np.ndarray, sun: SunGeometry) -> float:
    """Projected shadow length in pixels, measured along the solar azimuth."""
    ys, xs = np.nonzero(blob)
    if xs.size < 8:
        return 0.0
    dx, dy = sun.shadow_vector
    proj = xs * dx + ys * dy
    lo, hi = np.percentile(proj, 2), np.percentile(proj, 98)
    return float(max(0.0, hi - lo))


# --------------------------------------------------------------------------- #
# Label-free metrics
# --------------------------------------------------------------------------- #
@dataclass
class CropMetrics:
    """Everything the reward function is allowed to know about a crop."""
    shadow_ratio: float = 0.0      # fraction of the crop that is shadow
    contrast: float = 0.0          # sunlit vs shadow luminance separation, 0..1
    isolation: float = 0.0         # dominant blob / total shadow, 0..1
    components: int = 0            # how many separate shadow blobs
    entropy: float = 0.0           # Shannon entropy of the grey histogram, 0..1
    edge_coherence: float = 0.0    # edges that sit on the shadow boundary, 0..1
    axis_alignment: float = 0.0    # blob axis vs solar azimuth, 0..1
    elongation: float = 0.0        # major/minor axis ratio, clipped
    truncation: float = 0.0        # blob pixels touching the crop border, 0..1
    occlusion: float = 0.0         # contamination by neighbouring shadows, 0..1
    shadow_len_px: float = 0.0     # projected length along the sun azimuth

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(64))


def _border_fraction(blob: np.ndarray) -> float:
    if blob.max() == 0:
        return 0.0
    total = int((blob > 0).sum())
    edge = int((blob[0, :] > 0).sum() + (blob[-1, :] > 0).sum()
               + (blob[:, 0] > 0).sum() + (blob[:, -1] > 0).sum())
    perimeter = 2 * (blob.shape[0] + blob.shape[1])
    return float(min(1.0, (edge / max(perimeter, 1)) * (perimeter / max(math.sqrt(total) * 4, 1)) * 0.25 + edge / max(perimeter, 1)))


def analyse_crop(crop_bgr: np.ndarray, sun: SunGeometry, mask: np.ndarray | None = None) -> CropMetrics:
    """Compute the full metric vector for one candidate zone."""
    m = CropMetrics()
    if crop_bgr.size == 0:
        return m

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mask = shadow_mask(crop_bgr) if mask is None else mask
    total_px = float(gray.size)
    shadow_px = float((mask > 0).sum())

    m.shadow_ratio = shadow_px / total_px
    m.entropy = _entropy(gray)

    if shadow_px < 16 or shadow_px > 0.92 * total_px:
        # Degenerate: no shadow at all, or the crop is entirely in shade.
        return m

    lit = gray[mask == 0].astype(np.float32)
    shd = gray[mask > 0].astype(np.float32)
    sep = (lit.mean() - shd.mean()) / 255.0
    spread = (lit.std() + shd.std()) / 255.0 + 1e-3
    m.contrast = float(np.clip(sep / (sep + spread), 0.0, 1.0))

    blob, info = largest_blob(mask)
    if info["area"] == 0:
        return m

    m.isolation = float(info["area"] / max(shadow_px, 1.0))
    m.components = int(info["components"])

    axis_deg, major_len, elong = blob_axis(blob)
    delta = abs(axis_deg - sun.axis_deg)
    delta = min(delta, 180.0 - delta)                       # fold to [0, 90]
    m.axis_alignment = float(math.cos(math.radians(delta)) ** 2)
    m.elongation = float(np.clip(elong / 6.0, 0.0, 1.0))
    m.shadow_len_px = shadow_extent_along(blob, sun)

    # Edge coherence: Canny edges that coincide with the shadow boundary.
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
    boundary = cv2.morphologyEx(blob, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    on_boundary = float(((edges > 0) & (boundary > 0)).sum())
    m.edge_coherence = float(np.clip(on_boundary / max((edges > 0).sum(), 1.0), 0.0, 1.0))

    m.truncation = float(np.clip(_border_fraction(blob), 0.0, 1.0))

    # Occlusion: everything that is shadow but does *not* belong to the
    # dominant blob is, by construction, another building's shadow leaking in.
    other = shadow_px - info["area"]
    m.occlusion = float(np.clip(other / max(shadow_px, 1.0), 0.0, 1.0))
    return m


# --------------------------------------------------------------------------- #
# Visual overlays (used by every UI)
# --------------------------------------------------------------------------- #
def overlay_shadow(bgr: np.ndarray, mask: np.ndarray, colour_bgr=(228, 211, 63), alpha: float = 0.42) -> np.ndarray:
    """Tint the shadow mask with the palette's cyan, keeping texture visible."""
    out = bgr.copy()
    tint = np.zeros_like(out)
    tint[:] = colour_bgr
    sel = mask > 0
    out[sel] = cv2.addWeighted(out, 1 - alpha, tint, alpha, 0)[sel]
    return out


def draw_zone(bgr: np.ndarray, box: tuple[int, int, int, int],
              colour_bgr=(32, 176, 255), thickness: int = 2, corner: int = 12) -> np.ndarray:
    """Reticle-style bracket instead of a plain rectangle - reads as an instrument."""
    out = bgr.copy()
    x, y, w, h = box
    x2, y2 = x + w, y + h
    for (cx, cy, sx, sy) in ((x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(out, (cx, cy), (cx + sx * corner, cy), colour_bgr, thickness, cv2.LINE_AA)
        cv2.line(out, (cx, cy), (cx, cy + sy * corner), colour_bgr, thickness, cv2.LINE_AA)
    faint = tuple(int(v * 0.45) for v in colour_bgr)
    cv2.rectangle(out, (x, y), (x2, y2), faint, 1, cv2.LINE_AA)
    return out
