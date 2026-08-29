"""Structure detection on overhead imagery - the green boxes.

Before the RL agent can hunt a shadow, something has to say *which* buildings
are in the frame. This does it without a trained detector and without labels,
by exploiting the same physics the rest of the project runs on:

    a building is a bright, textured, compact roof
    that has a shadow attached to it, on the anti-solar side.

That second clause is what separates a roof from a car park, a sandbank or a
white truck - and it costs nothing extra, because the shadow mask is already
computed. Every candidate therefore arrives with a physically meaningful
confidence rather than a softmax.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

from ...core.geo import SunGeometry, height_from_shadow
from .shadow_ops import shadow_mask


@dataclass
class Structure:
    """One detected building footprint candidate."""
    box: tuple[int, int, int, int]        # x, y, w, h in image pixels
    score: float                          # 0..1 overall confidence
    roof_score: float                     # brightness/texture/compactness
    shadow_support: float                 # fraction of the anti-solar strip in shadow
    area_px: int
    width_m: float
    height_m: float                        # footprint extent, NOT building height
    shadow_len_px: float
    quick_height_m: float                  # h = L*tan(theta) straight off the strip

    @property
    def center(self) -> tuple[int, int]:
        return self.box[0] + self.box[2] // 2, self.box[1] + self.box[3] // 2

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["box"] = list(self.box)
        d["center"] = list(self.center)
        for key in ("score", "roof_score", "shadow_support", "width_m", "height_m",
                    "shadow_len_px", "quick_height_m"):
            d[key] = round(float(d[key]), 3)
        return d


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def water_mask(bgr: np.ndarray) -> np.ndarray:
    """Large, dark, *smooth* regions - sea, river, pool.

    Water is the single worst false positive for shadow work: it is dark and
    it is huge, so it dominates any contrast-based reward. It is also almost
    featureless, and that is how we tell it apart from a building's shadow,
    which is full of the texture bleeding through from the ground beneath.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # high-frequency energy: water has almost none
    texture = cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F))
    texture = cv2.blur(texture, (15, 15))

    dark = gray < max(60, int(np.percentile(gray, 35)))
    smooth = texture < max(3.0, float(np.percentile(texture, 25)))
    bluish = hsv[:, :, 0].astype(np.int16)
    watery = ((bluish > 70) & (bluish < 110)) | (hsv[:, :, 1] > 60)

    mask = ((dark & smooth) | (smooth & watery & dark)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # only keep components big enough to actually be a body of water
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    min_area = max(900, int(0.004 * gray.size))
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def roof_mask(bgr: np.ndarray, shadows: np.ndarray, water: np.ndarray,
              min_px: int = 10) -> np.ndarray:
    """Bright, textured, non-shadow, non-water surfaces."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0]

    # Flatten the illumination first, then threshold the flattened field.
    #
    # The tempting version - "brighter than a small local blur" - is an edge
    # detector in disguise: the middle of a large uniform roof is not brighter
    # than its own neighbourhood, so a tower comes back as a hollow ring of
    # speckle and then shatters into dozens of 20-pixel fragments. The blur
    # radius therefore has to be far *wider* than any building, so it removes
    # the vignette and the haze gradient and nothing else.
    sigma = max(40.0, min_px * 6.0)
    flat = cv2.subtract(lum, cv2.GaussianBlur(lum, (0, 0), sigma))
    flat = cv2.add(flat, int(np.mean(lum)))

    # Threshold measured only over the candidate (lit, dry) pixels - a
    # percentile of the whole frame is meaningless when 40% of it is water.
    candidate = (shadows == 0) & (water == 0)
    reference = flat[candidate] if candidate.sum() > 256 else flat
    floor = float(np.percentile(reference, 60))
    bright = ((flat >= floor) & candidate).astype(np.uint8) * 255

    # Close across gaps up to a building-scale kernel. Rooftop plant, lift
    # cores and dark parapets punch holes in an otherwise solid roof; without
    # this the watershed inherits the holes and reports one tower as eight.
    span = max(5, min_px | 1)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)),
                              iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                              iterations=1)
    return bright


def split_touching(mask: np.ndarray, min_px: int,
                   guide: np.ndarray | None = None) -> np.ndarray:
    """Watershed a merged roof blob back into individual buildings.

    In a dense block every roof touches its neighbour through a shared
    parapet or a one-pixel bridge of bright render, and plain connected
    components returns the whole district as a single 200 000-pixel region -
    which then fails the size filter and the block detects as nothing at all.
    Distance-transform maxima give one seed per building; watershed draws the
    boundary between them.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() < min_px * min_px:
        return cv2.connectedComponents(binary, 8)[1]

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    seed_radius = max(5, int(min_px * 2.2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_radius * 2 + 1,) * 2)
    peaks = ((dist >= cv2.dilate(dist, kernel) - 1e-3) & (dist > min_px * 0.5))
    peaks = peaks.astype(np.uint8)

    n_seeds, seeds = cv2.connectedComponents(peaks, 8)
    if n_seeds <= 2:
        return cv2.connectedComponents(binary, 8)[1]

    # 0 = unknown (watershed floods here), 1 = background, 2.. = one per seed.
    # Leaving the un-seeded roof pixels at 0 is the whole point; mark them as
    # background and watershed has nothing left to grow into.
    markers = np.zeros(binary.shape, np.int32)
    markers[binary == 0] = 1
    seeded = seeds > 0
    markers[seeded] = seeds[seeded] + 1

    # Flood along the *inverted* distance transform: basins sit on the seeds,
    # ridges fall on the narrow necks between touching roofs.
    topo = cv2.normalize(dist.max() - dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if guide is not None and guide.shape[:2] == binary.shape:
        topo = cv2.addWeighted(topo, 0.6,
                               cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY), 0.4, 0)
    cv2.watershed(cv2.cvtColor(topo, cv2.COLOR_GRAY2BGR), markers)

    markers[markers <= 1] = 0                      # background + ridge pixels
    markers[binary == 0] = 0
    return markers


# --------------------------------------------------------------------------- #
# Reading the sun off the image itself
# --------------------------------------------------------------------------- #
def estimate_shadow_bearing(bgr: np.ndarray, shadows: np.ndarray | None = None,
                            water: np.ndarray | None = None,
                            roofs: np.ndarray | None = None,
                            step_deg: int = 5,
                            probes_px: tuple[int, ...] = (3, 6, 10, 16)) -> dict[str, Any]:
    """Recover the direction shadows fall, from the pixels alone.

    A basemap mosaic carries no acquisition timestamp, so the honest sun angle
    is not "whatever the operator typed into a date picker" - it is whatever
    the shadows in this image actually say. Shift the roof mask along each
    candidate bearing and measure how much shadow it lands on; the direction
    that maximises the hit rate is where the sun casts.

    Two details make it work on real tiles. The source is the *roof* mask, not
    a brightness percentile, so sand and surf do not vote. And the response is
    measured against the scene's own shadow fraction, so a tile that is 40%
    shadow does not score every direction equally well.
    """
    shadows = shadow_mask(bgr) if shadows is None else shadows
    water = water_mask(bgr) if water is None else water
    shade = (cv2.bitwise_and(shadows, cv2.bitwise_not(water)) > 0).astype(np.float32)
    if roofs is None:
        roofs = roof_mask(bgr, shadows, water)

    # Sample from the roof *rim*, not the whole roof: the discriminating
    # evidence lives in the few pixels just outside the footprint, where one
    # side is lit ground and the other is the cast shadow.
    rim = cv2.morphologyEx(
        (roofs > 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    source = (rim > 0).astype(np.float32)
    if source.sum() < 64:
        source = (roofs > 0).astype(np.float32)

    baseline = float(shade.mean())
    if source.sum() < 64 or shade.sum() < 64 or baseline > 0.9:
        return {"shadow_bearing_deg": 0.0, "sun_azimuth_deg": 180.0,
                "confidence": 0.0, "baseline": round(baseline, 4), "responses": {}}

    h, w = shade.shape[:2]
    responses: dict[int, float] = {}
    for deg in range(0, 360, step_deg):
        rad = math.radians(deg)
        dx, dy = math.sin(rad), -math.cos(rad)       # compass bearing in image space
        hits = []
        for probe in probes_px:
            matrix = np.float32([[1, 0, dx * probe], [0, 1, dy * probe]])
            shifted = cv2.warpAffine(source, matrix, (w, h),
                                     flags=cv2.INTER_NEAREST, borderValue=0)
            total = float(shifted.sum())
            if total > 32:
                hits.append(float((shifted * shade).sum()) / total)
        responses[deg] = float(np.mean(hits)) if hits else 0.0

    # Difference each direction against its opposite. A tile that is 40%
    # shadow scores every direction about 0.4; the *asymmetry* between a
    # direction and its antipode is pure signal, and it cancels the scene's
    # shadow fraction exactly.
    signed = {deg: responses[deg] - responses[(deg + 180) % 360] for deg in responses}
    values = np.array(list(signed.values()), np.float32)
    best = max(signed, key=signed.get)
    peak = float(values.max())
    spread = float(values.std())
    confidence = float(np.clip(peak / max(0.06, 3.0 * spread) * 0.5
                               + min(peak / 0.18, 1.0) * 0.5, 0.0, 1.0)) if peak > 0 else 0.0

    left = signed[(best - step_deg) % 360]
    right = signed[(best + step_deg) % 360]
    denom = left - 2 * peak + right
    offset = 0.5 * (left - right) / denom * step_deg if abs(denom) > 1e-9 else 0.0
    bearing = (best + float(np.clip(offset, -step_deg, step_deg))) % 360.0

    return {
        "shadow_bearing_deg": round(bearing, 1),
        "sun_azimuth_deg": round((bearing + 180.0) % 360.0, 1),
        "confidence": round(confidence, 3),
        "baseline": round(baseline, 4),
        "peak": round(peak, 4),
        "responses": {int(k): round(v, 4) for k, v in signed.items()},
    }


# --------------------------------------------------------------------------- #
# Shadow support - the physics test
# --------------------------------------------------------------------------- #
def _shadow_strip(shadows: np.ndarray, cx: float, cy: float, sun: SunGeometry,
                  reach_px: float, half_width_px: float,
                  skip_px: float = 0.0) -> tuple[float, float]:
    """Walk from a roof centre along the shadow direction.

    ``skip_px`` steps over the roof itself before sampling starts. Without it
    the walk begins on the bright roof, trips the "shadow has ended" guard in
    its first four steps, and every building reports a shadow length of zero
    while still showing high support further out.

    Returns ``(support, length_px)`` - how much of the strip is shadow, and how
    far the shadow continues before it stops.
    """
    dx, dy = sun.shadow_vector
    H, W = shadows.shape[:2]
    px, py = -dy, dx                                    # perpendicular offset

    start = max(1, int(skip_px))
    steps = start + max(6, int(reach_px))
    hits: list[bool] = []
    for i in range(start, steps + 1):
        samples = 0
        shadowed = 0
        for w in np.linspace(-half_width_px, half_width_px, 5):
            sx = int(round(cx + dx * i + px * w))
            sy = int(round(cy + dy * i + py * w))
            if 0 <= sx < W and 0 <= sy < H:
                samples += 1
                shadowed += 1 if shadows[sy, sx] > 0 else 0
        hits.append(samples > 0 and shadowed / samples >= 0.5)

    if not hits:
        return 0.0, 0.0
    support = sum(hits) / len(hits)

    # Real shadows have holes - a lit courtyard, a bright car, a gap between
    # two wings - so the run tolerates a gap proportional to the reach before
    # deciding the shadow has genuinely ended.
    tolerance = max(4, int(len(hits) * 0.08))
    length = 0
    gap = 0
    for i, hit in enumerate(hits):
        if hit:
            length = i + 1
            gap = 0
        else:
            gap += 1
            if gap > tolerance:
                break
    return float(support), float(length)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_structures(bgr: np.ndarray, sun: SunGeometry, *,
                      min_size_m: float = 10.0, max_size_m: float = 260.0,
                      max_results: int = 80, min_score: float = 0.18,
                      work_width: int = 1024, auto_sun: bool = False) -> dict[str, Any]:
    """Find building-like structures and score each by its shadow evidence.

    The azimuth is always *estimated* from the image and reported under
    ``sun_estimate``, but it only overrides the supplied value when
    ``auto_sun`` is on and the estimate is confident. A sun angle computed
    from latitude, longitude and a timestamp is rigorous; one read off the
    pixels is evidence with an error bar, and in a scene where shadows overlap
    everything it can be flatly wrong. The operator sees both and decides.
    Elevation is never guessed this way - it cannot be recovered without a
    known height - so it always stays as supplied.
    """
    H0, W0 = bgr.shape[:2]
    scale = min(1.0, work_width / max(W0, 1))
    work = cv2.resize(bgr, (int(W0 * scale), int(H0 * scale)),
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    gsd = sun.gsd_m / max(scale, 1e-6)                  # metres per *working* pixel

    shadows = shadow_mask(work)
    water = water_mask(work)
    shadows = cv2.bitwise_and(shadows, cv2.bitwise_not(water))

    min_px = max(6, int(min_size_m / max(gsd, 1e-6)))
    max_px = int(max_size_m / max(gsd, 1e-6))
    min_area = int(min_px ** 2 * 0.9)
    max_area = int(max_px ** 2)

    roofs = roof_mask(work, shadows, water, min_px)

    # Read the illumination off the image when asked to. A live basemap tile
    # has no acquisition time, so a sun angle derived from a date picker is a
    # guess; the shadows themselves are evidence.
    bearing_info = estimate_shadow_bearing(work, shadows, water, roofs)
    bearing_info.pop("responses", None)
    bearing_info["applied"] = bool(auto_sun and bearing_info["confidence"] >= 0.35)
    if bearing_info["applied"]:
        sun = SunGeometry(azimuth_deg=bearing_info["sun_azimuth_deg"],
                          elevation_deg=sun.elevation_deg, gsd_m=sun.gsd_m)

    # A tight AOI (one monument in frame) is shattered by watershed; keep blobs.
    if min(work.shape[0], work.shape[1]) < 320:
        labels = cv2.connectedComponents((roofs > 0).astype(np.uint8), 8)[1]
    else:
        labels = split_touching(roofs, min_px, work)
    label_ids = [int(v) for v in np.unique(labels) if v > 0]
    night_or_poor = float(sun.elevation_deg) < 8.0

    reach = float(np.clip(180.0 / max(gsd, 1e-6), 20, 260))
    candidates: list[Structure] = []

    for label_id in label_ids:
        component = (labels == label_id)
        area = int(component.sum())
        if area < min_area or area > max_area:
            continue
        ys, xs = np.nonzero(component)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
        if w < min_px * 0.7 or h < min_px * 0.7:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 9.0:                                 # a road, a wall, a wake
            continue
        fill = area / float(max(w * h, 1))
        pts = np.column_stack((xs, ys)).astype(np.int32)
        hull = cv2.convexHull(pts)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        hull_fill = area / hull_area
        # Arches (Azadi) are holey in bbox fill; hull fill still reads as a monument.
        if fill < 0.10 and hull_fill < 0.20:
            continue

        cx, cy = float(xs.mean()), float(ys.mean())
        # step over the roof: its own extent projected on the shadow direction
        dxs, dys = sun.shadow_vector
        skip = float(np.percentile((xs - cx) * dxs + (ys - cy) * dys, 96)) + 2.0
        support, length = _shadow_strip(shadows, cx, cy, sun, reach,
                                        max(3.0, min(w, h) * 0.3), skip)

        # roof plausibility: compactness x squareness x size sanity
        compactness = float(np.clip(fill, 0, 1))
        squareness = float(np.clip(1.0 - (aspect - 1.0) / 6.0, 0, 1))
        size_fit = float(np.exp(-((math.sqrt(area) * gsd - 34.0) / 46.0) ** 2))
        roof_score = float(np.clip(0.45 * compactness + 0.3 * squareness + 0.25 * size_fit, 0, 1))

        # A roof with no shadow on the sun-opposite side is probably not a
        # building - but never zero it out, because flat low sheds are real.
        # Night / near-horizon ephemeris cannot vote on shadow support.
        if night_or_poor:
            score = float(np.clip(0.82 * roof_score + 0.18 * support, 0, 1))
            floor = min(min_score, 0.12)
        else:
            score = float(np.clip(0.42 * roof_score + 0.58 * support, 0, 1))
            floor = min_score
        if score < floor:
            continue

        inv = 1.0 / max(scale, 1e-6)
        box = (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
        candidates.append(Structure(
            box=box, score=score, roof_score=roof_score, shadow_support=support,
            area_px=int(area * inv * inv),
            width_m=w * gsd, height_m=h * gsd,
            shadow_len_px=length * inv,
            quick_height_m=height_from_shadow(length, SunGeometry(
                sun.azimuth_deg, sun.elevation_deg, gsd)),
        ))

    candidates = _non_max_suppress(candidates, iou_threshold=0.3)
    candidates.sort(key=lambda s: s.score, reverse=True)
    candidates = candidates[:max_results]

    return {
        "structures": [s.to_dict() for s in candidates],
        "count": len(candidates),
        "with_shadow": sum(1 for s in candidates if s.shadow_support > 0.35),
        "mean_score": round(float(np.mean([s.score for s in candidates])), 3) if candidates else 0.0,
        "shadow_coverage": round(float((shadows > 0).mean()), 4),
        "water_coverage": round(float((water > 0).mean()), 4),
        "gsd_m": round(sun.gsd_m, 4),
        "sun": {"azimuth_deg": round(sun.azimuth_deg, 1),
                "elevation_deg": round(sun.elevation_deg, 1),
                "gsd_m": round(sun.gsd_m, 4),
                "shadow_bearing_deg": round(sun.shadow_bearing_deg, 1)},
        "sun_estimate": bearing_info,
        "_masks": {"shadow": shadows, "water": water, "roof": roofs, "scale": scale},
    }


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _non_max_suppress(items: list[Structure], iou_threshold: float = 0.3) -> list[Structure]:
    kept: list[Structure] = []
    for cand in sorted(items, key=lambda s: s.score, reverse=True):
        if all(_iou(cand.box, k.box) < iou_threshold for k in kept):
            kept.append(cand)
    return kept


def compact_footprint(bgr: np.ndarray, sun: SunGeometry,
                      min_size_m: float = 10.0, *,
                      allow_large: bool = False) -> dict[str, Any] | None:
    """Frame the central compact monument instead of the whole AOI.

    Whole-frame seeds produce a beige slab covering the plaza. A white marble
    tower (Azadi) is a bright, compact blob near the centre — that box becomes
    the 3D footprint.
    """
    img = np.ascontiguousarray(bgr)
    if img.ndim == 2:
        gray = img
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img[:, :, :3]
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    frame = float(max(h * w, 1))
    gsd = max(float(sun.gsd_m), 1e-6)
    min_px = max(8, int(round(min_size_m / gsd)))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, k)
    hi = int(max(12, np.percentile(tophat, 88)))
    _, mask = cv2.threshold(tophat, max(hi - 1, 0), 255, cv2.THRESH_BINARY)
    if int((mask > 0).sum()) < min_px * min_px:
        _, mask = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                            iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    yy, xx = np.mgrid[0:h, 0:w]
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
    dist = np.sqrt(((xx - cx0) / max(w, 1)) ** 2 + ((yy - cy0) / max(h, 1)) ** 2)
    dist_cut = 0.92 if (allow_large or min(h, w) < 360) else 0.62
    mask[dist > dist_cut] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, Any] | None = None
    max_frac = 0.94 if allow_large else 0.62
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)
        if area < min_px * min_px or area / frame >= max_frac:
            continue
        if min(bw, bh) < min_px * 0.55:
            continue
        contour_area = float(cv2.contourArea(cnt)) or area
        peri = float(cv2.arcLength(cnt, True)) or 1.0
        compactness = float(np.clip(4.0 * math.pi * contour_area / (peri * peri), 0.05, 1.0))
        ccx, ccy = x + bw / 2.0, y + bh / 2.0
        centrality = 1.0 - min(1.0, math.hypot(
            (ccx - cx0) / max(w, 1), (ccy - cy0) / max(h, 1)) / 0.62)
        # Plaza ovals are round (high compactness). Arches / inverted-Y are not.
        irregular = float(np.clip(1.20 - compactness, 0.18, 1.0))
        score = contour_area * (0.3 + 0.7 * centrality) * irregular
        if best is None or score > best[0]:
            best = (score, cnt)
    if best is None:
        return None

    _, cnt = best
    x, y, bw, bh = cv2.boundingRect(cnt)
    pad_x = max(2, int(bw * 0.08))
    pad_y = max(2, int(bh * 0.08))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)
    box = (x0, y0, max(4, x1 - x0), max(4, y1 - y0))
    item = Structure(
        box=box,
        score=0.62,
        roof_score=0.7,
        shadow_support=0.0,
        area_px=int(box[2] * box[3]),
        width_m=box[2] * gsd,
        height_m=box[3] * gsd,
        shadow_len_px=0.0,
        quick_height_m=0.0,
    ).to_dict()
    item["seeded"] = False
    item["compact"] = True
    item["index"] = 0
    peri = float(cv2.arcLength(cnt, True)) or 1.0
    approx = cv2.approxPolyDP(cnt, max(1.4, 0.016 * peri), True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        pts = cnt.reshape(-1, 2)
    item["outline_px"] = [[int(p[0]), int(p[1])] for p in pts]
    return item


def seed_frame_structure(bgr: np.ndarray, sun: SunGeometry) -> dict[str, Any]:
    """Treat the whole frame as one footprint — the operator already boxed it."""
    H, W = int(bgr.shape[0]), int(bgr.shape[1])
    pad = max(4, int(0.05 * min(H, W)))
    x, y = pad, pad
    w, h = max(8, W - 2 * pad), max(8, H - 2 * pad)
    gsd = max(float(sun.gsd_m), 1e-6)
    cx, cy = x + w / 2.0, y + h / 2.0
    item = Structure(
        box=(x, y, w, h),
        score=0.35,
        roof_score=0.55,
        shadow_support=0.0,
        area_px=int(w * h),
        width_m=w * gsd,
        height_m=h * gsd,
        shadow_len_px=0.0,
        quick_height_m=0.0,
    ).to_dict()
    item["seeded"] = True
    item["index"] = 0
    item["center"] = [int(cx), int(cy)]
    return item


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #
def draw_structures(bgr: np.ndarray, structures: list[dict[str, Any]], *,
                    colour_bgr: tuple[int, int, int] = (140, 227, 105),
                    label: bool = True, selected: int | None = None,
                    thickness: int = 2) -> np.ndarray:
    """Green boxes, numbered, with the confidence-scaled corner brackets."""
    out = bgr.copy()
    for i, s in enumerate(structures):
        x, y, w, h = s["box"]
        hot = (selected == i)
        colour = (32, 176, 255) if hot else colour_bgr
        arm = max(6, int(min(w, h) * 0.28))

        cv2.rectangle(out, (x, y), (x + w, y + h),
                      tuple(int(c * 0.45) for c in colour), 1, cv2.LINE_AA)
        outline = s.get("outline_px")
        if isinstance(outline, list) and len(outline) >= 3:
            pts = np.array(outline, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, colour, max(1, thickness), cv2.LINE_AA)
        for (cx, cy, sx, sy) in ((x, y, 1, 1), (x + w, y, -1, 1),
                                 (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(out, (cx, cy), (cx + sx * arm, cy), colour, thickness, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + sy * arm), colour, thickness, cv2.LINE_AA)

        # Numbering a 12-pixel box with a 14-pixel chip hides the thing it
        # labels, so only boxes with room to spare get one.
        if label and min(w, h) >= 22:
            tag = f"{i + 1}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.36, 1)
            cv2.rectangle(out, (x, y - th - 6), (x + tw + 7, y - 1), colour, -1)
            cv2.putText(out, tag, (x + 3, y - 4), cv2.FONT_HERSHEY_DUPLEX, 0.36,
                        (8, 10, 12), 1, cv2.LINE_AA)
    return out
