"""Extrude detected structures into a tiny 3D scene (glTF + Three.js viewer).

The HTML viewer reads ``scene.json`` next to it (served by the API). A copy of
that JSON is also inlined so a local preview can still render buildings if
``fetch`` is blocked. The mosaic texture loads from ``mosaic.jpg``.
"""
from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.geo import SunGeometry, height_from_shadow


def _write_jpeg(path: Path, bgr: np.ndarray, quality: int = 86) -> None:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"jpeg encode failed: {path}")
    path.write_bytes(buf.tobytes())


def _fit_max_side(bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return np.ascontiguousarray(bgr)
    scale = max_side / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _measure_for(index: int, measures: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not measures:
        return None
    for item in measures:
        if int(item.get("index", -1)) == index:
            return item
    if index < len(measures):
        item = measures[index]
        if "index" not in item or int(item["index"]) == index:
            return item
    return None


def _roof_rgb(mosaic: np.ndarray, box: list[int] | tuple[int, ...]) -> list[int]:
    """Mean roof colour as RGB 0-255, sampled from the surveyed mosaic."""
    h, w = mosaic.shape[:2]
    x, y, bw, bh = (int(v) for v in list(box)[:4])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + max(bw, 1)), min(h, y + max(bh, 1))
    if x1 <= x0 or y1 <= y0:
        return [180, 140, 80]
    crop = mosaic[y0:y1, x0:x1]
    if crop.size == 0:
        return [180, 140, 80]
    b, g, r = (float(v) for v in crop.reshape(-1, crop.shape[-1])[:, :3].mean(axis=0))
    return [int(round(r)), int(round(g)), int(round(b))]


def _ring_px(cnt: np.ndarray, peri_frac: float = 0.008) -> list[list[int]]:
    peri = float(cv2.arcLength(cnt, True)) or 1.0
    approx = cv2.approxPolyDP(cnt, max(0.8, peri_frac * peri), True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        pts = cv2.convexHull(cnt).reshape(-1, 2)
    if len(pts) > 96:
        pts = cv2.approxPolyDP(cnt, peri * 0.016, True).reshape(-1, 2)
    return [[int(p[0]), int(p[1])] for p in pts]


def _px_to_m(ring: list[list[int]] | list[list[float]], gsd: float) -> list[list[float]]:
    return [[round(float(x) * gsd, 4), round(-float(y) * gsd, 4)] for x, y in ring]


def _mean_rgb_in(mosaic: np.ndarray, ring: list[list[int]]) -> list[int] | None:
    if len(ring) < 3:
        return None
    canvas = np.zeros(mosaic.shape[:2], np.uint8)
    cv2.fillPoly(canvas, [np.array(ring, np.int32)], 255)
    pix = mosaic[canvas > 0]
    if pix.size == 0:
        return None
    b, g, r = (float(v) for v in pix[:, :3].mean(axis=0))
    return [int(round(r)), int(round(g)), int(round(b))]


def _bright_mask(mosaic: np.ndarray, box: list[float] | tuple[float, ...]) -> np.ndarray:
    h, w = mosaic.shape[:2]
    x, y, bw, bh = (int(v) for v in list(box)[:4])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + max(bw, 1)), min(h, y + max(bh, 1))
    mask = np.zeros((h, w), np.uint8)
    if x1 <= x0 or y1 <= y0:
        return mask
    crop = mosaic[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    hi = int(max(8, np.percentile(eq, 60)))
    _, local = cv2.threshold(eq, max(hi - 1, 0), 255, cv2.THRESH_BINARY)
    if int((local > 0).sum()) < 32:
        _, local = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, k, iterations=1)
    local = cv2.morphologyEx(
        local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask[y0:y1, x0:x1] = local
    return mask


def _footprint_rings(
    mosaic: np.ndarray,
    box: list[float] | tuple[float, ...],
    outline_px: list | None,
) -> tuple[list[list[int]] | None, list[list[list[int]]]]:
    """Nadir silhouette (outer ring + arch holes) in image pixels."""
    h, w = mosaic.shape[:2]
    outer: list[list[int]] | None = None
    if outline_px and len(outline_px) >= 3:
        outer = [[int(p[0]), int(p[1])] for p in outline_px]
    else:
        mask = _bright_mask(mosaic, box)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) >= 24:
                outer = _ring_px(cnt)
    if not outer or len(outer) < 3:
        return None, []

    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY) if mosaic.ndim == 3 else mosaic
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillPoly(canvas, [np.array(outer, np.int32)], 255)
    vals = gray[canvas > 0]
    if vals.size < 40:
        return outer, []
    lo = int(np.percentile(vals, 22))
    hole_mask = ((gray < lo) & (canvas > 0)).astype(np.uint8) * 255
    hole_mask = cv2.morphologyEx(
        hole_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    hole_mask = cv2.bitwise_and(hole_mask, canvas)
    hcnts, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_area = max(abs(cv2.contourArea(np.array(outer, np.int32))), 1.0)
    holes: list[list[list[int]]] = []
    for c in sorted(hcnts, key=cv2.contourArea, reverse=True)[:3]:
        area = abs(cv2.contourArea(c))
        if area < 0.035 * outer_area or area > 0.42 * outer_area:
            continue
        ring = _ring_px(c, peri_frac=0.02)
        if len(ring) >= 3:
            holes.append(ring)
    return outer, holes


def _extrusion_m(structure: dict[str, Any], measured: dict[str, Any] | None,
                 sun: SunGeometry) -> tuple[float, str]:
    """Pick an extrusion height: hunt, shadow length, or footprint — never a squat stub."""
    width_m = float(structure.get("width_m") or 0.0)
    depth_m = float(structure.get("height_m") or 0.0)  # footprint extent, not extrusion
    span = min(v for v in (width_m, depth_m) if v > 0) if (width_m > 0 or depth_m > 0) else 0.0
    if span <= 0:
        box = structure.get("box") or [0, 0, 20, 20]
        span = min(float(box[2]), float(box[3])) * max(sun.gsd_m, 1e-6)

    stated = 0.0
    for blob in (structure, measured or {}):
        if blob.get("stated_height_m") is not None:
            try:
                stated = float(blob["stated_height_m"])
            except (TypeError, ValueError):
                stated = 0.0
            if stated >= 8.0:
                break

    hunt = 0.0
    if measured is not None and measured.get("height_m") is not None:
        hunt = float(measured["height_m"])

    shadow_h = 0.0
    sl = float(structure.get("shadow_len_px") or (measured or {}).get("shadow_len_px") or 0.0)
    if sl > 2 and sun.elevation_deg >= 8.0:
        shadow_h = height_from_shadow(sl, sun)

    span_h = max(8.0, min(80.0, 0.55 * span)) if span > 1.0 else 12.0

    # Named monuments: operator-stated metres are the extrusion. Hunt stays
    # INDICATIVE in measures/UI and must not steal the volume (even at ≥0.4×).
    if stated >= 8.0:
        return stated, "stated"
    if hunt > 0.0 and span > 0.0 and hunt >= 0.45 * span:
        return hunt, "measured"
    if shadow_h > max(hunt, 8.0):
        return shadow_h, "shadow"
    return span_h, "footprint"


def _is_cover(structure: dict[str, Any], frame_px: int, frame_m2: float,
              width_m: float, depth_m: float, *,
              outline_frac: float | None = None) -> bool:
    """True only for an empty whole-AOI seed with nothing to extrude.

    A tight crop around a named monument *is* the structure. Stated height,
    compact footprint, or a real silhouette must become a prism — never the
    luminance-relief fallback.
    """
    if structure.get("compact") or structure.get("tight_aoi"):
        return False
    try:
        stated = float(structure.get("stated_height_m") or 0)
    except (TypeError, ValueError):
        stated = 0.0
    if stated >= 8.0:
        return False
    if outline_frac is not None and 0.02 <= outline_frac < 0.70:
        return False
    box = structure.get("box") or [0, 0, 1, 1]
    frac = (float(box[2]) * float(box[3])) / max(int(frame_px), 1)
    footprint = width_m * depth_m
    covering = frac >= 0.45 or footprint >= 0.45 * max(frame_m2, 1.0)
    weak = (float(structure.get("shadow_support") or 0) < 0.15
            and float(structure.get("shadow_len_px") or 0) < 4)
    return bool(structure.get("seeded")) and covering and weak


def compose_field(
    mosaic_bgr: np.ndarray,
    structures: list[dict[str, Any]] | None,
    measures: list[dict[str, Any]] | None,
    sun: dict[str, Any] | SunGeometry,
    gsd_m: float,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the live field scene: metres, roof RGB, sun-cast shadows."""
    mosaic = np.ascontiguousarray(mosaic_bgr)
    if mosaic.ndim == 2:
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)
    elif mosaic.shape[2] == 4:
        mosaic = mosaic[:, :, :3]
    h_px, w_px = mosaic.shape[:2]
    gsd = float(gsd_m) if gsd_m > 0 else 0.5
    geom = sun if isinstance(sun, SunGeometry) else SunGeometry(
        azimuth_deg=float(sun.get("azimuth_deg", 148.0)),
        elevation_deg=float(sun.get("elevation_deg", 41.0)),
        gsd_m=float(sun.get("gsd_m", gsd)),
    )
    items = list(structures or [])
    if not items:
        pad = max(4, int(0.05 * min(h_px, w_px)))
        bw, bh = max(8, w_px - 2 * pad), max(8, h_px - 2 * pad)
        items = [{
            "box": [pad, pad, bw, bh], "index": 0, "seeded": True,
            "width_m": bw * gsd, "height_m": bh * gsd,
            "quick_height_m": 0.0, "score": 0.25, "shadow_len_px": 0.0,
        }]

    landmark = (meta or {}).get("landmark") if isinstance(meta, dict) else None
    if not isinstance(landmark, dict):
        landmark = None

    elev = max(float(geom.elevation_deg), 0.5)
    tan_e = math.tan(math.radians(min(elev, 89.0)))
    sdx, sdy = geom.shadow_vector  # image: +x right, +y down

    buildings: list[dict[str, Any]] = []
    for i, raw in enumerate(items):
        structure = dict(raw)
        if landmark and structure.get("stated_height_m") is None:
            try:
                stated_aoi = float(landmark.get("height_m") or 0)
            except (TypeError, ValueError):
                stated_aoi = 0.0
            if stated_aoi >= 8.0:
                structure["stated_height_m"] = stated_aoi
        idx = int(structure.get("index", i))
        measured = _measure_for(idx, measures)
        x, y, bw, bh, cx, cy = _box_center(structure)
        height, source = _extrusion_m(structure, measured, geom)
        # Footprint = detect/compact silhouette. Hunt/RL crop is never the 3D plan.
        width_m = max(bw, 1.0) * gsd
        depth_m = max(bh, 1.0) * gsd
        outer_px, holes_px = _footprint_rings(
            mosaic, [x, y, bw, bh], structure.get("outline_px"))
        outline_m = _px_to_m(outer_px, gsd) if outer_px else None
        holes_m = [_px_to_m(ring, gsd) for ring in holes_px]
        if outline_m:
            xs = [p[0] for p in outline_m]
            ys = [p[1] for p in outline_m]
            width_m = max(max(xs) - min(xs), 0.5)
            depth_m = max(max(ys) - min(ys), 0.5)
            cx = float(np.mean([p[0] for p in outer_px]))
            cy = float(np.mean([p[1] for p in outer_px]))
        rgb = _mean_rgb_in(mosaic, outer_px) if outer_px else None
        if rgb is None:
            rgb = _roof_rgb(mosaic, [int(x), int(y), int(bw), int(bh)])
        shadow_len_m = height / tan_e if tan_e > 1e-6 else 0.0
        outline_frac = None
        if outer_px and len(outer_px) >= 3:
            outline_frac = abs(cv2.contourArea(np.array(outer_px, np.int32))) / max(w_px * h_px, 1)
        cover = _is_cover(structure, w_px * h_px, (w_px * gsd) * (h_px * gsd),
                          width_m, depth_m, outline_frac=outline_frac)
        entry: dict[str, Any] = {
            "index": idx,
            "x_m": round(cx * gsd, 4),
            "y_m": round(-cy * gsd, 4),
            "width_m": round(max(width_m, 0.5), 4),
            "depth_m": round(max(depth_m, 0.5), 4),
            "height_m": round(height, 3),
            "height_source": source,
            "rgb": rgb,
            "cover": cover,
            "score": round(float(structure.get("score") or 0.0), 4),
            "box": [int(x), int(y), int(bw), int(bh)],
            "geom": {
                "width_m": round(max(width_m, 0.5), 2),
                "depth_m": round(max(depth_m, 0.5), 2),
                "height_m": round(height, 2),
                "source": source,
            },
            "stated_height_m": structure.get("stated_height_m"),
            "shadow_dx_m": round(float(sdx) * shadow_len_m, 4),
            "shadow_dy_m": round(-float(sdy) * shadow_len_m, 4),
            "floors": max(1, int(round(height / 3.2))),
        }
        if outline_m:
            entry["outline_m"] = outline_m
            entry["outline_px"] = outer_px
        if holes_m:
            entry["holes_m"] = holes_m
            entry["holes_px"] = holes_px
        if measured:
            if measured.get("lon") is not None:
                entry["lon"] = measured["lon"]
            if measured.get("lat") is not None:
                entry["lat"] = measured["lat"]
        elif structure.get("lat") is not None:
            entry["lat"] = structure.get("lat")
            entry["lon"] = structure.get("lon")
        if not cover:
            from .massing import assemble
            lat_b = entry.get("lat")
            lon_b = entry.get("lon")
            packed = assemble(
                height_m=height, width_m=width_m, depth_m=depth_m,
                x_m=float(entry["x_m"]), y_m=float(entry["y_m"]),
                outline_m=outline_m, rgb=rgb, landmark=landmark,
                vlm=(meta or {}).get("vlm"),
                construct=(meta or {}).get("construct"),
                measured=measured, cover=False,
                lat=float(lat_b) if lat_b is not None else None,
                lon=float(lon_b) if lon_b is not None else None,
            )
            if packed:
                entry["parts"] = packed["parts"]
                entry["massing_kind"] = packed["kind"]
                entry["massing_source"] = packed["source"]
                entry["agents"] = packed["agents"]
                if packed.get("plan_width_m"):
                    entry["width_m"] = max(float(entry["width_m"]), float(packed["plan_width_m"]))
                if packed.get("plan_depth_m"):
                    entry["depth_m"] = max(float(entry["depth_m"]), float(packed["plan_depth_m"]))
                # Marble tint when mosaic crop is dark / shadowed.
                if packed["kind"] == "azadi_arch":
                    entry["rgb"] = [236, 228, 214]
        buildings.append(entry)

    viable = any(not b.get("cover") for b in buildings)
    return {
        "title": "Shadow Hunter · field reconstruction",
        "title_fa": "شکارچی سایه · بازسازی میدان",
        "gsd_m": gsd,
        "viable": viable,
        "viable_reason": "volumes" if viable else "no_compact_shadow",
        "sun": {
            "azimuth_deg": round(geom.azimuth_deg, 2),
            "elevation_deg": round(geom.elevation_deg, 2),
            "gsd_m": round(geom.gsd_m, 4),
            "shadow_bearing_deg": round(geom.shadow_bearing_deg, 2),
        },
        "ground": {
            "width_m": round(w_px * gsd, 4),
            "height_m": round(h_px * gsd, 4),
            "texture": "mosaic.jpg",
            "pixels": [int(w_px), int(h_px)],
        },
        "buildings": buildings,
        "meta": dict(meta or {}),
    }


def _box_center(structure: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    box = structure.get("box") or [0, 0, 1, 1]
    x, y, w, h = (float(v) for v in box[:4])
    center = structure.get("center")
    if center and len(center) >= 2:
        cx, cy = float(center[0]), float(center[1])
    else:
        cx, cy = x + w / 2.0, y + h / 2.0
    return x, y, w, h, cx, cy


def _pad4(buf: bytearray) -> None:
    while len(buf) % 4:
        buf.append(0)


def _accessor_minmax(positions: list[float]) -> tuple[list[float], list[float]]:
    xs, ys, zs = positions[0::3], positions[1::3], positions[2::3]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def _build_gltf(buildings: list[dict[str, Any]], ground_w: float, ground_d: float) -> dict[str, Any]:
    """Minimal glTF 2.0, Y-up: our (x, y_north, z_up) -> (x, z_up, -y_north)."""
    ground_pos = [0.0, 0.0, 0.0, ground_w, 0.0, 0.0, ground_w, 0.0, ground_d, 0.0, 0.0, ground_d]
    ground_idx = [0, 1, 2, 0, 2, 3]
    box_pos = [
        -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5, -0.5,
        -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5, -0.5,  0.5,  0.5,
    ]
    box_idx = [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3,
        3, 7, 4, 3, 4, 0,
    ]

    blob = bytearray()
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def add_f32(values: list[float], count: int) -> int:
        _pad4(blob)
        offset = len(blob)
        blob.extend(struct.pack("<" + "f" * len(values), *values))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(values) * 4, "target": 34962})
        lo, hi = _accessor_minmax(values)
        accessors.append({
            "bufferView": len(views) - 1, "componentType": 5126, "count": count,
            "type": "VEC3", "min": lo, "max": hi,
        })
        return len(accessors) - 1

    def add_u16(values: list[int], count: int) -> int:
        _pad4(blob)
        offset = len(blob)
        blob.extend(struct.pack("<" + "H" * len(values), *values))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(values) * 2, "target": 34963})
        accessors.append({
            "bufferView": len(views) - 1, "componentType": 5123, "count": count, "type": "SCALAR",
        })
        return len(accessors) - 1

    acc_ground_pos = add_f32(ground_pos, 4)
    acc_ground_idx = add_u16(ground_idx, 6)
    acc_box_pos = add_f32(box_pos, 8)
    acc_box_idx = add_u16(box_idx, 36)

    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii")

    nodes: list[dict[str, Any]] = [{"mesh": 0, "name": "ground"}]
    scene_nodes = [0]
    for i, bld in enumerate(buildings):
        if bld.get("cover"):
            continue
        # glTF Y-up: X east, Y height, Z south-ish (pixel y).
        nodes.append({
            "mesh": 1,
            "name": f"building_{i}",
            "translation": [bld["x_m"], bld["height_m"] / 2.0, -bld["y_m"]],
            "scale": [bld["width_m"], bld["height_m"], bld["depth_m"]],
        })
        scene_nodes.append(len(nodes) - 1)

    return {
        "asset": {
            "version": "2.0",
            "generator": "Shadow Hunter",
            "extras": {
                "up": "source Z-up (XY ground, +Y north); this file is glTF Y-up",
            },
        },
        "scene": 0,
        "scenes": [{"nodes": scene_nodes, "name": "shadowhunter"}],
        "nodes": nodes,
        "meshes": [
            {"name": "ground", "primitives": [{
                "attributes": {"POSITION": acc_ground_pos},
                "indices": acc_ground_idx, "material": 0,
            }]},
            {"name": "box", "primitives": [{
                "attributes": {"POSITION": acc_box_pos},
                "indices": acc_box_idx, "material": 1,
            }]},
        ],
        "materials": [
            {"name": "ground", "pbrMetallicRoughness": {
                "baseColorFactor": [0.07, 0.09, 0.11, 1.0],
                "metallicFactor": 0.0, "roughnessFactor": 1.0,
            }},
            {"name": "building", "pbrMetallicRoughness": {
                "baseColorFactor": [0.95, 0.69, 0.125, 1.0],
                "metallicFactor": 0.05, "roughnessFactor": 0.45,
            }},
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"uri": uri, "byteLength": len(blob)}],
    }


def _viewer_html(embedded_json: str) -> str:
    return _HTML.replace("__SCENE_JSON__", embedded_json)


def export_scene(
    out_dir: Path,
    *,
    mosaic_bgr: np.ndarray,
    structures: list[dict[str, Any]],
    measures: list[dict[str, Any]] | None,
    gsd_m: float,
    sun: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Write mosaic.jpg, scene.gltf, scene.html, scene.json into *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mosaic = np.ascontiguousarray(mosaic_bgr)
    if mosaic.ndim == 2:
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)
    elif mosaic.shape[2] == 4:
        mosaic = mosaic[:, :, :3]

    scene = compose_field(mosaic, structures, measures, sun, gsd_m, meta)
    buildings = scene["buildings"]
    ground_w = float(scene["ground"]["width_m"])
    ground_d = float(scene["ground"]["height_m"])

    preview = _fit_max_side(mosaic, 1024)
    mosaic_path = out_dir / "mosaic.jpg"
    _write_jpeg(mosaic_path, preview)
    ok, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if ok:
        scene["ground"]["texture_data"] = "data:image/jpeg;base64," + base64.b64encode(
            buf.tobytes()).decode("ascii")

    json_path = out_dir / "scene.json"
    json_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")

    gltf = _build_gltf(buildings, ground_w, ground_d)
    gltf_path = out_dir / "scene.gltf"
    gltf_path.write_text(json.dumps(gltf, ensure_ascii=False), encoding="utf-8")

    embedded = json.dumps(scene, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    html_path = out_dir / "scene.html"
    html_path.write_text(_viewer_html(embedded), encoding="utf-8")

    return {
        "dir": str(out_dir),
        "html": str(html_path),
        "gltf": str(gltf_path),
        "json": str(json_path),
        "buildings": len(buildings),
    }


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Shadow Hunter · field reconstruction</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=IBM+Plex+Sans:wght@400;500&family=Vazirmatn:wght@500;700&display=swap" rel="stylesheet"/>
<style>
  :root { --void:#06080B; --ink:#E8F0F6; --muted:#8497A6; --solar:#FFB020; --cyan:#3FD3E4; --hair:#1F2C36; }
  html, body { margin:0; height:100%; background:var(--void); color:var(--ink);
    font-family:"IBM Plex Sans", "Vazirmatn", "Segoe UI", sans-serif; overflow:hidden; }
  canvas { display:block; }
  #hud { position:absolute; inset:0 0 auto 0; pointer-events:none;
    padding:18px 22px 0; z-index:2; }
  h1 { margin:0; font-family:"Chakra Petch", "Vazirmatn", sans-serif;
    font-size:18px; letter-spacing:1.4px; font-weight:700; }
  h1 span { color:var(--solar); }
  .fa { font-family:"Vazirmatn", sans-serif; color:var(--muted); font-size:13px;
    letter-spacing:0; margin-top:4px; direction:rtl; }
  #note { position:absolute; left:22px; bottom:18px; z-index:2; color:var(--muted);
    font-size:11px; letter-spacing:0.4px; max-width:min(520px, 80vw); }
  #note b { color:var(--cyan); font-weight:500; }
  #fallback { display:none; position:absolute; inset:0; z-index:3; place-items:center;
    background:var(--void); text-align:center; padding:32px; }
  #fallback.show { display:grid; }
  #fallback p { color:var(--muted); max-width:420px; }
</style>
</head>
<body>
<div id="hud">
  <h1>SHADOW HUNTER <span>·</span> FIELD RECONSTRUCTION</h1>
  <div class="fa">شکارچی سایه · بازسازی میدان</div>
</div>
<div id="note">Loading reconstruction… orbit drag · scroll zoom. Texture from mosaic.jpg.</div>
<div id="fallback">
  <div>
    <h1>WebGL unavailable</h1>
    <p>This viewer needs WebGL. Serve the folder over HTTP (Shadow Hunter API) —
    <code>file://</code> often blocks the Three.js ES module importmap.</p>
    <p class="fa">وب‌جی‌ال در دسترس نیست. پوشه را از طریق سرور برنامه باز کنید.</p>
  </div>
</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script>window.SCENE_DATA = __SCENE_JSON__;</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const note = document.getElementById('note');
const fallback = document.getElementById('fallback');
let ready = false;
setTimeout(() => {
  if (!ready) {
    note.innerHTML = 'Module import blocked by <b>file://</b>. Serve this folder over HTTP (Shadow Hunter API) so Three.js can load.';
  }
}, 2800);

async function loadScene() {
  try {
    const r = await fetch('./scene.json', { cache: 'no-store' });
    if (r.ok) return await r.json();
  } catch (err) { /* file:// or CORS — use the inlined copy */ }
  return window.SCENE_DATA;
}

function sunOffset(azDeg, elDeg, dist) {
  const az = (Number(azDeg) || 148) * Math.PI / 180;
  const el = (Number(elDeg) || 41) * Math.PI / 180;
  return new THREE.Vector3(
    Math.sin(az) * Math.cos(el) * dist,
    Math.cos(az) * Math.cos(el) * dist,
    Math.sin(el) * dist
  );
}

try {
  const data = await loadScene();
  const ground = data.ground || { width_m: 100, height_m: 100 };
  const W = Number(ground.width_m) || 100;
  const D = Number(ground.height_m) || 100;
  const cx = W * 0.5, cy = -D * 0.5;
  const span = Math.max(W, D, 20);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x06080B, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x06080B);
  scene.fog = new THREE.Fog(0x06080B, span * 0.9, span * 3.2);

  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.5, span * 20);
  camera.up.set(0, 0, 1);
  camera.position.set(cx + span * 0.55, cy - span * 0.75, span * 0.62);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(cx, cy, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 8;
  controls.maxDistance = span * 6;

  scene.add(new THREE.AmbientLight(0x8497A6, 0.42));
  scene.add(new THREE.HemisphereLight(0x3FD3E4, 0x1A1408, 0.18));
  const sun = data.sun || {};
  const key = new THREE.DirectionalLight(0xFFB020, 1.25);
  key.position.copy(sunOffset(sun.azimuth_deg, sun.elevation_deg, span * 2.2));
  key.target.position.set(cx, cy, 0);
  scene.add(key); scene.add(key.target);

  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(W, D),
    new THREE.MeshStandardMaterial({ color: 0x121A21, roughness: 1, metalness: 0 })
  );
  plane.position.set(cx, cy, 0);
  scene.add(plane);

  const loader = new THREE.TextureLoader();
  const texSrc = (ground.texture_data || ground.texture || 'mosaic.jpg');
  loader.load(texSrc, (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    plane.material.map = tex;
    plane.material.color.set(0xffffff);
    plane.material.needsUpdate = true;
  }, undefined, () => {
    note.innerHTML = 'Ground texture blocked — buildings still render.';
  });

  const grid = new THREE.GridHelper(span, 24, 0x1F2C36, 0x121A21);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(cx, cy, 0.02);
  scene.add(grid);

  const edgeMat = new THREE.LineBasicMaterial({ color: 0x3FD3E4, transparent: true, opacity: 0.55 });
  const shadowMat = new THREE.MeshBasicMaterial({
    color: 0x06080B, transparent: true, opacity: 0.45, depthWrite: false,
  });
  function ringToPath(ring, asHole) {
    const s = asHole ? new THREE.Path() : new THREE.Shape();
    ring.forEach((p, i) => {
      const x = Number(p[0]), y = Number(p[1]);
      if (i === 0) s.moveTo(x, y); else s.lineTo(x, y);
    });
    s.closePath();
    return s;
  }
  for (const b of (data.buildings || [])) {
    if (b.cover) continue;
    const hh = Math.max(Number(b.height_m) || 10, 0.5);
    const rgb = b.rgb || [255, 176, 32];
    const col = new THREE.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255);
    const mat = new THREE.MeshStandardMaterial({
      color: col, roughness: 0.48, metalness: 0.06,
      emissive: col.clone().multiplyScalar(0.12),
    });
    const parts = b.parts || [];
    if (parts.length >= 2) {
      parts.forEach(part => {
        const pr = part.rgb || rgb;
        const pcol = new THREE.Color(pr[0]/255, pr[1]/255, pr[2]/255);
        const pmat = new THREE.MeshStandardMaterial({
          color: pcol, roughness: 0.48, metalness: 0.06,
          emissive: pcol.clone().multiplyScalar(0.10),
        });
        const pw = Math.max(Number(part.width_m) || 1, 0.5);
        const pd = Math.max(Number(part.depth_m) || 1, 0.5);
        const ph = Math.max(Number(part.height_m) || 1, 0.5);
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(pw, pd, ph), pmat);
        mesh.position.set(Number(part.x_m)||0, Number(part.y_m)||0, Number(part.z_m)||(ph*0.5));
        mesh.rotation.z = Number(part.yaw) || 0;
        scene.add(mesh);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), edgeMat);
        edges.position.copy(mesh.position);
        edges.rotation.copy(mesh.rotation);
        scene.add(edges);
      });
      continue;
    }
    const outline = b.outline_m;
    let mesh;
    if (outline && outline.length >= 3) {
      const shape = ringToPath(outline, false);
      (b.holes_m || []).forEach(h => {
        if (h && h.length >= 3) shape.holes.push(ringToPath(h, true));
      });
      const geo = new THREE.ExtrudeGeometry(shape, {depth: hh, bevelEnabled: false, steps: 1, curveSegments: 1});
      mesh = new THREE.Mesh(geo, mat);
    } else {
      const hw = Math.max(Number(b.width_m) || 1, 0.5);
      const hd = Math.max(Number(b.depth_m) || 1, 0.5);
      mesh = new THREE.Mesh(new THREE.BoxGeometry(hw, hd, hh), mat);
      mesh.position.set(Number(b.x_m) || 0, Number(b.y_m) || 0, hh * 0.5);
    }
    scene.add(mesh);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), edgeMat);
    edges.position.copy(mesh.position);
    edges.rotation.copy(mesh.rotation);
    scene.add(edges);
    const sdx = Number(b.shadow_dx_m) || 0, sdy = Number(b.shadow_dy_m) || 0;
    if (Math.hypot(sdx, sdy) > 1) {
      const extra = new THREE.Shape();
      if (outline && outline.length >= 3) {
        outline.forEach((p, i) => {
          const x = Number(p[0]) + sdx, y = Number(p[1]) + sdy;
          if (i === 0) extra.moveTo(x, y); else extra.lineTo(x, y);
        });
        extra.closePath();
      } else {
        const hw = Math.max(Number(b.width_m) || 1, 0.5) / 2;
        const hd = Math.max(Number(b.depth_m) || 1, 0.5) / 2;
        const cx = Number(b.x_m) || 0, cy = Number(b.y_m) || 0;
        extra.moveTo(cx-hw, cy-hd); extra.lineTo(cx+hw, cy-hd);
        extra.lineTo(cx+hw+sdx, cy-hd+sdy); extra.lineTo(cx-hw+sdx, cy-hd+sdy);
      }
      const shadow = new THREE.Mesh(new THREE.ShapeGeometry(extra), shadowMat);
      shadow.position.set(0, 0, 0.04);
      scene.add(shadow);
    }
  }

  ready = true;
  const n = (data.buildings || []).filter(b => !b.cover).length;
  note.innerHTML = n + ' structures · gsd ' + (data.gsd_m || '?') + ' m/px · <b>orbit</b> to inspect. If blank on file://, open through the app server.';

  function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
} catch (err) {
  fallback.classList.add('show');
  note.textContent = String(err);
}
</script>
</body>
</html>
"""
