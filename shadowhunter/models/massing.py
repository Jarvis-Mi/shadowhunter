"""Agent massing: VLM identity + LLM recipe + RL footprint → 3D parts.

A single undated nadir mosaic cannot photogrammetrize facades. This module
builds a *visualization* from four agents:

    VLM     what the mosaic looks like (arch, plaza, roof)
    LLM     construction kind (azadi_arch / arch / tower / prism)
    RL/CNN  hunt metres — INDICATIVE only
    landmark / operator-stated height for named monuments

Azadi at 35.69974 N, 51.33810 E uses a parametric inverted-Y arch (two piers,
vault, wings) at the operator-stated 43 m — not a cookie-cutter slab.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .landmarks import lookup as lookup_landmark

AZADI_LAT = 35.69974
AZADI_LON = 51.33810

_ARCH_HINTS = (
    "azadi", "آزادی", "آزادي", "arch", "طاق", "monument", "برج",
    "inverted y", "y-shaped", "triumphal",
)


def plan_yaw(outline_m: list[list[float]] | None) -> float:
    """Principal axis of the nadir silhouette, radians about +Z."""
    if not outline_m or len(outline_m) < 3:
        return 0.0
    pts = np.asarray(outline_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return 0.0
    pts = pts[:, :2] - pts[:, :2].mean(axis=0)
    if float(np.abs(pts).sum()) < 1e-6:
        return 0.0
    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    axis = vh[0]
    return float(math.atan2(axis[1], axis[0]))


def _vlm_text(vlm: Any) -> str:
    if isinstance(vlm, dict):
        return str(vlm.get("text") or "")
    return str(vlm or "")


def infer_kind(landmark: dict[str, Any] | None, vlm: Any = None,
               construct: dict[str, Any] | None = None) -> tuple[str, str]:
    """Pick a massing grammar. Landmark identity wins; VLM/LLM can upgrade a prism."""
    if isinstance(landmark, dict):
        lid = str(landmark.get("id") or "").lower()
        name = str(landmark.get("name") or landmark.get("name_fa") or "").lower()
        if lid == "azadi_tower" or "azadi" in name or "آزادی" in name:
            return "azadi_arch", "landmark"
    blob = " ".join((
        _vlm_text(vlm),
        str((construct or {}).get("build_en") or ""),
        str((construct or {}).get("build_fa") or ""),
        str((construct or {}).get("massing") or ""),
    )).lower()
    parsed = (construct or {}).get("massing")
    if isinstance(parsed, dict) and parsed.get("kind"):
        kind = str(parsed["kind"]).strip().lower()
        if kind in {"azadi_arch", "arch", "tower", "prism"}:
            if kind == "arch":
                kind = "azadi_arch"
            return kind, "llm"
    if any(h in blob for h in _ARCH_HINTS):
        return "azadi_arch", "vlm"
    return "prism", "silhouette"


def _rot(cx: float, cy: float, x: float, y: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return cx + c * x - s * y, cy + s * x + c * y


def azadi_arch_parts(
    *,
    x_m: float,
    y_m: float,
    width_m: float,
    depth_m: float,
    height_m: float,
    yaw: float = 0.0,
    rgb: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Parametric Azadi: two piers, vault, inverted-Y wings. Height is stated metres."""
    # Enforce a readable plan even when detect cropped only the vault (≈37×33).
    w = max(float(width_m), 48.0)
    d = max(float(depth_m), 42.0)
    h = max(float(height_m), 8.0)
    yaw = float(yaw)
    if d < w * 0.85:
        # Prefer stem of inverted-Y along depth.
        pass
    pier_w = max(w * 0.18, 8.0)
    pier_d = max(d * 0.48, 18.0)
    arch_h = h * 0.62
    gap = max(w * 0.34, 16.0)
    colour = list(rgb or [236, 228, 214])
    pier_colour = [max(0, colour[0] - 18), max(0, colour[1] - 16), max(0, colour[2] - 12)]
    wing_colour = [min(255, colour[0] + 8), min(255, colour[1] + 6), min(255, colour[2] + 4)]
    parts: list[dict[str, Any]] = []

    def box(lx: float, ly: float, zc: float, bw: float, bd: float, bh: float,
            yaw_off: float = 0.0, role: str = "", tint: list[int] | None = None) -> dict[str, Any]:
        px, py = _rot(x_m, y_m, lx, ly, yaw)
        return {
            "kind": "box",
            "role": role,
            "x_m": round(px, 3),
            "y_m": round(py, 3),
            "z_m": round(zc, 3),
            "width_m": round(bw, 3),
            "depth_m": round(bd, 3),
            "height_m": round(bh, 3),
            "yaw": round(yaw + yaw_off, 4),
            "rgb": list(tint or colour),
        }

    for sign, role in ((-1.0, "pier_west"), (1.0, "pier_east")):
        lx = sign * (gap / 2.0 + pier_w / 2.0)
        parts.append(box(lx, d * 0.02, arch_h / 2.0, pier_w, pier_d, arch_h,
                         role=role, tint=pier_colour))

    crown_h = max(h - arch_h, 6.0)
    parts.append(box(0.0, 0.0, arch_h + crown_h / 2.0,
                     gap + pier_w * 0.55, pier_d * 0.72, crown_h, role="vault"))

    wing_h = h * 0.72
    wing_w = w * 0.28
    wing_d = d * 0.42
    for sign, ang, role in ((-1.0, 0.55, "wing_sw"), (1.0, -0.55, "wing_se")):
        parts.append(box(sign * w * 0.34, -d * 0.28, wing_h / 2.0,
                         wing_w, wing_d, wing_h, yaw_off=ang, role=role,
                         tint=wing_colour))
    return parts


def prism_part(
    *,
    x_m: float,
    y_m: float,
    width_m: float,
    depth_m: float,
    height_m: float,
    yaw: float = 0.0,
    rgb: list[int] | None = None,
) -> list[dict[str, Any]]:
    return [{
        "kind": "box",
        "role": "mass",
        "x_m": round(x_m, 3),
        "y_m": round(y_m, 3),
        "z_m": round(max(height_m, 0.8) / 2.0, 3),
        "width_m": round(max(width_m, 0.5), 3),
        "depth_m": round(max(depth_m, 0.5), 3),
        "height_m": round(max(height_m, 0.8), 3),
        "yaw": round(float(yaw), 4),
        "rgb": list(rgb or [200, 180, 140]),
    }]


def assemble(
    *,
    height_m: float,
    width_m: float,
    depth_m: float,
    x_m: float,
    y_m: float,
    outline_m: list[list[float]] | None = None,
    rgb: list[int] | None = None,
    landmark: dict[str, Any] | None = None,
    vlm: Any = None,
    construct: dict[str, Any] | None = None,
    measured: dict[str, Any] | None = None,
    cover: bool = False,
    lat: float | None = None,
    lon: float | None = None,
) -> dict[str, Any] | None:
    """Fuse agents into a part list. None when there is nothing to extrude."""
    if cover:
        return None
    hit = landmark if isinstance(landmark, dict) else None
    if hit is None and lat is not None and lon is not None:
        hit = lookup_landmark(float(lat), float(lon))
    kind, source = infer_kind(hit, vlm, construct)
    yaw = plan_yaw(outline_m)
    # Named monuments: use operator plan when detect box is a vault crop.
    plan_w, plan_d = float(width_m), float(depth_m)
    if isinstance(hit, dict):
        try:
            fw = float(hit.get("footprint_width_m") or 0)
            fd = float(hit.get("footprint_depth_m") or 0)
        except (TypeError, ValueError):
            fw = fd = 0.0
        if fw >= 20.0 and fd >= 20.0:
            plan_w = max(plan_w, fw)
            plan_d = max(plan_d, fd)
    hunt = None
    policy = None
    cnn = None
    if isinstance(measured, dict):
        try:
            hunt = float(measured["height_m"]) if measured.get("height_m") is not None else None
        except (TypeError, ValueError):
            hunt = None
        policy = measured.get("policy")
        cnn = measured.get("cnn_m")
    if kind == "azadi_arch":
        parts = azadi_arch_parts(
            x_m=x_m, y_m=y_m, width_m=plan_w, depth_m=plan_d,
            height_m=height_m, yaw=yaw, rgb=rgb,
        )
    elif kind == "tower":
        podium = height_m * 0.22
        parts = prism_part(
            x_m=x_m, y_m=y_m, width_m=plan_w, depth_m=plan_d,
            height_m=podium, yaw=yaw, rgb=rgb,
        )
        parts.extend(prism_part(
            x_m=x_m, y_m=y_m, width_m=plan_w * 0.72, depth_m=plan_d * 0.72,
            height_m=height_m - podium, yaw=yaw, rgb=rgb,
        ))
        parts[-1]["z_m"] = round(podium + (height_m - podium) / 2.0, 3)
        parts[-1]["role"] = "shaft"
        parts[0]["role"] = "podium"
    else:
        parts = prism_part(
            x_m=x_m, y_m=y_m, width_m=plan_w, depth_m=plan_d,
            height_m=height_m, yaw=yaw, rgb=rgb,
        )
    return {
        "kind": kind,
        "source": source,
        "parts": parts,
        "yaw": round(yaw, 4),
        "plan_width_m": round(plan_w, 2),
        "plan_depth_m": round(plan_d, 2),
        "honesty": "visualization — hunt metres INDICATIVE; stated height is operator identity",
        "agents": {
            "vlm": bool(_vlm_text(vlm).strip()),
            "llm": bool((construct or {}).get("used_llm")),
            "rl_policy": policy,
            "cnn_m": cnn,
            "hunt_m": hunt,
            "kind_source": source,
        },
    }
