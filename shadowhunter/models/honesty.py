"""Honesty flags for height claims.

Basemap mosaics have no acquisition time. The ephemeris sun is a fact about
*now* (or the date the operator typed), not about the pixels. Metres computed
from those pixels are indicative until a dated scene is loaded.
"""
from __future__ import annotations

from typing import Any


BASEMAP_PROVIDERS = frozenset({"esri", "osm", "carto_dark"})


def imagery_honesty(*, provider: str, source: str = "basemap",
                    is_daylight: bool = True, quality: float = 0.0,
                    sun_estimate: dict[str, Any] | None = None) -> dict[str, Any]:
    undated = provider in BASEMAP_PROVIDERS or source == "basemap"
    dated = (not undated) and source in {"geotiff", "dated"}
    estimate = sun_estimate or {}
    applied = bool(estimate.get("applied"))
    sun_source = "mixed" if applied else "ephemeris"

    warnings: list[str] = []
    if undated or not dated:
        warnings.append(
            "Imagery is an undated mosaic. Shadow length and solar elevation "
            "do not share a clock — reported metres are INDICATIVE, not a survey."
        )
        warnings.append(
            "موزاییک بدون زمان برداشت است. طول سایه و ارتفاع خورشید هم‌ساعت نیستند؛ "
            "اعداد متر تخمینی‌اند، نه نقشه‌برداری."
        )
    if not is_daylight:
        warnings.append("Sun is below the horizon at the chosen time — do not trust heights.")
        warnings.append("خورشید زیر افق است؛ ارتفاع‌ها معتبر نیستند.")
    if quality < 0.25 and is_daylight:
        warnings.append("Poor solar geometry (too high or too low). Prefer the best hour of day.")
        warnings.append("هندسهٔ خورشیدی ضعیف است. به بهترین ساعت روز بروید.")
    if applied and float(estimate.get("confidence") or 0) < 0.5:
        warnings.append("Image-derived azimuth is low-confidence; check the compass against the shadows.")

    if not is_daylight:
        verdict = "night"
    elif quality < 0.25:
        verdict = "poor_geometry"
    elif undated:
        verdict = "undated_basemap"
    elif dated:
        verdict = "measurable"
    else:
        verdict = "indicative"

    return {
        "imagery_dated": dated,
        "sun_source": sun_source,
        "elevation_trusted": bool(dated and is_daylight),
        "height_indicative": (not dated) or (not is_daylight),
        "verdict": verdict,
        "warnings": warnings,
    }
