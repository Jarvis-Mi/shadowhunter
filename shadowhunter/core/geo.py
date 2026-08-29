"""Solar geometry - the physics that turns a shadow into a height.

The whole project rests on one relation:

    h = L * tan(theta_sun)

where ``L`` is the ground length of the cast shadow and ``theta_sun`` the solar
elevation angle at acquisition time. Everything else (the RL agent, the CNN) is
machinery for measuring ``L`` honestly on a messy image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SunGeometry:
    """Illumination metadata, normally read from the scene's own metadata.

    azimuth_deg : compass bearing the sun is *coming from* (0 = N, 90 = E).
    elevation_deg : angle above the horizon.
    gsd_m : ground sample distance - metres per pixel.
    """
    azimuth_deg: float = 148.0
    elevation_deg: float = 41.0
    gsd_m: float = 0.5

    @property
    def shadow_bearing_deg(self) -> float:
        """Direction the shadow is cast toward (opposite the sun)."""
        return (self.azimuth_deg + 180.0) % 360.0

    @property
    def shadow_vector(self) -> tuple[float, float]:
        """Unit vector in image space (dx right, dy down) along the shadow."""
        rad = math.radians(self.shadow_bearing_deg)
        return math.sin(rad), -math.cos(rad)

    @property
    def axis_deg(self) -> float:
        """Shadow axis folded to [0, 180) - orientation without direction.

        Used to score how well a blob's principal axis lines up with the sun.
        """
        return self.shadow_bearing_deg % 180.0


def height_from_shadow(shadow_px: float, sun: SunGeometry) -> float:
    """Metres of building height implied by a shadow of ``shadow_px`` pixels."""
    if shadow_px <= 0:
        return 0.0
    elev = float(sun.elevation_deg)
    if elev <= 0:
        return 0.0
    t = math.tan(math.radians(min(elev, 89.0)))
    if not math.isfinite(t):
        return 0.0
    value = shadow_px * sun.gsd_m * t
    return value if math.isfinite(value) else 0.0


def shadow_from_height(height_m: float, sun: SunGeometry) -> float:
    """Inverse of :func:`height_from_shadow`, in pixels. Used by the simulator."""
    if height_m <= 0:
        return 0.0
    t = math.tan(math.radians(min(max(sun.elevation_deg, 0.0), 89.0)))
    if t <= 1e-6 or not math.isfinite(t):
        return 0.0
    return (height_m / t) / sun.gsd_m


def floors_from_height(height_m: float, floor_m: float = 3.2) -> int:
    """Storey count estimate - what a municipality actually wants to see."""
    return max(0, int(round(height_m / floor_m)))


def geometric_uncertainty(shadow_px: float, sun: SunGeometry, mask_noise_px: float = 1.5) -> float:
    """Propagate a +/- ``mask_noise_px`` segmentation error into metres.

    Low sun -> long shadow -> the same pixel error costs less height error.
    This is exactly why 'Idea 3' (temporal selection) is worth having, and it
    is what the UI shows next to every estimate.
    """
    elev = min(max(float(sun.elevation_deg), 0.0), 89.0)
    dh_dl = math.tan(math.radians(elev)) * sun.gsd_m
    value = abs(dh_dl) * mask_noise_px
    return value if math.isfinite(value) else 0.0


def quality_of_geometry(sun: SunGeometry) -> float:
    """0..1 score for how favourable the illumination is for shadow metrology.

    Peaks around 25-35 degrees elevation: long, well-formed shadows that still
    fit inside a tile. Near-zenith sun (short shadows) and near-horizon sun
    (shadows that merge into each other) are both penalised.
    """
    e = sun.elevation_deg
    if e <= 5 or e >= 85:
        return 0.05
    ideal = 30.0
    return float(max(0.05, math.exp(-((e - ideal) ** 2) / (2 * 18.0 ** 2))))
