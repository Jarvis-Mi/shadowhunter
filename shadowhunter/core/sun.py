"""Real solar position - NOAA's algorithm, in pure Python.

Once the operator picks a real place on a real map, the sun angle stops being
a slider and becomes a fact: it follows from latitude, longitude and the
moment of acquisition. This module computes it, so ``h = L*tan(theta)`` is
evaluated with the geometry that actually produced the shadow.

Reference: NOAA Solar Calculator (Astronomical Algorithms, Meeus).
Accuracy is well under a degree for any date between 1901 and 2099 - far
better than the pixel noise in the shadow mask.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .geo import SunGeometry


@dataclass(frozen=True)
class SolarPosition:
    azimuth_deg: float        # compass bearing the sun is coming FROM
    elevation_deg: float      # above the horizon, refraction-corrected
    declination_deg: float
    equation_of_time_min: float
    hour_angle_deg: float
    is_daylight: bool

    def to_geometry(self, gsd_m: float) -> SunGeometry:
        """Map this position onto a ``SunGeometry``.

        Callers MUST check ``is_daylight`` before treating the result as a
        valid daytime geometry. Elevation is passed through when it is above
        the horizon; night values (including negative) become 0.01 for
        numeric stability only and are not clamped to a daytime angle.
        """
        elev = self.elevation_deg if self.elevation_deg > 0 else 0.01
        return SunGeometry(azimuth_deg=self.azimuth_deg,
                           elevation_deg=elev,
                           gsd_m=gsd_m)


@dataclass
class IlluminationChoice:
    """Sun picked for shadow metrology, with the year's seasonal curve attached.

    Iterable over ``(moment, position, geometry, snapped)`` so existing
    unpacking ``a, b, c, d = resolve_illumination(...)`` keeps working.
    ``season`` is set only when the year's best marker beat the requested day.
    """
    moment: datetime
    position: SolarPosition
    geometry: SunGeometry
    snapped: bool
    season: str | None = None
    year_curve: list = field(default_factory=list)

    def __iter__(self):
        yield self.moment
        yield self.position
        yield self.geometry
        yield self.snapped


_SEASON_MARKERS: tuple[tuple[str, int, int], ...] = (
    ("spring", 3, 21),
    ("summer", 6, 21),
    ("autumn", 9, 22),
    ("winter", 12, 21),
)


def _julian_day(when: datetime) -> float:
    when = when.astimezone(timezone.utc)
    y, m = when.year, when.month
    d = (when.day + when.hour / 24 + when.minute / 1440
         + (when.second + when.microsecond / 1e6) / 86400)
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def solar_position(lat_deg: float, lon_deg: float, when: datetime) -> SolarPosition:
    """Sun azimuth and elevation for a place and an instant."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    jd = _julian_day(when)
    t = (jd - 2451545.0) / 36525.0                       # Julian centuries since J2000

    # --- orbital elements ---------------------------------------------------
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    m_rad = math.radians(m)
    c = (math.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
         + math.sin(3 * m_rad) * 0.000289)

    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # --- obliquity and declination -----------------------------------------
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    mean_obliq = 23.0 + (26.0 + seconds / 60.0) / 60.0
    obliq_corr = mean_obliq + 0.00256 * math.cos(math.radians(omega))

    decl = math.degrees(math.asin(math.sin(math.radians(obliq_corr))
                                  * math.sin(math.radians(app_long))))

    # --- equation of time ---------------------------------------------------
    y = math.tan(math.radians(obliq_corr / 2)) ** 2
    l0_rad = math.radians(l0)
    eq_time = 4 * math.degrees(
        y * math.sin(2 * l0_rad)
        - 2 * e * math.sin(m_rad)
        + 4 * e * y * math.sin(m_rad) * math.cos(2 * l0_rad)
        - 0.5 * y * y * math.sin(4 * l0_rad)
        - 1.25 * e * e * math.sin(2 * m_rad)
    )

    # --- hour angle ---------------------------------------------------------
    utc = when.astimezone(timezone.utc)
    minutes = utc.hour * 60 + utc.minute + utc.second / 60.0
    true_solar_time = (minutes + eq_time + 4 * lon_deg) % 1440
    hour_angle = true_solar_time / 4 - 180 if true_solar_time / 4 >= 0 else true_solar_time / 4 + 180
    if hour_angle < -180:
        hour_angle += 360

    # --- zenith / elevation -------------------------------------------------
    lat_rad = math.radians(lat_deg)
    decl_rad = math.radians(decl)
    ha_rad = math.radians(hour_angle)

    cos_zenith = (math.sin(lat_rad) * math.sin(decl_rad)
                  + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad))
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    elevation = 90.0 - zenith

    # atmospheric refraction - matters near the horizon, which is exactly
    # where the longest (most useful) shadows are
    if elevation > 85.0:
        refraction = 0.0
    elif elevation > 5.0:
        te = math.tan(math.radians(elevation))
        refraction = (58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5) / 3600.0
    elif elevation > -0.575:
        refraction = (1735.0 + elevation * (-518.2 + elevation
                      * (103.4 + elevation * (-12.79 + elevation * 0.711)))) / 3600.0
    else:
        refraction = -20.772 / math.tan(math.radians(elevation)) / 3600.0
    elevation += refraction

    # --- azimuth ------------------------------------------------------------
    denom = math.cos(lat_rad) * math.sin(math.radians(zenith))
    if abs(denom) < 1e-9:
        azimuth = 180.0 if lat_deg > 0 else 0.0
    else:
        cos_az = ((math.sin(lat_rad) * cos_zenith) - math.sin(decl_rad)) / denom
        cos_az = max(-1.0, min(1.0, cos_az))
        azimuth = math.degrees(math.acos(cos_az))
        azimuth = (180.0 + azimuth) % 360.0 if hour_angle > 0 else (540.0 - azimuth) % 360.0

    return SolarPosition(
        azimuth_deg=azimuth, elevation_deg=elevation, declination_deg=decl,
        equation_of_time_min=eq_time, hour_angle_deg=hour_angle,
        is_daylight=elevation > 0.0,
    )


def resolve_illumination(lat_deg: float, lon_deg: float, when: datetime,
                         gsd_m: float = 0.5, *, min_quality: float = 0.22
                         ) -> IlluminationChoice:
    """Pick a usable sun across the day *and* the year.

    Candidates: the requested instant (if daylight and ``>= min_quality``),
    that day's best hour, and the year's best season marker. Automatic
    snaps prefer whichever of the day's best hour and the year's best
    season scores higher - undated mosaics need the best shadow geometry
    of the year, not summer noon. ``season`` is set when the year-best
    marker is the one returned.
    """
    from .geo import quality_of_geometry

    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    curve = year_seasons(lat_deg, lon_deg, when.year, gsd_m=gsd_m)
    requested_pos = solar_position(lat_deg, lon_deg, when)
    requested_geom = requested_pos.to_geometry(gsd_m)
    requested_q = (quality_of_geometry(requested_geom)
                   if requested_pos.is_daylight else None)

    day_moment, day_pos = best_hour_of_day(lat_deg, lon_deg, when)
    day_geom = day_pos.to_geometry(gsd_m)
    day_q = quality_of_geometry(day_geom)

    year_best = max(curve, key=lambda row: row["quality"])
    year_q = float(year_best["quality"])
    year_when = datetime.fromisoformat(year_best["when"])
    if year_when.tzinfo is None:
        year_when = year_when.replace(tzinfo=timezone.utc)
    year_pos = solar_position(lat_deg, lon_deg, year_when)
    year_geom = year_pos.to_geometry(gsd_m)

    if year_q > day_q:
        auto_moment, auto_pos, auto_geom = year_when, year_pos, year_geom
        auto_q = year_q
        auto_season: str | None = year_best["season"]
    else:
        auto_moment, auto_pos, auto_geom = day_moment, day_pos, day_geom
        auto_q = day_q
        auto_season = None

    if (requested_q is not None and requested_q >= min_quality
            and requested_q >= auto_q):
        return IlluminationChoice(when, requested_pos, requested_geom, False,
                                  None, curve)
    return IlluminationChoice(auto_moment, auto_pos, auto_geom, True,
                              auto_season, curve)


def best_hour_of_day(lat_deg: float, lon_deg: float, day: datetime,
                     step_minutes: int = 20) -> tuple[datetime, SolarPosition]:
    """The acquisition time whose geometry is best for shadow metrology.

    Not solar noon - the *opposite*. Noon gives the shortest shadow and the
    largest height error. :func:`quality_of_geometry` peaks around 30 degrees
    elevation, so this walks the day and returns the moment that scores best
    while the sun is still comfortably up.
    """
    from .geo import quality_of_geometry

    day = day.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    best: tuple[datetime, SolarPosition, float] | None = None
    for minute in range(0, 24 * 60, step_minutes):
        moment = day + timedelta(minutes=minute)
        pos = solar_position(lat_deg, lon_deg, moment)
        if pos.elevation_deg < 8.0:
            continue
        score = quality_of_geometry(pos.to_geometry(1.0))
        if best is None or score > best[2]:
            best = (moment, pos, score)
    if best is None:                                   # polar night
        noon = day + timedelta(hours=12)
        return noon, solar_position(lat_deg, lon_deg, noon)
    return best[0], best[1]


def year_seasons(lat: float, lon: float, year: int, gsd_m: float = 0.5) -> list[dict]:
    """Best metrology hour on the four seasonal markers of ``year``.

    Spring is 21 Mar, summer 21 Jun, autumn 22 Sep, winter 21 Dec. Each
    dict carries ``season``, ISO ``when``, ``elevation_deg``, ``azimuth_deg``,
    ``quality`` and ``is_daylight``.
    """
    from .geo import quality_of_geometry

    rows: list[dict] = []
    for season, month, day in _SEASON_MARKERS:
        stamp = datetime(int(year), month, day, tzinfo=timezone.utc)
        moment, pos = best_hour_of_day(lat, lon, stamp)
        geom = pos.to_geometry(gsd_m)
        rows.append({
            "season": season,
            "when": moment.isoformat(),
            "elevation_deg": pos.elevation_deg,
            "azimuth_deg": pos.azimuth_deg,
            "quality": quality_of_geometry(geom),
            "is_daylight": pos.is_daylight,
        })
    return rows


def daylight_window(lat_deg: float, lon_deg: float, day: datetime) -> tuple[datetime | None, datetime | None]:
    """Approximate sunrise/sunset in UTC, by 5-minute scan. Used for UI bounds."""
    day = day.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rise = set_ = None
    previous = False
    for minute in range(0, 24 * 60 + 1, 5):
        moment = day + timedelta(minutes=minute)
        up = solar_position(lat_deg, lon_deg, moment).elevation_deg > 0
        if up and not previous and rise is None:
            rise = moment
        if not up and previous and set_ is None:
            set_ = moment
        previous = up
    return rise, set_
