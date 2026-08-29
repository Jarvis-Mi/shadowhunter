"""Toolkit-independent SVG components.

Used by the NiceGUI observatory today; any view that can render SVG (Flet
can, via ``flet.Image`` with a data URI) gets the same instrument for free.
"""
from __future__ import annotations

import base64
import math

from .. import theme


def sun_dial(azimuth_deg: float, elevation_deg: float, quality: float, size: int = 168) -> str:
    """Compass dial: solar needle, dashed shadow-cast needle, elevation arc."""
    c = size / 2
    r = c - 12
    parts: list[str] = [
        f'<svg class="sh-dial" viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="solar geometry dial">',
        f'<circle class="face" cx="{c}" cy="{c}" r="{r}" stroke-width="1"/>',
    ]

    for deg in range(0, 360, 15):
        major = deg % 90 == 0
        a = math.radians(deg - 90)
        r0 = r - (9 if major else 4)
        parts.append(
            f'<line class="tick{" major" if major else ""}" '
            f'x1="{c + math.cos(a) * r0:.1f}" y1="{c + math.sin(a) * r0:.1f}" '
            f'x2="{c + math.cos(a) * r:.1f}" y2="{c + math.sin(a) * r:.1f}" stroke-width="1"/>'
        )

    for deg, letter in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        a = math.radians(deg - 90)
        parts.append(
            f'<text x="{c + math.cos(a) * (r - 20):.1f}" y="{c + math.sin(a) * (r - 20) + 3:.1f}" '
            f'text-anchor="middle">{letter}</text>'
        )

    # elevation arc, drawn from due east clockwise
    ar = r - 14
    sweep = max(0.0, min(90.0, elevation_deg))
    end = math.radians(sweep - 90)
    parts.append(
        f'<path class="arc" d="M {c + ar:.1f} {c:.1f} A {ar:.1f} {ar:.1f} 0 0 0 '
        f'{c + math.cos(end) * ar:.1f} {c + math.sin(end) * ar:.1f}"/>'
    )

    a = math.radians(azimuth_deg - 90)
    parts.append(f'<line class="needle" x1="{c}" y1="{c}" '
                 f'x2="{c + math.cos(a) * (r - 10):.1f}" y2="{c + math.sin(a) * (r - 10):.1f}"/>')
    parts.append(f'<circle cx="{c + math.cos(a) * (r - 10):.1f}" cy="{c + math.sin(a) * (r - 10):.1f}" '
                 f'r="4" fill="{theme.c("solar")}"/>')

    a2 = math.radians(azimuth_deg + 180 - 90)
    parts.append(f'<line class="cast" x1="{c}" y1="{c}" '
                 f'x2="{c + math.cos(a2) * (r - 18):.1f}" y2="{c + math.sin(a2) * (r - 18):.1f}"/>')

    tone = theme.c("signal") if quality > 0.66 else theme.c("solar") if quality > 0.33 else theme.c("alert")
    parts.append(f'<circle cx="{c}" cy="{c}" r="21" fill="{theme.c("panel")}" '
                 f'stroke="{theme.c("hairlineHot")}" stroke-width="1"/>')
    parts.append(f'<text class="value" x="{c}" y="{c - 1}" text-anchor="middle" '
                 f'fill="{tone}">{elevation_deg:.0f}&#176;</text>')
    parts.append(f'<text x="{c}" y="{c + 11}" text-anchor="middle">Q {quality:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def sparkline(values: list[float], width: int = 320, height: int = 72,
              tone: str = "solar") -> str:
    """Filled area trace. Empty input renders an honest placeholder."""
    colour = theme.c(tone)
    if len(values) < 2:
        return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
                f'xmlns="http://www.w3.org/2000/svg">'
                f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
                f'fill="{theme.c("inkFaint")}" font-family="{theme.font_first("mono")}" '
                f'font-size="10" letter-spacing="2">AWAITING TELEMETRY</text></svg>')

    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-6)
    pad = 8
    inner = height - pad * 2
    pts = [(i / (len(values) - 1) * width, pad + inner - (v - lo) / span * inner)
           for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pts[0][0]:.1f},{height} " + line + f" {pts[-1][0]:.1f},{height}"
    uid = f"g{abs(hash((tone, len(values)))) % 99999}"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{colour}" stop-opacity=".38"/>'
        f'<stop offset="100%" stop-color="{colour}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#{uid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{colour}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.6" fill="{colour}"/>'
        f'<text x="4" y="11" fill="{theme.c("inkFaint")}" font-family="{theme.font_first("mono")}" '
        f'font-size="9">max {hi:+.2f}</text>'
        f'<text x="4" y="{height - 4}" fill="{theme.c("inkFaint")}" '
        f'font-family="{theme.font_first("mono")}" font-size="9">min {lo:+.2f}</text>'
        f"</svg>"
    )


def metric_row(label: str, value: float, tone: str = "shadow", invert: bool = False) -> str:
    """A labelled proportional bar, as HTML that the web sheet already styles."""
    cls = "bad" if (invert and value > 0.35) else tone
    pct = max(0.0, min(1.0, value)) * 100
    return (f'<div class="sh-metric {cls}"><div class="row"><span>{label}</span>'
            f'<b>{value:.3f}</b></div><div class="track">'
            f'<div class="fill" style="width:{pct:.1f}%"></div></div></div>')


def data_uri(png_b64: str | None) -> str:
    return f"data:image/png;base64,{png_b64}" if png_b64 else ""


def encode_svg(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
