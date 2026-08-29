"""Design-token gateway - the single source of truth for every view.

MVT note: this module is the *T* (Template) layer. No view is allowed to
hard-code a colour, a font name or a radius. Every toolkit (Qt, Tk, Flet,
NiceGUI, DearPyGui) asks this module and receives the same palette in the
format it understands. Change ``tokens.json`` once, all five UIs move.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).resolve().parent
TOKENS_PATH = TEMPLATES_DIR / "tokens.json"


@lru_cache(maxsize=1)
def tokens() -> dict[str, Any]:
    return json.loads(TOKENS_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Primitive accessors
# --------------------------------------------------------------------------- #
def c(name: str) -> str:
    """Colour by token name -> '#RRGGBB'."""
    return tokens()["color"][name]


def font_stack(role: str) -> list[str]:
    return list(tokens()["font"][role])


def font_first(role: str) -> str:
    return tokens()["font"][role][0]


def type_(role: str) -> dict[str, Any]:
    return tokens()["type"][role]


def space(name: str) -> int:
    return tokens()["space"][name]


def radius(name: str) -> int:
    return tokens()["radius"][name]


def rgba(name: str, alpha: float) -> str:
    """Token colour as a CSS/QSS rgba() string."""
    h = c(name).lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:g})"


def rgb_tuple(name: str) -> tuple[int, int, int]:
    h = c(name).lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgba_tuple(name: str, alpha: int = 255) -> tuple[int, int, int, int]:
    r, g, b = rgb_tuple(name)
    return r, g, b, alpha


# --------------------------------------------------------------------------- #
# Qt / PySide6
# --------------------------------------------------------------------------- #
def qss() -> str:
    """Render the QSS stylesheet with token substitution.

    ``qss/app.qss`` is written with ``{{token.path}}`` placeholders so the
    stylesheet stays readable and the palette stays in exactly one file.
    """
    raw = (TEMPLATES_DIR / "qss" / "app.qss").read_text(encoding="utf-8")
    t = tokens()

    def resolve(path: str) -> str:
        node: Any = t
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"unknown token path in app.qss: {{{{{path}}}}}")
            node = node[part]
        if isinstance(node, list):
            return ", ".join(f"'{f}'" if " " in f else f for f in node)
        return str(node)

    out: list[str] = []
    i = 0
    while True:
        start = raw.find("{{", i)
        if start == -1:
            out.append(raw[i:])
            break
        out.append(raw[i:start])
        end = raw.find("}}", start)
        out.append(resolve(raw[start + 2:end].strip()))
        i = end + 2
    return "".join(out)


# --------------------------------------------------------------------------- #
# Web / NiceGUI
# --------------------------------------------------------------------------- #
def _kebab(s: str) -> str:
    return "".join(f"-{ch.lower()}" if ch.isupper() else ch for ch in s)


def css_variables() -> str:
    t = tokens()
    lines = [":root {"]
    for k, v in t["color"].items():
        lines.append(f"  --sh-{_kebab(k)}: {v};")
    for k, v in t["gradient"].items():
        lines.append(f"  --sh-grad-{_kebab(k)}: {v};")
    for k, v in t["font"].items():
        stack = ", ".join(f"'{f}'" if " " in f else f for f in v)
        lines.append(f"  --sh-font-{_kebab(k)}: {stack};")
    for k, v in t["space"].items():
        lines.append(f"  --sh-space-{_kebab(k)}: {v}px;")
    for k, v in t["radius"].items():
        lines.append(f"  --sh-radius-{_kebab(k)}: {v}px;")
    for k, v in t["elevation"].items():
        lines.append(f"  --sh-elev-{_kebab(k)}: {v};")
    for k, v in t["motion"].items():
        suffix = "ms" if isinstance(v, int) else ""
        lines.append(f"  --sh-motion-{_kebab(k)}: {v}{suffix};")
    lines.append("}")
    return "\n".join(lines)


def web_stylesheet() -> str:
    """CSS variables + the hand-written sheet, concatenated."""
    sheet = (TEMPLATES_DIR / "web" / "styles.css").read_text(encoding="utf-8")
    return css_variables() + "\n\n" + sheet


# --------------------------------------------------------------------------- #
# Flet
# --------------------------------------------------------------------------- #
def flet_theme_dict() -> dict[str, Any]:
    return {
        "bg": c("void"),
        "surface": c("panel"),
        "surface_high": c("elevated"),
        "raised": c("raised"),
        "outline": c("hairline"),
        "outline_hot": c("hairlineHot"),
        "on_surface": c("ink"),
        "on_surface_muted": c("inkMuted"),
        "on_surface_faint": c("inkFaint"),
        "primary": c("solar"),
        "primary_deep": c("solarDeep"),
        "secondary": c("shadow"),
        "secondary_deep": c("shadowDeep"),
        "success": c("signal"),
        "error": c("alert"),
        "violet": c("violet"),
        "font_display": font_first("display"),
        "font_body": font_first("body"),
        "font_mono": font_first("mono"),
    }


# --------------------------------------------------------------------------- #
# CustomTkinter
# --------------------------------------------------------------------------- #
def ctk_theme_dict() -> dict[str, Any]:
    """CustomTkinter consumes a JSON theme file; we synthesise it from tokens."""
    solar, shadow = c("solar"), c("shadow")
    return {
        "CTk": {"fg_color": [c("void"), c("void")]},
        "CTkToplevel": {"fg_color": [c("void"), c("void")]},
        "CTkFrame": {
            "corner_radius": radius("md"),
            "border_width": 1,
            "fg_color": [c("panel"), c("panel")],
            "top_fg_color": [c("elevated"), c("elevated")],
            "border_color": [c("hairline"), c("hairline")],
        },
        "CTkButton": {
            "corner_radius": radius("sm"),
            "border_width": 1,
            "fg_color": [c("raised"), c("raised")],
            "hover_color": [c("hairlineHot"), c("hairlineHot")],
            "border_color": [c("hairlineHot"), c("hairlineHot")],
            "text_color": [c("ink"), c("ink")],
            "text_color_disabled": [c("inkFaint"), c("inkFaint")],
        },
        "CTkLabel": {
            "corner_radius": 0,
            "fg_color": "transparent",
            "text_color": [c("ink"), c("ink")],
        },
        "CTkEntry": {
            "corner_radius": radius("sm"),
            "border_width": 1,
            "fg_color": [c("void"), c("void")],
            "border_color": [c("hairline"), c("hairline")],
            "text_color": [c("ink"), c("ink")],
            "placeholder_text_color": [c("inkFaint"), c("inkFaint")],
        },
        "CTkCheckBox": {
            "corner_radius": radius("sm"),
            "border_width": 2,
            "fg_color": [solar, solar],
            "border_color": [c("hairlineHot"), c("hairlineHot")],
            "hover_color": [c("solarDeep"), c("solarDeep")],
            "checkmark_color": [c("void"), c("void")],
            "text_color": [c("ink"), c("ink")],
            "text_color_disabled": [c("inkFaint"), c("inkFaint")],
        },
        "CTkSwitch": {
            "corner_radius": radius("pill"),
            "border_width": 3,
            "button_length": 0,
            "fg_color": [c("hairline"), c("hairline")],
            "progress_color": [shadow, shadow],
            "button_color": [c("ink"), c("ink")],
            "button_hover_color": [solar, solar],
            "text_color": [c("ink"), c("ink")],
            "text_color_disabled": [c("inkFaint"), c("inkFaint")],
        },
        "CTkProgressBar": {
            "corner_radius": radius("pill"),
            "border_width": 0,
            "fg_color": [c("hairline"), c("hairline")],
            "progress_color": [solar, solar],
            "border_color": [c("hairline"), c("hairline")],
        },
        "CTkSlider": {
            "corner_radius": radius("pill"),
            "button_corner_radius": radius("pill"),
            "border_width": 5,
            "button_length": 0,
            "fg_color": [c("hairline"), c("hairline")],
            "progress_color": [shadow, shadow],
            "button_color": [solar, solar],
            "button_hover_color": [c("solarDeep"), c("solarDeep")],
        },
        "CTkOptionMenu": {
            "corner_radius": radius("sm"),
            "fg_color": [c("raised"), c("raised")],
            "button_color": [c("hairlineHot"), c("hairlineHot")],
            "button_hover_color": [c("shadowDeep"), c("shadowDeep")],
            "text_color": [c("ink"), c("ink")],
            "text_color_disabled": [c("inkFaint"), c("inkFaint")],
        },
        "CTkComboBox": {
            "corner_radius": radius("sm"),
            "border_width": 1,
            "fg_color": [c("void"), c("void")],
            "border_color": [c("hairline"), c("hairline")],
            "button_color": [c("hairlineHot"), c("hairlineHot")],
            "button_hover_color": [c("shadowDeep"), c("shadowDeep")],
            "text_color": [c("ink"), c("ink")],
            "text_color_disabled": [c("inkFaint"), c("inkFaint")],
        },
        "CTkScrollbar": {
            "corner_radius": radius("pill"),
            "border_spacing": 4,
            "fg_color": "transparent",
            "button_color": [c("hairlineHot"), c("hairlineHot")],
            "button_hover_color": [c("inkFaint"), c("inkFaint")],
        },
        "CTkScrollableFrame": {"label_fg_color": [c("elevated"), c("elevated")]},
        "CTkTextbox": {
            "corner_radius": radius("sm"),
            "border_width": 1,
            "fg_color": [c("void"), c("void")],
            "border_color": [c("hairline"), c("hairline")],
            "text_color": [c("ink"), c("ink")],
            "scrollbar_button_color": [c("hairlineHot"), c("hairlineHot")],
            "scrollbar_button_hover_color": [c("inkFaint"), c("inkFaint")],
        },
        "CTkSegmentedButton": {
            "corner_radius": radius("sm"),
            "border_width": 1,
            "fg_color": [c("panel"), c("panel")],
            "selected_color": [c("solarWash"), c("solarWash")],
            "selected_hover_color": [c("solarDeep"), c("solarDeep")],
            "unselected_color": [c("panel"), c("panel")],
            "unselected_hover_color": [c("elevated"), c("elevated")],
            "text_color": [c("ink"), c("ink")],
            "text_color_disabled": [c("inkFaint"), c("inkFaint")],
        },
        "DropdownMenu": {
            "fg_color": [c("elevated"), c("elevated")],
            "hover_color": [c("hairlineHot"), c("hairlineHot")],
            "text_color": [c("ink"), c("ink")],
        },
        "CTkFont": {
            "Windows": {"family": font_first("body"), "size": 13, "weight": "normal"},
            "macOS": {"family": font_first("body"), "size": 13, "weight": "normal"},
            "Linux": {"family": font_first("body"), "size": 13, "weight": "normal"},
        },
    }


def write_ctk_theme(path: Path) -> Path:
    """Emit a CustomTkinter theme file.

    Our overrides are merged *over* the packaged ``blue.json`` so that any
    widget key CustomTkinter expects but we do not style (CTkRadioButton,
    future additions) is still present and the toolkit cannot KeyError.
    """
    merged: dict[str, Any] = {}
    try:
        import customtkinter

        base_path = Path(customtkinter.__file__).parent / "assets" / "themes" / "blue.json"
        merged = json.loads(base_path.read_text(encoding="utf-8"))
    except Exception:
        merged = {}

    for widget, values in ctk_theme_dict().items():
        node = merged.setdefault(widget, {})
        if isinstance(node, dict) and isinstance(values, dict):
            node.update(values)
        else:
            merged[widget] = values

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# DearPyGui
# --------------------------------------------------------------------------- #
def dpg_palette() -> dict[str, tuple[int, int, int, int]]:
    return {
        "void": rgba_tuple("void"),
        "panel": rgba_tuple("panel"),
        "elevated": rgba_tuple("elevated"),
        "raised": rgba_tuple("raised"),
        "hairline": rgba_tuple("hairline"),
        "hairline_hot": rgba_tuple("hairlineHot"),
        "ink": rgba_tuple("ink"),
        "ink_muted": rgba_tuple("inkMuted"),
        "ink_faint": rgba_tuple("inkFaint"),
        "solar": rgba_tuple("solar"),
        "solar_deep": rgba_tuple("solarDeep"),
        "shadow": rgba_tuple("shadow"),
        "shadow_deep": rgba_tuple("shadowDeep"),
        "signal": rgba_tuple("signal"),
        "alert": rgba_tuple("alert"),
        "violet": rgba_tuple("violet"),
    }


__all__ = [
    "tokens", "c", "rgba", "rgb_tuple", "rgba_tuple", "font_stack", "font_first",
    "type_", "space", "radius", "qss", "css_variables", "web_stylesheet",
    "flet_theme_dict", "ctk_theme_dict", "write_ctk_theme", "dpg_palette",
]
