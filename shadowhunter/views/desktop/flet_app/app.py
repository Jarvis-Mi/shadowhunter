"""Flet mission control.

Role in the suite: the *portable* console. The same file runs as a native
Windows window or, with ``--web``, as a browser app on a tablet next to the
workstation. Layout is deliberately card-first and touch-sized.

Written against Flet's stable primitives (Container / Column / Row / Text /
Image / Slider / Dropdown / ProgressBar / DataTable) so it survives the
toolkit's fast-moving high-level widgets. Buttons are hand-built Containers -
which is also how they end up matching the rest of the design system exactly.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import flet as ft

from ....core.config import SETTINGS
from ....core.geo import SunGeometry, quality_of_geometry
from ....core.logging import get_logger
from ....services.client import DeckClient, DeckError, png_bytes
from ....templates import theme

log = get_logger(__name__)
T = theme.flet_theme_dict()
CACHE = SETTINGS.data_dir / ".flet"


# --------------------------------------------------------------------------- #
# Design-system primitives
# --------------------------------------------------------------------------- #
def mono(value: str, size: int = 12, color: str | None = None, weight=None) -> ft.Text:
    return ft.Text(value, size=size, color=color or T["on_surface_muted"],
                   font_family=T["font_mono"], weight=weight)


def section(text: str) -> ft.Text:
    return ft.Text(text.upper(), size=10, color=T["on_surface_faint"],
                   font_family=T["font_mono"], weight=ft.FontWeight.BOLD)


def card(*controls: ft.Control, accent: bool = False, expand: bool | int = False,
         padding: int = 18) -> ft.Container:
    return ft.Container(
        content=ft.Column(list(controls), spacing=10, tight=True),
        bgcolor=T["surface"],
        border=ft.Border.all(1, T["primary"] if accent else T["outline"]),
        border_radius=6,
        padding=padding,
        expand=expand,
    )


def button(label: str, on_click: Callable, kind: str = "ghost",
           expand: bool = False) -> ft.Container:
    """Hand-built button - the only way to get exact token colours in Flet."""
    fill = {"primary": T["primary"], "ghost": "transparent", "danger": "transparent"}[kind]
    fg = {"primary": T["bg"], "ghost": T["on_surface_muted"], "danger": T["error"]}[kind]
    edge = {"primary": T["primary"], "ghost": T["outline_hot"], "danger": T["error"]}[kind]

    text = ft.Text(label.upper(), size=11, color=fg, font_family=T["font_mono"],
                   weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER)
    container = ft.Container(
        content=text, bgcolor=fill, border=ft.Border.all(1, edge), border_radius=3,
        padding=ft.Padding.symmetric(vertical=12, horizontal=18), alignment=ft.Alignment.CENTER,
        on_click=on_click, expand=expand, animate=140,
    )

    def hover(e: ft.ControlEvent) -> None:
        hot = e.data == "true"
        if kind == "primary":
            container.bgcolor = T["primary_deep"] if hot else T["primary"]
        elif kind == "danger":
            container.bgcolor = T["error"] if hot else "transparent"
            text.color = T["bg"] if hot else T["error"]
        else:
            container.border = ft.Border.all(1, T["secondary"] if hot else T["outline_hot"])
            text.color = T["secondary"] if hot else T["on_surface_muted"]
        container.update()

    container.on_hover = hover
    return container


class Readout(ft.Column):
    """Big mono number + unit. Mirrors the Qt deck's Readout exactly."""

    def __init__(self, caption: str, unit: str, tone: str = "on_surface") -> None:
        self._value = ft.Text("--", size=30, color=T[tone], font_family=T["font_mono"],
                              weight=ft.FontWeight.W_600)
        self._unit = ft.Text(unit, size=10, color=T["on_surface_faint"], font_family=T["font_mono"])
        super().__init__([section(caption), self._value, self._unit], spacing=2, tight=True)

    def set(self, value: str, unit: str | None = None) -> None:
        self._value.value = value
        if unit is not None:
            self._unit.value = unit


class MetricBar(ft.Column):
    """Label, mono value, proportional track."""

    def __init__(self, label: str, tone: str = "secondary", invert: bool = False) -> None:
        self.tone, self.invert = tone, invert
        self._label = ft.Text(label.upper(), size=10, color=T["on_surface_muted"],
                              font_family=T["font_mono"])
        self._value = ft.Text("0.000", size=10, color=T[tone], font_family=T["font_mono"],
                              weight=ft.FontWeight.BOLD)
        self._bar = ft.ProgressBar(value=0.0, bgcolor=T["outline"], color=T[tone],
                                   height=3, border_radius=2)
        super().__init__([
            ft.Row([self._label, self._value],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            self._bar,
        ], spacing=4, tight=True)

    def set(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        tone = T["error"] if (self.invert and value > 0.35) else T[self.tone]
        self._value.value = f"{value:.3f}"
        self._value.color = tone
        self._bar.color = tone
        self._bar.value = value


# --------------------------------------------------------------------------- #
# The console
# --------------------------------------------------------------------------- #
class MissionControl:
    def __init__(self, page: ft.Page, client: DeckClient) -> None:
        self.page = page
        self.client = client
        self.busy = False
        self.active_job: str | None = None
        self.frame = 0
        CACHE.mkdir(parents=True, exist_ok=True)

        page.title = "Shadow Hunter · Mission Control"
        page.bgcolor = T["bg"]
        page.padding = 0
        page.theme_mode = ft.ThemeMode.DARK
        page.scroll = ft.ScrollMode.AUTO
        page.fonts = {}

        self._build()
        threading.Thread(target=self._poll_loop, daemon=True, name="flet-poll").start()
        threading.Thread(target=self._boot, daemon=True, name="flet-boot").start()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        self.scene_image = ft.Image(src="", fit=ft.BoxFit.CONTAIN, expand=True,
                                    gapless_playback=True)
        self.scene_placeholder = ft.Container(
            content=mono("NO SCENE LOADED", 11, T["on_surface_faint"]),
            alignment=ft.Alignment.CENTER, height=300)
        self.viewport = ft.Container(
            content=self.scene_placeholder, bgcolor=T["bg"],
            border=ft.Border.all(1, T["outline_hot"]), border_radius=3,
            padding=6, expand=True, height=430)

        self.scene_note = mono("", 11)

        self.in_size = ft.Slider(min=256, max=1024, divisions=12, value=512, label="{value} px",
                                 active_color=T["secondary"], thumb_color=T["primary"])
        self.in_buildings = ft.Slider(min=3, max=40, divisions=37, value=16, label="{value}",
                                      active_color=T["secondary"], thumb_color=T["primary"])
        self.in_elev = ft.Slider(min=8, max=80, divisions=72, value=34, label="{value}°",
                                 active_color=T["secondary"], thumb_color=T["primary"],
                                 on_change=lambda e: self._sun_note())
        self.in_azim = ft.Slider(min=0, max=359, divisions=359, value=148, label="{value}°",
                                 active_color=T["secondary"], thumb_color=T["primary"],
                                 on_change=lambda e: self._sun_note())
        self.in_policy = ft.Dropdown(
            value="auto", width=150, text_size=12, border_color=T["outline"],
            color=T["on_surface"], bgcolor=T["surface_high"],
            options=[ft.DropdownOption("auto"), ft.DropdownOption("learned"),
                     ft.DropdownOption("greedy")])

        self.r_height = Readout("estimated height", "metres above ground", "primary")
        self.r_floors = Readout("storeys", "at 3.2 m per floor")
        self.r_sigma = Readout("uncertainty", "1σ metres", "secondary")
        self.r_conf = Readout("zone confidence", "occlusion · truncation · sun", "success")

        self.bars = {
            "contrast": MetricBar("shadow contrast", "secondary"),
            "isolation": MetricBar("blob isolation", "secondary"),
            "edge_coherence": MetricBar("edge coherence", "primary"),
            "axis_alignment": MetricBar("azimuth alignment", "primary"),
            "occlusion": MetricBar("occlusion", "secondary", invert=True),
            "truncation": MetricBar("truncation", "secondary", invert=True),
        }
        self.reward_bars = {
            "r1_contrast": MetricBar("R1 contrast · isolation", "violet"),
            "r2_structure": MetricBar("R2 edge · entropy", "violet"),
            "r3_azimuth": MetricBar("R3 azimuth coherence", "violet"),
        }

        self.sun_text = mono("", 11)
        self.job_text = mono("idle", 11)
        self.job_bar = ft.ProgressBar(value=0, bgcolor=T["outline"], color=T["primary"], height=3)
        self.status_text = mono("connecting…", 11)

        self.rl_steps = ft.TextField(value="20000", label="TIMESTEPS", dense=True, width=140,
                                     text_size=12, color=T["on_surface"],
                                     border_color=T["outline"], label_style=ft.TextStyle(size=10))
        self.rl_algo = ft.Dropdown(value="PPO", width=110, text_size=12,
                                   border_color=T["outline"], color=T["on_surface"],
                                   bgcolor=T["surface_high"],
                                   options=[ft.DropdownOption("PPO"), ft.DropdownOption("DQN")])
        self.cnn_scenes = ft.TextField(value="24", label="SCENES", dense=True, width=110,
                                       text_size=12, color=T["on_surface"],
                                       border_color=T["outline"], label_style=ft.TextStyle(size=10))
        self.cnn_epochs = ft.TextField(value="20", label="EPOCHS", dense=True, width=110,
                                       text_size=12, color=T["on_surface"],
                                       border_color=T["outline"], label_style=ft.TextStyle(size=10))

        header = ft.Container(
            content=ft.Row([
                ft.Text("SHADOW", size=22, color=T["on_surface"],
                        font_family=T["font_display"], weight=ft.FontWeight.BOLD),
                ft.Text("HUNTER", size=22, color=T["primary"],
                        font_family=T["font_display"], weight=ft.FontWeight.BOLD),
                ft.VerticalDivider(width=20, color=T["outline_hot"]),
                mono("mission control  ·  flet", 11),
                ft.Container(expand=True),
                self.status_text,
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.symmetric(vertical=16, horizontal=26),
            bgcolor=T["surface"],
            border=ft.Border.only(bottom=ft.BorderSide(1, T["outline"])),
        )

        controls = card(
            section("acquisition parameters"),
            ft.Row([
                ft.Column([section("tile size"), self.in_size,
                           section("buildings"), self.in_buildings], expand=1, spacing=0),
                ft.Column([section("sun elevation"), self.in_elev,
                           section("sun azimuth"), self.in_azim], expand=1, spacing=0),
                ft.Column([section("policy"), self.in_policy, self.sun_text],
                          expand=1, spacing=8),
            ], vertical_alignment=ft.CrossAxisAlignment.START),
            ft.Row([button("run hunt", self.run_hunt, "primary", expand=True),
                    button("sweep scene", self.run_sweep, "ghost", expand=True)], spacing=10),
        )

        training = card(
            section("training"),
            ft.Row([self.rl_algo, self.rl_steps,
                    button("train policy", self.train_rl, "primary")], spacing=10),
            ft.Row([self.cnn_scenes, self.cnn_epochs,
                    button("train regressor", self.train_cnn, "ghost")], spacing=10),
            ft.Divider(height=1, color=T["outline"]),
            self.job_text, self.job_bar,
            button("abort run", self.abort, "danger"),
        )

        left = ft.Column([
            card(section("scene viewport"), self.viewport, self.scene_note, expand=True),
            controls,
            training,
        ], spacing=14, expand=3)

        right = ft.Column([
            card(self.r_height, ft.Divider(height=1, color=T["outline"]),
                 ft.Row([self.r_floors, self.r_sigma],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                 self.r_conf, accent=True),
            card(section("zone metrics"), *self.bars.values()),
            card(section("proxy reward breakdown"), *self.reward_bars.values(),
                 mono("R1 contrast·isolation · R2 edge·entropy · R3 azimuth coherence. "
                      "None of the three reads a ground-truth height.", 10)),
        ], spacing=14, width=330, scroll=ft.ScrollMode.AUTO)

        self.page.add(header, ft.Container(
            content=ft.Row([left, right], vertical_alignment=ft.CrossAxisAlignment.START,
                           spacing=14),
            padding=18, expand=True))
        self._sun_note()

    # ------------------------------------------------------------------ #
    # Behaviour
    # ------------------------------------------------------------------ #
    def _sun(self) -> dict[str, float]:
        return {"azimuth_deg": float(self.in_azim.value),
                "elevation_deg": float(self.in_elev.value), "gsd_m": 0.5}

    def _sun_note(self) -> None:
        sun = SunGeometry(float(self.in_azim.value), float(self.in_elev.value), 0.5)
        q = quality_of_geometry(sun)
        verdict = ("ideal metrology" if q > 0.66 else
                   "usable geometry" if q > 0.33 else "poor — shadows too short or merging")
        self.sun_text.value = (f"shadow cast toward {sun.shadow_bearing_deg:.0f}°\n"
                               f"geometry quality {q:.2f} — {verdict}")
        self.sun_text.color = (T["success"] if q > 0.66 else
                               T["primary"] if q > 0.33 else T["error"])
        self._safe_update()

    def _safe_update(self) -> None:
        try:
            self.page.update()
        except Exception:
            pass

    def _run_async(self, fn: Callable[[], Any], on_done: Callable[[Any], None]) -> None:
        def worker() -> None:
            try:
                result = fn()
            except DeckError as exc:
                self.busy = False
                self.status_text.value = str(exc)[:120]
                self.status_text.color = T["error"]
                self._safe_update()
                return
            except Exception as exc:
                self.busy = False
                self.status_text.value = f"{type(exc).__name__}: {exc}"[:120]
                self.status_text.color = T["error"]
                self._safe_update()
                return
            on_done(result)
        threading.Thread(target=worker, daemon=True).start()

    def _boot(self) -> None:
        self.client.wait_until_ready()
        self._sync_health()
        self.run_hunt(None)

    def run_hunt(self, _e: Any = None) -> None:
        if self.busy:
            return
        self.busy = True
        self.status_text.value = "hunting for a clean shadow zone…"
        self.status_text.color = T["primary"]
        self._safe_update()
        self._run_async(
            lambda: self.client.analyze(
                size=int(self.in_size.value), buildings=int(self.in_buildings.value),
                sun=self._sun(), policy=self.in_policy.value, max_steps=48),
            self._on_hunt)

    def run_sweep(self, _e: Any = None) -> None:
        if self.busy:
            return
        self.busy = True
        self.status_text.value = "sweeping the tile…"
        self._safe_update()
        self._run_async(
            lambda: self.client.sweep(
                size=int(self.in_size.value), buildings=int(self.in_buildings.value),
                sun=self._sun(), policy=self.in_policy.value,
                limit=int(self.in_buildings.value)),
            self._on_sweep)

    def _show_png(self, b64: str | None) -> None:
        """Write the frame into the assets dir and point the Image at it.

        Flet serves ``src`` relative to ``assets_dir``; filenames are cycled
        so the client-side image cache can never hand back a stale frame.
        """
        data = png_bytes(b64)
        if not data:
            return
        self.frame = (self.frame + 1) % 6
        (CACHE / f"scene_{self.frame}.png").write_bytes(data)
        self.scene_image.src = f"/scene_{self.frame}.png"
        self.viewport.content = self.scene_image

    def _on_hunt(self, p: dict[str, Any]) -> None:
        self.busy = False
        self._show_png(p.get("overlay_png"))
        h = p["height"]
        self.r_height.set(f"{h['fused_m']:.1f}", "metres above ground")
        self.r_floors.set(str(h["floors"]), "at 3.2 m per floor")
        self.r_sigma.set(f"{h['sigma_m']:.2f}", "1σ metres")
        self.r_conf.set(f"{h['confidence'] * 100:.0f}%", "occlusion · truncation · sun")

        for key, bar in self.bars.items():
            bar.set(p["metrics"].get(key, 0.0))
        for key, bar in self.reward_bars.items():
            bar.set(p["breakdown"].get(key, 0.0))

        cnn = h["cnn_m"]
        self.scene_note.value = (
            f"policy {p['policy']} · {p['steps']} steps · score {p['score']:+.3f} · "
            f"zone {p['box']} · geometric {h['geometric_m']:.1f} m · "
            f"cnn {f'{cnn:.1f} m' if cnn else 'not loaded'} · {p['elapsed_ms']:.0f} ms")
        self.status_text.value = "nominal"
        self.status_text.color = T["success"]
        self._safe_update()

    def _on_sweep(self, p: dict[str, Any]) -> None:
        self.busy = False
        self._show_png(p.get("overlay_png"))
        self.r_height.set(f"{p['mae_m']:.2f}" if p["mae_m"] else "--", "MAE across tile, metres")
        self.r_sigma.set(f"{p['rmse_m']:.2f}" if p["rmse_m"] else "--", "RMSE metres")
        self.r_floors.set(str(len(p["items"])), "buildings measured")
        self.r_conf.set(f"{p['mean_score'] * 100:.0f}%", "mean proxy score")
        self.scene_note.value = f"swept {len(p['items'])} buildings in {p['elapsed_ms']:.0f} ms"
        self.status_text.value = "sweep complete"
        self.status_text.color = T["success"]
        self._safe_update()

    def train_rl(self, _e: Any = None) -> None:
        self._run_async(
            lambda: self.client.train_rl(algo=self.rl_algo.value,
                                         total_timesteps=int(self.rl_steps.value),
                                         tag="shadow_hunter"),
            self._on_job)

    def train_cnn(self, _e: Any = None) -> None:
        self._run_async(
            lambda: self.client.train_cnn(scenes=int(self.cnn_scenes.value),
                                          epochs=int(self.cnn_epochs.value),
                                          tag="height_cnn"),
            self._on_job)

    def _on_job(self, job: dict[str, Any]) -> None:
        self.active_job = job["id"]
        self.job_text.value = f"{job['id']} · {job['kind']} · queued"
        self._safe_update()

    def abort(self, _e: Any = None) -> None:
        if self.active_job:
            self._run_async(lambda: self.client.abort(self.active_job),
                            lambda _: None)

    def _sync_health(self) -> None:
        try:
            h = self.client.health()
        except DeckError:
            self.status_text.value = "backend unreachable"
            self.status_text.color = T["error"]
            self._safe_update()
            return
        policy = "policy ✓" if h["policy_loaded"] else "policy · greedy"
        cnn = "cnn ✓" if h["cnn_loaded"] else "cnn · geometric"
        self.status_text.value = (f"{'CUDA' if h['cuda'] else 'CPU'} · torch {h['torch']} · "
                                  f"{policy} · {cnn}")
        self.status_text.color = T["on_surface_muted"]
        self._safe_update()

    def _poll_loop(self) -> None:
        while True:
            time.sleep(1.1)
            if not self.active_job:
                continue
            try:
                job = self.client.job(self.active_job)
            except DeckError:
                continue
            self.job_text.value = f"{job['id']} · {job['kind']} · {job['state']} · {job['message']}"
            self.job_bar.value = job["progress"]
            if job["state"] in {"done", "failed", "aborted"}:
                self.active_job = None
                self._sync_health()
            self._safe_update()


# --------------------------------------------------------------------------- #
def main(base_url: str | None = None, web: bool = False) -> int:
    if base_url:
        client = DeckClient(base_url=base_url)
    else:
        from ....services.supervisor import serve_in_thread

        # Wait for the backend *before* Flet spawns its window: the Flutter
        # host and uvicorn otherwise fight for the GIL during startup and the
        # first hunt can sit behind a 40-second handshake.
        client = serve_in_thread()

    def target(page: ft.Page) -> None:
        MissionControl(page, client)

    view = ft.AppView.WEB_BROWSER if web else ft.AppView.FLET_APP
    runner = getattr(ft, "run", None) or ft.app
    runner(target, view=view, assets_dir=str(CACHE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
