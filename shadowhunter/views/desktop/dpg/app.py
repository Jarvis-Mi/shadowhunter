"""DearPyGui telemetry scope.

Role in the suite: the *oscilloscope*. DearPyGui draws immediate-mode plots at
display rate with almost no CPU, so this is the window you leave open on the
second monitor for the whole of a long training run - episode return,
validation MAE and throughput, updating live while PPO grinds.

It is intentionally not a second copy of the deck: no sliders you will never
touch mid-run, no archive browser. Plots, controls to start and stop, and the
current zone.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from ....core.logging import get_logger
from ....services.client import DeckClient, DeckError, decode_png
from ....templates import theme

log = get_logger(__name__)
P = theme.dpg_palette()
TEX = 512                              # dynamic texture side, pixels
FONT_DIR = Path(__file__).resolve().parents[4] / "assets" / "fonts"


def _c(name: str, alpha: int = 255) -> list[int]:
    r, g, b, _ = P[name]
    return [r, g, b, alpha]


class Scope:
    def __init__(self, client: DeckClient) -> None:
        self.client = client
        self.busy = False
        self.active_job: str | None = None
        self.rl: deque[float] = deque(maxlen=600)
        self.cnn: deque[float] = deque(maxlen=600)
        self.fps: deque[float] = deque(maxlen=600)
        self.lock = threading.Lock()
        self.dirty = True
        self.frame = np.zeros((TEX, TEX, 4), np.float32)
        self.frame[:, :, 3] = 1.0

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #
    def _theme(self) -> int:
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                for target, colour in (
                    (dpg.mvThemeCol_WindowBg, "void"),
                    (dpg.mvThemeCol_ChildBg, "panel"),
                    (dpg.mvThemeCol_PopupBg, "elevated"),
                    (dpg.mvThemeCol_Border, "hairline"),
                    (dpg.mvThemeCol_FrameBg, "elevated"),
                    (dpg.mvThemeCol_FrameBgHovered, "raised"),
                    (dpg.mvThemeCol_FrameBgActive, "hairline_hot"),
                    (dpg.mvThemeCol_TitleBg, "panel"),
                    (dpg.mvThemeCol_TitleBgActive, "elevated"),
                    (dpg.mvThemeCol_Text, "ink"),
                    (dpg.mvThemeCol_TextDisabled, "ink_faint"),
                    (dpg.mvThemeCol_Button, "raised"),
                    (dpg.mvThemeCol_ButtonHovered, "hairline_hot"),
                    (dpg.mvThemeCol_ButtonActive, "shadow_deep"),
                    (dpg.mvThemeCol_Header, "elevated"),
                    (dpg.mvThemeCol_HeaderHovered, "hairline_hot"),
                    (dpg.mvThemeCol_SliderGrab, "solar"),
                    (dpg.mvThemeCol_SliderGrabActive, "solar_deep"),
                    (dpg.mvThemeCol_CheckMark, "solar"),
                    (dpg.mvThemeCol_Separator, "hairline"),
                    (dpg.mvThemeCol_PlotHistogram, "solar"),
                    (dpg.mvThemeCol_TableHeaderBg, "void"),
                    (dpg.mvThemeCol_TableBorderStrong, "hairline_hot"),
                    (dpg.mvThemeCol_TableBorderLight, "hairline"),
                    (dpg.mvThemeCol_TableRowBgAlt, "elevated"),
                    (dpg.mvThemeCol_ScrollbarBg, "void"),
                    (dpg.mvThemeCol_ScrollbarGrab, "hairline_hot"),
                ):
                    dpg.add_theme_color(target, _c(colour), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 14, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 9, 8, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1, category=dpg.mvThemeCat_Core)

            with dpg.theme_component(dpg.mvAll):
                for target, colour in ((dpg.mvPlotCol_FrameBg, "void"),
                                       (dpg.mvPlotCol_PlotBg, "void"),
                                       (dpg.mvPlotCol_PlotBorder, "hairline"),
                                       (dpg.mvPlotCol_AxisText, "ink_faint"),
                                       (dpg.mvPlotCol_AxisGrid, "hairline"),
                                       (dpg.mvPlotCol_LegendBg, "elevated"),
                                       (dpg.mvPlotCol_LegendText, "ink_muted")):
                    dpg.add_theme_color(target, _c(colour), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.8, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 8, 6, category=dpg.mvThemeCat_Plots)
        return t

    @staticmethod
    def _series_theme(colour: str) -> int:
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, _c(colour), category=dpg.mvThemeCat_Plots)
            with dpg.theme_component(dpg.mvShadeSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, _c(colour, 46), category=dpg.mvThemeCat_Plots)
        return t

    def _accent_button_theme(self) -> int:
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _c("solar"))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _c("solar_deep"))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _c("solar_deep"))
                dpg.add_theme_color(dpg.mvThemeCol_Text, _c("void"))
        return t

    def _load_fonts(self) -> tuple[int | None, int | None]:
        if not FONT_DIR.exists():
            return None, None
        mono = display = None
        with dpg.font_registry():
            for path in FONT_DIR.glob("*.[to]tf"):
                stem = path.stem.lower()
                if "mono" in stem and mono is None:
                    mono = dpg.add_font(str(path), 15)
                elif display is None and "mono" not in stem:
                    display = dpg.add_font(str(path), 22)
        return mono, display

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def build(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Shadow Hunter - Telemetry Scope", width=1480, height=900,
                            clear_color=[c / 255 for c in _c("void")[:3]] + [1.0])
        mono_font, display_font = self._load_fonts()

        with dpg.texture_registry():
            dpg.add_raw_texture(TEX, TEX, self.frame.ravel(), tag="scene_tex",
                                format=dpg.mvFormat_Float_rgba)

        with dpg.window(tag="root", no_title_bar=True, no_move=True, no_resize=True,
                        no_collapse=True):
            with dpg.group(horizontal=True):
                dpg.add_text("SHADOW", tag="mark_a", color=_c("ink"))
                dpg.add_text("HUNTER", tag="mark_b", color=_c("solar"))
                dpg.add_text("|", color=_c("hairline_hot"))
                dpg.add_text("telemetry scope  ·  dearpygui", color=_c("ink_faint"))
                dpg.add_spacer(width=30)
                dpg.add_text("connecting...", tag="health", color=_c("ink_muted"))
            if display_font:
                dpg.bind_item_font("mark_a", display_font)
                dpg.bind_item_font("mark_b", display_font)
            dpg.add_separator()

            with dpg.group(horizontal=True):
                # ---------------- left: viewport + controls ----------------
                with dpg.child_window(width=560, border=True):
                    dpg.add_text("CURRENT ZONE", color=_c("ink_faint"))
                    dpg.add_image("scene_tex", width=524, height=524)
                    dpg.add_text("", tag="scene_note", color=_c("ink_muted"), wrap=520)
                    dpg.add_separator()

                    with dpg.group(horizontal=True):
                        dpg.add_combo(("auto", "learned", "greedy"), default_value="auto",
                                      tag="policy", width=110)
                        dpg.add_slider_int(label="buildings", tag="buildings", default_value=16,
                                           min_value=3, max_value=40, width=170)
                    with dpg.group(horizontal=True):
                        dpg.add_slider_int(label="sun elev", tag="elev", default_value=34,
                                           min_value=8, max_value=80, width=170)
                        dpg.add_slider_int(label="azimuth", tag="azim", default_value=148,
                                           min_value=0, max_value=359, width=170)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="RUN HUNT", tag="btn_hunt", width=150, height=32,
                                       callback=lambda: self._spawn(self._hunt))
                        dpg.add_button(label="TRAIN POLICY", width=150, height=32,
                                       callback=lambda: self._spawn(self._train_rl))
                        dpg.add_button(label="TRAIN CNN", width=140, height=32,
                                       callback=lambda: self._spawn(self._train_cnn))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="ABORT", width=150, height=28,
                                       callback=lambda: self._spawn(self._abort))
                        dpg.add_progress_bar(tag="job_bar", default_value=0.0, width=290)
                    dpg.add_text("no run active", tag="job_line", color=_c("ink_faint"))

                # ---------------- right: the scope ------------------------
                with dpg.child_window(border=True):
                    dpg.add_text("EPISODE RETURN  ·  ZONE POLICY", color=_c("ink_faint"))
                    with dpg.plot(height=210, width=-1, no_mouse_pos=False, tag="plot_rl"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="update", tag="rl_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="return", tag="rl_y"):
                            dpg.add_shade_series([0.0], [0.0], tag="rl_fill")
                            dpg.add_line_series([0.0], [0.0], tag="rl_line")

                    dpg.add_text("VALIDATION MAE (METRES)  ·  HEIGHT REGRESSOR", color=_c("ink_faint"))
                    with dpg.plot(height=210, width=-1, tag="plot_cnn"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="cnn_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="MAE m", tag="cnn_y"):
                            dpg.add_line_series([0.0], [0.0], tag="cnn_line")

                    dpg.add_text("THROUGHPUT (ENV STEPS / SECOND)", color=_c("ink_faint"))
                    with dpg.plot(height=170, width=-1, tag="plot_fps"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="update", tag="fps_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="fps", tag="fps_y"):
                            dpg.add_line_series([0.0], [0.0], tag="fps_line")

                    dpg.add_separator()
                    dpg.add_text("ZONE METRICS", color=_c("ink_faint"))
                    with dpg.table(header_row=True, borders_innerH=False, borders_outerH=False,
                                   borders_innerV=False, borders_outerV=False, tag="metrics_table"):
                        dpg.add_table_column(label="METRIC")
                        dpg.add_table_column(label="VALUE")
                        dpg.add_table_column(label="BAR", width_stretch=True)
                        for key, label in (("contrast", "shadow contrast"),
                                           ("isolation", "blob isolation"),
                                           ("edge_coherence", "edge coherence"),
                                           ("axis_alignment", "azimuth alignment"),
                                           ("occlusion", "occlusion"),
                                           ("truncation", "truncation"),
                                           ("r1_contrast", "R1 contrast·isolation"),
                                           ("r2_structure", "R2 edge·entropy"),
                                           ("r3_azimuth", "R3 azimuth coherence")):
                            with dpg.table_row():
                                dpg.add_text(label, color=_c("ink_muted"))
                                dpg.add_text("0.000", tag=f"val_{key}", color=_c("shadow"))
                                dpg.add_progress_bar(tag=f"bar_{key}", default_value=0.0, width=-1)

        dpg.bind_theme(self._theme())
        if mono_font:
            dpg.bind_font(mono_font)
        dpg.bind_item_theme("rl_line", self._series_theme("solar"))
        dpg.bind_item_theme("rl_fill", self._series_theme("solar"))
        dpg.bind_item_theme("cnn_line", self._series_theme("shadow"))
        dpg.bind_item_theme("fps_line", self._series_theme("signal"))
        dpg.bind_item_theme("btn_hunt", self._accent_button_theme())

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("root", True)

    # ------------------------------------------------------------------ #
    # Work
    # ------------------------------------------------------------------ #
    @staticmethod
    def _spawn(fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def _sun(self) -> dict[str, float]:
        return {"azimuth_deg": float(dpg.get_value("azim")),
                "elevation_deg": float(dpg.get_value("elev")), "gsd_m": 0.5}

    def _hunt(self) -> None:
        if self.busy:
            return
        self.busy = True
        try:
            p = self.client.analyze(size=512, buildings=int(dpg.get_value("buildings")),
                                    sun=self._sun(), policy=dpg.get_value("policy"))
        except DeckError as exc:
            dpg.set_value("scene_note", str(exc))
            self.busy = False
            return
        self.busy = False
        self._push_frame(decode_png(p.get("overlay_png")))

        h = p["height"]
        cnn = h["cnn_m"]
        dpg.set_value("scene_note",
                      f"h = {h['fused_m']:.1f} m  +/- {h['sigma_m']:.2f}  ({h['floors']} storeys)   "
                      f"policy {p['policy']} · {p['steps']} steps · score {p['score']:+.3f}\n"
                      f"geometric {h['geometric_m']:.1f} m · "
                      f"cnn {f'{cnn:.1f} m' if cnn else 'not loaded'} · "
                      f"confidence {h['confidence'] * 100:.0f}% · {p['elapsed_ms']:.0f} ms")

        for key in ("contrast", "isolation", "edge_coherence", "axis_alignment",
                    "occlusion", "truncation"):
            value = float(p["metrics"].get(key, 0.0))
            bad = key in {"occlusion", "truncation"} and value > 0.35
            dpg.set_value(f"val_{key}", f"{value:.3f}")
            dpg.configure_item(f"val_{key}", color=_c("alert" if bad else "shadow"))
            dpg.set_value(f"bar_{key}", value)
        for key in ("r1_contrast", "r2_structure", "r3_azimuth"):
            value = float(p["breakdown"].get(key, 0.0))
            dpg.set_value(f"val_{key}", f"{value:.3f}")
            dpg.configure_item(f"val_{key}", color=_c("violet"))
            dpg.set_value(f"bar_{key}", value)

    def _push_frame(self, bgr: np.ndarray | None) -> None:
        """Blit a BGR image into the fixed-size RGBA float texture, letterboxed."""
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        k = min(TEX / w, TEX / h)
        dw, dh = max(1, int(w * k)), max(1, int(h * k))
        small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((TEX, TEX, 4), np.float32)
        canvas[:, :, 3] = 1.0
        ox, oy = (TEX - dw) // 2, (TEX - dh) // 2
        canvas[oy:oy + dh, ox:ox + dw, :3] = small[:, :, ::-1].astype(np.float32) / 255.0
        with self.lock:
            self.frame = canvas
            self.dirty = True

    def _train_rl(self) -> None:
        try:
            job = self.client.train_rl(algo="PPO", total_timesteps=20_000, tag="shadow_hunter")
            self.active_job = job["id"]
            self.rl.clear()
        except DeckError as exc:
            dpg.set_value("job_line", str(exc))

    def _train_cnn(self) -> None:
        try:
            job = self.client.train_cnn(scenes=24, epochs=20, tag="height_cnn")
            self.active_job = job["id"]
            self.cnn.clear()
        except DeckError as exc:
            dpg.set_value("job_line", str(exc))

    def _abort(self) -> None:
        if self.active_job:
            try:
                self.client.abort(self.active_job)
            except DeckError:
                pass

    # ------------------------------------------------------------------ #
    # Polling thread
    # ------------------------------------------------------------------ #
    def _poll_loop(self) -> None:
        seen: set[str] = set()
        while True:
            time.sleep(0.5)
            try:
                events = self.client.telemetry(limit=60)
            except DeckError:
                continue
            for e in events:
                key = f"{e.get('ts')}|{e.get('topic')}|{e.get('timesteps', e.get('epoch', ''))}"
                if key in seen:
                    continue
                seen.add(key)
                if e["topic"] == "rl.progress" and e.get("ep_reward_mean") is not None:
                    self.rl.append(float(e["ep_reward_mean"]))
                    self.fps.append(float(e.get("fps", 0.0)))
                elif e["topic"] == "cnn.progress":
                    self.cnn.append(float(e["val_mae"]))
            if len(seen) > 3000:
                seen.clear()

            if self.active_job:
                try:
                    job = self.client.job(self.active_job)
                except DeckError:
                    continue
                dpg.set_value("job_bar", job["progress"])
                dpg.set_value("job_line", f"{job['id']} · {job['kind']} · "
                                          f"{job['state']} · {job['message']}")
                if job["state"] in {"done", "failed", "aborted"}:
                    self.active_job = None
                    self._sync_health()

    def _sync_health(self) -> None:
        try:
            h = self.client.health()
        except DeckError:
            dpg.set_value("health", "backend unreachable")
            return
        policy = "policy OK" if h["policy_loaded"] else "policy: greedy"
        cnn = "cnn OK" if h["cnn_loaded"] else "cnn: geometric"
        dpg.set_value("health", f"{'CUDA' if h['cuda'] else 'CPU'} · torch {h['torch']} · "
                                f"{policy} · {cnn}")

    # ------------------------------------------------------------------ #
    def run(self) -> int:
        self.build()
        threading.Thread(target=self._poll_loop, daemon=True, name="scope-poll").start()
        self._spawn(self._sync_health)
        self._spawn(self._hunt)

        while dpg.is_dearpygui_running():
            with self.lock:
                if self.dirty:
                    dpg.set_value("scene_tex", self.frame.ravel())
                    self.dirty = False

            if self.rl:
                xs = list(range(len(self.rl)))
                ys = list(self.rl)
                floor = min(ys) - 0.05 * (max(ys) - min(ys) + 1e-6)
                dpg.set_value("rl_line", [xs, ys])
                dpg.set_value("rl_fill", [xs, ys, [floor] * len(xs)])
                dpg.fit_axis_data("rl_x")
                dpg.fit_axis_data("rl_y")
            if self.cnn:
                dpg.set_value("cnn_line", [list(range(1, len(self.cnn) + 1)), list(self.cnn)])
                dpg.fit_axis_data("cnn_x")
                dpg.fit_axis_data("cnn_y")
            if self.fps:
                dpg.set_value("fps_line", [list(range(len(self.fps))), list(self.fps)])
                dpg.fit_axis_data("fps_x")
                dpg.fit_axis_data("fps_y")

            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        return 0


def main(base_url: str | None = None) -> int:
    if base_url:
        client = DeckClient(base_url=base_url)
    else:
        from ....services.supervisor import serve_in_thread

        client = serve_in_thread()
    return Scope(client).run()


if __name__ == "__main__":
    raise SystemExit(main())
