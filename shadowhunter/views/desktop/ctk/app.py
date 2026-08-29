"""CustomTkinter field console.

Role in the suite: the *light* client. Starts in well under a second, has no
Qt or Chromium behind it, and fits on a 1366x768 field laptop. Everything an
operator needs to take one measurement and read the numbers - nothing else.

The scene view and the metric gauges are drawn on a raw ``tkinter.Canvas``,
because that is the only way to get the reticle and the hairline instrument
look inside Tk.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from typing import Any, Callable

import customtkinter as ctk
from PIL import Image, ImageTk

from ....core.config import SETTINGS
from ....core.geo import SunGeometry, quality_of_geometry
from ....core.logging import get_logger
from ....services.client import DeckClient, DeckError, decode_png
from ....templates import theme

log = get_logger(__name__)
THEME_FILE = SETTINGS.data_dir / ".ctk_orbital_dusk.json"

C = {k: theme.c(k) for k in ("void", "panel", "elevated", "raised", "hairline", "hairlineHot",
                             "ink", "inkMuted", "inkFaint", "solar", "solarDeep", "shadow",
                             "shadowDeep", "signal", "alert", "violet")}
FONT_MONO = theme.font_first("mono")
FONT_BODY = theme.font_first("body")
FONT_DISPLAY = theme.font_first("display")


# --------------------------------------------------------------------------- #
# Canvas instruments
# --------------------------------------------------------------------------- #
class SceneView(tk.Canvas):
    """Scene bitmap + reticle + search trail, drawn by hand."""

    def __init__(self, master: Any, **kw: Any) -> None:
        super().__init__(master, bg=C["void"], highlightthickness=0, bd=0, **kw)
        self._photo: ImageTk.PhotoImage | None = None
        self._bgr = None
        self._box = None
        self._trail: list[tuple[int, int]] = []
        self._label = ""
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_scene(self, bgr, box=None, trail=None, label: str = "") -> None:
        self._bgr = bgr
        self._box = box
        self._trail = [(t["box"][0] + t["box"][2] // 2, t["box"][1] + t["box"][3] // 2)
                       for t in (trail or [])]
        self._label = label
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 8 or h < 8:
            return

        for x in range(0, w, 34):
            self.create_line(x, 0, x, h, fill=C["hairline"])
        for y in range(0, h, 34):
            self.create_line(0, y, w, y, fill=C["hairline"])

        if self._bgr is None:
            self.create_text(w // 2, h // 2, text="NO SCENE LOADED",
                             fill=C["inkFaint"], font=(FONT_MONO, 10))
            return

        ih, iw = self._bgr.shape[:2]
        pad = 10
        k = min((w - pad * 2) / iw, (h - pad * 2) / ih)
        dw, dh = max(1, int(iw * k)), max(1, int(ih * k))
        ox, oy = (w - dw) // 2, (h - dh) // 2

        rgb = self._bgr[:, :, ::-1]
        img = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)          # keep a reference alive
        self.create_image(ox, oy, image=self._photo, anchor="nw")
        self.create_rectangle(ox, oy, ox + dw, oy + dh, outline=C["hairlineHot"])

        for i in range(1, len(self._trail)):
            a, b = self._trail[i - 1], self._trail[i]
            self.create_line(ox + a[0] * k, oy + a[1] * k, ox + b[0] * k, oy + b[1] * k,
                             fill=C["violet"], width=1)

        if self._box:
            x, y, bw, bh = self._box
            x0, y0 = ox + x * k, oy + y * k
            x1, y1 = x0 + bw * k, y0 + bh * k
            self.create_rectangle(x0, y0, x1, y1, outline=C["solarDeep"], dash=(3, 3))
            arm = min(16, bw * k * 0.3)
            for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                self.create_line(cx, cy, cx + sx * arm, cy, fill=C["solar"], width=2)
                self.create_line(cx, cy, cx, cy + sy * arm, fill=C["solar"], width=2)
            if self._label:
                self.create_rectangle(x0, y0 - 20, x0 + 9 * len(self._label) + 10, y0 - 3,
                                      fill=C["solar"], outline="")
                self.create_text(x0 + 6, y0 - 12, text=self._label, anchor="w",
                                 fill=C["void"], font=(FONT_MONO, 9, "bold"))


class Gauge(tk.Canvas):
    """One metric: label, mono value, hairline track."""

    def __init__(self, master: Any, label: str, tone: str = "shadow", invert: bool = False) -> None:
        super().__init__(master, height=32, bg=C["panel"], highlightthickness=0, bd=0)
        self.label, self.tone, self.invert = label, tone, invert
        self.value = 0.0
        self.bind("<Configure>", lambda _e: self.redraw())

    def set(self, value: float) -> None:
        self.value = max(0.0, min(1.0, float(value)))
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w = self.winfo_width()
        if w < 8:
            return
        tone = C["alert"] if (self.invert and self.value > 0.35) else C[self.tone]
        self.create_text(0, 8, text=self.label.upper(), anchor="w",
                         fill=C["inkMuted"], font=(FONT_MONO, 8))
        self.create_text(w, 8, text=f"{self.value:.3f}", anchor="e",
                         fill=tone, font=(FONT_MONO, 9, "bold"))
        self.create_rectangle(0, 20, w, 23, fill=C["hairline"], outline="")
        if self.value > 0:
            self.create_rectangle(0, 20, w * self.value, 23, fill=tone, outline="")


class Dial(tk.Canvas):
    """Compact solar compass - azimuth needle, dashed cast direction, elevation."""

    def __init__(self, master: Any) -> None:
        super().__init__(master, height=128, bg=C["panel"], highlightthickness=0, bd=0)
        self.az, self.el, self.q = 148.0, 34.0, 0.7
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_sun(self, az: float, el: float, q: float) -> None:
        self.az, self.el, self.q = az, el, q
        self.redraw()

    def redraw(self) -> None:
        import math

        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 20:
            return
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 10
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=C["hairlineHot"])

        for deg in range(0, 360, 15):
            major = deg % 90 == 0
            a = math.radians(deg - 90)
            r0 = r - (8 if major else 4)
            self.create_line(cx + math.cos(a) * r0, cy + math.sin(a) * r0,
                             cx + math.cos(a) * r, cy + math.sin(a) * r,
                             fill=C["inkFaint"] if major else C["hairlineHot"])
        for deg, letter in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            a = math.radians(deg - 90)
            self.create_text(cx + math.cos(a) * (r - 18), cy + math.sin(a) * (r - 18),
                             text=letter, fill=C["inkFaint"], font=(FONT_MONO, 7))

        self.create_arc(cx - r + 11, cy - r + 11, cx + r - 11, cy + r - 11,
                        start=0, extent=-self.el, style="arc", outline=C["shadowDeep"], width=3)

        a = math.radians(self.az - 90)
        self.create_line(cx, cy, cx + math.cos(a) * (r - 9), cy + math.sin(a) * (r - 9),
                         fill=C["solar"], width=2)
        a2 = math.radians(self.az + 90)
        self.create_line(cx, cy, cx + math.cos(a2) * (r - 16), cy + math.sin(a2) * (r - 16),
                         fill=C["shadow"], width=2, dash=(4, 3))

        self.create_oval(cx - 19, cy - 19, cx + 19, cy + 19, fill=C["void"], outline=C["hairlineHot"])
        tone = C["signal"] if self.q > 0.66 else C["solar"] if self.q > 0.33 else C["alert"]
        self.create_text(cx, cy - 4, text=f"{self.el:.0f}°", fill=tone, font=(FONT_MONO, 11, "bold"))
        self.create_text(cx, cy + 9, text=f"Q {self.q:.2f}", fill=C["inkFaint"], font=(FONT_MONO, 7))


# --------------------------------------------------------------------------- #
# Console
# --------------------------------------------------------------------------- #
class FieldConsole(ctk.CTk):
    def __init__(self, client: DeckClient) -> None:
        theme.write_ctk_theme(THEME_FILE)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme(str(THEME_FILE))
        super().__init__()

        self.client = client
        self.busy = False
        self.active_job: str | None = None
        self.inbox: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        self.title("Shadow Hunter — Field Console")
        self.geometry("1280x820")
        self.minsize(1060, 700)
        self.configure(fg_color=C["void"])
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._header()
        self._sidebar()
        self._stage()
        self._readouts()

        self.after(60, self._drain)
        self.after(120, lambda: self._call(self.client.health, "health"))
        self.after(300, self.run_hunt)
        self.after(1200, self._poll)

    # ------------------------------------------------------------------ #
    def _header(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=58,
                           border_width=0)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        bar.grid_propagate(False)

        wrap = ctk.CTkFrame(bar, fg_color="transparent")
        wrap.pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(wrap, text="SHADOW", font=(FONT_DISPLAY, 19, "bold"),
                     text_color=C["ink"]).pack(side="left")
        ctk.CTkLabel(wrap, text="HUNTER", font=(FONT_DISPLAY, 19, "bold"),
                     text_color=C["solar"]).pack(side="left")
        ctk.CTkLabel(bar, text="field console  ·  customtkinter",
                     font=(FONT_MONO, 10), text_color=C["inkFaint"]).pack(side="left", padx=16)

        self.health = ctk.CTkLabel(bar, text="connecting…", font=(FONT_MONO, 10),
                                   text_color=C["inkMuted"])
        self.health.pack(side="right", padx=20)

    def _sidebar(self) -> None:
        side = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, width=286,
                            border_width=0)
        side.grid(row=1, column=0, sticky="nsw")
        side.grid_propagate(False)

        def label(text: str) -> None:
            ctk.CTkLabel(side, text=text.upper(), font=(FONT_MONO, 9, "bold"),
                         text_color=C["inkFaint"], anchor="w").pack(fill="x", padx=18, pady=(14, 2))

        def slider(lo: int, hi: int, value: int, suffix: str = "",
                   on_change: Callable | None = None):
            row = ctk.CTkFrame(side, fg_color="transparent")
            row.pack(fill="x", padx=18)
            readout = ctk.CTkLabel(row, text=f"{value}{suffix}", font=(FONT_MONO, 11),
                                   text_color=C["shadow"], anchor="e")
            readout.pack(side="right")
            s = ctk.CTkSlider(side, from_=lo, to=hi, number_of_steps=hi - lo)
            s.set(value)
            s.pack(fill="x", padx=18, pady=(2, 0))

            def handler(v: float) -> None:
                readout.configure(text=f"{int(v)}{suffix}")
                if on_change:
                    on_change(v)

            s.configure(command=handler)
            return s

        label("tile size")
        self.in_size = slider(256, 1024, 512, " px")
        label("buildings")
        self.in_buildings = slider(3, 40, 16)
        label("sun elevation")
        self.in_elev = slider(8, 80, 34, "°", lambda _v: self._sun_note())
        label("sun azimuth")
        self.in_azim = slider(0, 359, 148, "°", lambda _v: self._sun_note())
        label("step budget")
        self.in_steps = slider(8, 160, 48)

        label("policy")
        self.in_policy = ctk.CTkSegmentedButton(side, values=["auto", "learned", "greedy"])
        self.in_policy.set("auto")
        self.in_policy.pack(fill="x", padx=18, pady=(2, 12))

        ctk.CTkButton(side, text="RUN HUNT", command=self.run_hunt, height=38,
                      font=(FONT_MONO, 12, "bold"), fg_color=C["solar"],
                      hover_color=C["solarDeep"], text_color=C["void"],
                      border_color=C["solar"]).pack(fill="x", padx=18, pady=(4, 6))
        ctk.CTkButton(side, text="SWEEP SCENE", command=self.run_sweep, height=34,
                      font=(FONT_MONO, 11), fg_color="transparent",
                      hover_color=C["elevated"], text_color=C["inkMuted"]).pack(fill="x", padx=18)

        ctk.CTkFrame(side, height=1, fg_color=C["hairline"]).pack(fill="x", padx=18, pady=14)
        label("training")
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(2, 6))
        ctk.CTkButton(row, text="POLICY", width=118, height=30, font=(FONT_MONO, 10),
                      command=self.train_rl).pack(side="left")
        ctk.CTkButton(row, text="REGRESSOR", width=118, height=30, font=(FONT_MONO, 10),
                      command=self.train_cnn).pack(side="right")
        self.job_bar = ctk.CTkProgressBar(side, height=4)
        self.job_bar.set(0)
        self.job_bar.pack(fill="x", padx=18, pady=(4, 4))
        self.job_line = ctk.CTkLabel(side, text="no run active", font=(FONT_MONO, 9),
                                     text_color=C["inkFaint"], anchor="w", justify="left")
        self.job_line.pack(fill="x", padx=18)
        ctk.CTkButton(side, text="ABORT", height=28, font=(FONT_MONO, 10),
                      fg_color="transparent", text_color=C["alert"],
                      hover_color=C["alert"], border_color=C["alert"], border_width=1,
                      command=self.abort).pack(fill="x", padx=18, pady=10)

    def _stage(self) -> None:
        stage = ctk.CTkFrame(self, fg_color=C["void"], corner_radius=0)
        stage.grid(row=1, column=1, sticky="nsew", padx=12, pady=12)
        stage.grid_rowconfigure(0, weight=1)
        stage.grid_columnconfigure(0, weight=1)

        shell = ctk.CTkFrame(stage, fg_color=C["panel"], border_width=1, border_color=C["hairline"])
        shell.grid(row=0, column=0, sticky="nsew")
        shell.grid_rowconfigure(0, weight=1)
        shell.grid_columnconfigure(0, weight=1)
        self.view = SceneView(shell)
        self.view.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)

        self.note = ctk.CTkLabel(stage, text="", font=(FONT_MONO, 10),
                                 text_color=C["inkMuted"], anchor="w", justify="left")
        self.note.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        self.log = ctk.CTkTextbox(stage, height=132, font=(FONT_MONO, 10),
                                  fg_color=C["void"], text_color=C["inkMuted"],
                                  border_width=1, border_color=C["hairline"])
        self.log.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self._log("console ready")

    def _readouts(self) -> None:
        col = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, width=306, border_width=0)
        col.grid(row=1, column=2, sticky="nse")
        col.grid_propagate(False)

        def readout(caption: str, unit: str, tone: str):
            box = ctk.CTkFrame(col, fg_color=C["elevated"], border_width=1,
                               border_color=C["hairline"], corner_radius=6)
            box.pack(fill="x", padx=16, pady=(12, 0))
            ctk.CTkLabel(box, text=caption.upper(), font=(FONT_MONO, 9, "bold"),
                         text_color=C["inkFaint"], anchor="w").pack(fill="x", padx=14, pady=(10, 0))
            value = ctk.CTkLabel(box, text="--", font=(FONT_MONO, 26, "bold"),
                                 text_color=C[tone], anchor="w")
            value.pack(fill="x", padx=14)
            unit_lbl = ctk.CTkLabel(box, text=unit, font=(FONT_MONO, 9),
                                    text_color=C["inkFaint"], anchor="w")
            unit_lbl.pack(fill="x", padx=14, pady=(0, 10))
            return value, unit_lbl

        self.r_height = readout("estimated height", "metres above ground", "solar")
        self.r_pair = ctk.CTkFrame(col, fg_color="transparent")
        self.r_pair.pack(fill="x")
        self.r_sigma = readout("uncertainty", "1σ metres", "shadow")
        self.r_conf = readout("zone confidence", "occlusion · truncation · sun", "signal")

        ctk.CTkLabel(col, text="SOLAR GEOMETRY", font=(FONT_MONO, 9, "bold"),
                     text_color=C["inkFaint"], anchor="w").pack(fill="x", padx=16, pady=(16, 2))
        self.dial = Dial(col)
        self.dial.pack(fill="x", padx=16)
        self.sun_note = ctk.CTkLabel(col, text="", font=(FONT_BODY, 10), wraplength=262,
                                     text_color=C["inkMuted"], anchor="w", justify="left")
        self.sun_note.pack(fill="x", padx=16, pady=(4, 0))

        ctk.CTkLabel(col, text="ZONE METRICS", font=(FONT_MONO, 9, "bold"),
                     text_color=C["inkFaint"], anchor="w").pack(fill="x", padx=16, pady=(14, 4))
        self.gauges: dict[str, Gauge] = {}
        for key, label, tone, inv in (
            ("contrast", "shadow contrast", "shadow", False),
            ("isolation", "blob isolation", "shadow", False),
            ("edge_coherence", "edge coherence", "solar", False),
            ("axis_alignment", "azimuth alignment", "solar", False),
            ("occlusion", "occlusion", "signal", True),
            ("truncation", "truncation", "signal", True),
        ):
            g = Gauge(col, label, tone, inv)
            g.pack(fill="x", padx=16)
            self.gauges[key] = g

        ctk.CTkLabel(col, text="PROXY REWARD", font=(FONT_MONO, 9, "bold"),
                     text_color=C["inkFaint"], anchor="w").pack(fill="x", padx=16, pady=(12, 4))
        self.reward_gauges: dict[str, Gauge] = {}
        for key, label in (("r1_contrast", "R1 contrast·isolation"),
                           ("r2_structure", "R2 edge·entropy"),
                           ("r3_azimuth", "R3 azimuth coherence")):
            g = Gauge(col, label, "violet")
            g.pack(fill="x", padx=16)
            self.reward_gauges[key] = g

        self._sun_note()

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    def _call(self, fn: Callable, tag: str, *args: Any, **kwargs: Any) -> None:
        def worker() -> None:
            try:
                self.inbox.put((tag, fn(*args, **kwargs)))
            except DeckError as exc:
                self.inbox.put(("error", str(exc)))
            except Exception as exc:
                self.inbox.put(("error", f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=worker, daemon=True).start()

    def _drain(self) -> None:
        while True:
            try:
                tag, payload = self.inbox.get_nowait()
            except queue.Empty:
                break
            handler = getattr(self, f"_on_{tag}", None)
            if handler:
                handler(payload)
        self.after(60, self._drain)

    def _log(self, message: str) -> None:
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")

    def _sun(self) -> dict[str, float]:
        return {"azimuth_deg": float(self.in_azim.get()),
                "elevation_deg": float(self.in_elev.get()), "gsd_m": 0.5}

    def _sun_note(self) -> None:
        sun = SunGeometry(float(self.in_azim.get()), float(self.in_elev.get()), 0.5)
        q = quality_of_geometry(sun)
        self.dial.set_sun(sun.azimuth_deg, sun.elevation_deg, q)
        verdict = ("long, clean shadows — ideal metrology" if q > 0.66 else
                   "usable geometry" if q > 0.33 else
                   "poor geometry — shadows too short or merging")
        self.sun_note.configure(text=f"Shadow cast toward {sun.shadow_bearing_deg:.0f}°. {verdict}.")

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
    def run_hunt(self) -> None:
        if self.busy:
            return
        self.busy = True
        self._log("hunting…")
        self._call(self.client.analyze, "hunt",
                   size=int(self.in_size.get()), buildings=int(self.in_buildings.get()),
                   sun=self._sun(), policy=self.in_policy.get(),
                   max_steps=int(self.in_steps.get()))

    def run_sweep(self) -> None:
        if self.busy:
            return
        self.busy = True
        self._log("sweeping the tile…")
        self._call(self.client.sweep, "sweep",
                   size=int(self.in_size.get()), buildings=int(self.in_buildings.get()),
                   sun=self._sun(), policy=self.in_policy.get(),
                   limit=int(self.in_buildings.get()))

    def train_rl(self) -> None:
        self._call(self.client.train_rl, "job", algo="PPO", total_timesteps=20_000,
                   tag="shadow_hunter")

    def train_cnn(self) -> None:
        self._call(self.client.train_cnn, "job", scenes=24, epochs=20, tag="height_cnn")

    def abort(self) -> None:
        if self.active_job:
            self._call(self.client.abort, "noop", self.active_job)

    def _poll(self) -> None:
        if self.active_job:
            self._call(self.client.job, "jobstatus", self.active_job)
        self.after(1100, self._poll)

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    def _on_hunt(self, p: dict[str, Any]) -> None:
        self.busy = False
        h = p["height"]
        self.view.set_scene(decode_png(p.get("overlay_png")), p["box"],
                            p.get("trajectory"), f"{h['fused_m']:.1f} m")
        self.r_height[0].configure(text=f"{h['fused_m']:.1f}")
        self.r_height[1].configure(text=f"{h['floors']} storeys at 3.2 m")
        self.r_sigma[0].configure(text=f"{h['sigma_m']:.2f}")
        self.r_conf[0].configure(text=f"{h['confidence'] * 100:.0f}%")

        for key, gauge in self.gauges.items():
            gauge.set(p["metrics"].get(key, 0.0))
        for key, gauge in self.reward_gauges.items():
            gauge.set(p["breakdown"].get(key, 0.0))

        sun = p["scene"]["sun"]
        geom = SunGeometry(sun["azimuth_deg"], sun["elevation_deg"], sun["gsd_m"])
        self.dial.set_sun(geom.azimuth_deg, geom.elevation_deg, quality_of_geometry(geom))

        cnn = h["cnn_m"]
        self.note.configure(
            text=f"policy {p['policy']} · {p['steps']} steps · score {p['score']:+.3f} · "
                 f"zone {p['box']}\ngeometric {h['geometric_m']:.1f} m · "
                 f"cnn {f'{cnn:.1f} m' if cnn else 'not loaded'} · "
                 f"shadow {p['metrics']['shadow_len_px']:.0f} px · {p['elapsed_ms']:.0f} ms")
        self._log(f"zone {p['box']}  h={h['fused_m']:.1f}m  score={p['score']:+.3f}")

    def _on_sweep(self, p: dict[str, Any]) -> None:
        self.busy = False
        self.view.set_scene(decode_png(p.get("overlay_png")))
        self.r_height[0].configure(text=f"{p['mae_m']:.2f}" if p["mae_m"] else "--")
        self.r_height[1].configure(text="MAE across the tile, metres")
        self.r_sigma[0].configure(text=f"{p['rmse_m']:.2f}" if p["rmse_m"] else "--")
        self.r_conf[0].configure(text=f"{p['mean_score'] * 100:.0f}%")
        self.note.configure(text=f"swept {len(p['items'])} buildings in {p['elapsed_ms']:.0f} ms")
        self._log(f"sweep: MAE {p['mae_m']} m  RMSE {p['rmse_m']} m")

    def _on_job(self, job: dict[str, Any]) -> None:
        self.active_job = job["id"]
        self._log(f"job {job['id']} ({job['kind']}) queued")

    def _on_jobstatus(self, job: dict[str, Any]) -> None:
        self.job_bar.set(job["progress"])
        self.job_line.configure(text=f"{job['kind']} · {job['state']}\n{job['message'][:44]}")
        if job["state"] in {"done", "failed", "aborted"}:
            self._log(f"job {job['id']} {job['state']}: {job['message']}")
            self.active_job = None
            self._call(self.client.health, "health")

    def _on_health(self, h: dict[str, Any]) -> None:
        policy = "policy ✓" if h["policy_loaded"] else "policy · greedy"
        cnn = "cnn ✓" if h["cnn_loaded"] else "cnn · geometric"
        self.health.configure(text=f"{'CUDA' if h['cuda'] else 'CPU'} · torch {h['torch']} · "
                                   f"{policy} · {cnn}")

    def _on_error(self, message: str) -> None:
        self.busy = False
        self._log(f"! {message}")

    def _on_noop(self, _payload: Any) -> None:
        pass


def main(base_url: str | None = None) -> int:
    if base_url:
        client = DeckClient(base_url=base_url)
    else:
        from ....services.supervisor import serve_in_thread

        client = serve_in_thread()
    FieldConsole(client).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
