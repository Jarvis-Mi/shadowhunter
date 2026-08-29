"""NiceGUI browser observatory.

Role in the suite: the *shared* surface. The Qt deck is for the operator at
the machine; this one is for the supervisor, the reviewer and the demo - it
runs in any browser on the network, including a phone at a conference stand.
"""
from __future__ import annotations

import asyncio
from typing import Any

from nicegui import app as nicegui_app
from nicegui import ui

from ...core.geo import SunGeometry, quality_of_geometry
from ...core.logging import get_logger
from ...services.client import DeckClient, DeckError
from ...templates import theme
from ...templates.components import svg

log = get_logger(__name__)

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2"
    "?family=Chakra+Petch:wght@400;600;700"
    "&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@400;600"
    "&display=swap"
)


class Observatory:
    """One page, three tabs, one client. State lives here, not in globals."""

    def __init__(self, client: DeckClient) -> None:
        self.client = client
        self.busy = False
        self.result: dict[str, Any] | None = None
        self.rl_series: list[float] = []
        self.cnn_series: list[float] = []
        self.active_job: str | None = None
        self.log_lines: list[str] = []

    # ------------------------------------------------------------------ #
    # Page
    # ------------------------------------------------------------------ #
    def build(self) -> None:
        ui.add_head_html(f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                         f'<link rel="stylesheet" href="{GOOGLE_FONTS}">')
        ui.add_head_html(f"<style>{theme.web_stylesheet()}</style>")

        with ui.element("div").classes("sh-root w-full"):
            self._header()
            with ui.tabs().classes("w-full px-6") as tabs:
                t_hunt = ui.tab("HUNT")
                t_train = ui.tab("TRAIN")
                t_archive = ui.tab("ARCHIVE")
            with ui.tab_panels(tabs, value=t_hunt).classes("w-full bg-transparent"):
                with ui.tab_panel(t_hunt).classes("bg-transparent"):
                    self._panel_hunt()
                with ui.tab_panel(t_train).classes("bg-transparent"):
                    self._panel_train()
                with ui.tab_panel(t_archive).classes("bg-transparent"):
                    self._panel_archive()

        ui.timer(1.0, self._tick)
        ui.timer(0.4, self._first_run, once=True)

    def _header(self) -> None:
        with ui.element("div").classes("sh-header"):
            ui.html('<h1 class="sh-wordmark">SHADOW<em>HUNTER</em></h1>')
            ui.element("div").classes("sh-divider")
            ui.html('<p class="sh-caption">Deep reinforcement learning selects the zone.<br>'
                    'A physics-anchored CNN turns its shadow into metres.</p>')
            ui.element("div").classes("grow")
            self.pill_state = ui.html(self._pill("connecting", "warn"))
            self.pill_device = ui.html(self._pill("device ?", ""))

    @staticmethod
    def _pill(text: str, kind: str = "") -> str:
        cls = f"sh-pill sh-pill--{kind}" if kind else "sh-pill"
        pulse = " sh-dot--pulse" if kind == "warn" else ""
        return f'<span class="{cls}"><i class="sh-dot{pulse}"></i>{text}</span>'

    # ------------------------------------------------------------------ #
    # HUNT
    # ------------------------------------------------------------------ #
    def _panel_hunt(self) -> None:
        with ui.row().classes("w-full no-wrap gap-4 p-6 items-stretch"):
            # --- left: viewport + controls ---------------------------------
            with ui.column().classes("grow gap-4"):
                with ui.element("div").classes("sh-card sh-rise grow"):
                    ui.html('<p class="sh-section">scene viewport</p>')
                    self.viewport = ui.element("div").classes("sh-viewport")
                    with self.viewport:
                        self.scene_img = ui.html('<div class="sh-empty">no scene loaded</div>')
                    self.scene_note = ui.html('<p class="sh-mono" style="margin-top:12px"></p>')

                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">acquisition parameters</p>')
                    with ui.row().classes("w-full no-wrap gap-6"):
                        with ui.column().classes("grow gap-1"):
                            ui.label("TILE SIZE").classes("sh-section").style("margin:0")
                            self.in_size = ui.slider(min=256, max=1024, step=64, value=512) \
                                .props("label-always markers")
                            ui.label("BUILDINGS").classes("sh-section").style("margin:0")
                            self.in_buildings = ui.slider(min=3, max=40, value=16).props("label-always")
                        with ui.column().classes("grow gap-1"):
                            ui.label("SUN ELEVATION °").classes("sh-section").style("margin:0")
                            self.in_elev = ui.slider(min=8, max=80, value=34).props("label-always")
                            ui.label("SUN AZIMUTH °").classes("sh-section").style("margin:0")
                            self.in_azim = ui.slider(min=0, max=359, value=148).props("label-always")
                        with ui.column().classes("gap-2").style("min-width:190px"):
                            self.in_policy = ui.select(["auto", "learned", "greedy"], value="auto",
                                                       label="POLICY").classes("w-full")
                            self.in_steps = ui.number(label="STEP BUDGET", value=48, min=8, max=200,
                                                      step=8).classes("w-full")
                            ui.button("RUN HUNT", on_click=self.run_hunt) \
                                .classes("sh-btn sh-btn--primary w-full").props("flat no-caps")
                            ui.button("SWEEP SCENE", on_click=self.run_sweep) \
                                .classes("sh-btn sh-btn--ghost w-full").props("flat no-caps")

                    for s in (self.in_elev, self.in_azim):
                        s.on_value_change(lambda _: self._refresh_dial())

            # --- right: readouts -------------------------------------------
            with ui.column().classes("gap-4").style("width:344px;min-width:344px"):
                with ui.element("div").classes("sh-card sh-card--accent sh-rise"):
                    ui.html('<p class="sh-section">estimated height</p>')
                    self.out_height = ui.html(self._readout("--", "metres above ground", "solar"))
                    ui.html('<hr class="sh-rule">')
                    with ui.row().classes("w-full justify-between"):
                        self.out_floors = ui.html(self._readout("--", "storeys", ""))
                        self.out_sigma = ui.html(self._readout("--", "1σ metres", "shadow"))
                        self.out_conf = ui.html(self._readout("--", "confidence", "signal"))

                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">solar geometry</p>')
                    self.dial = ui.html(svg.sun_dial(148, 34, 0.7))
                    self.dial_note = ui.html('<p class="sh-caption" style="margin-top:10px"></p>')

                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">zone metrics</p>')
                    self.out_metrics = ui.html("")

                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">proxy reward breakdown</p>')
                    self.out_reward = ui.html("")
                    ui.html('<p class="sh-caption" style="margin-top:10px">'
                            'R1 contrast·isolation &nbsp;·&nbsp; R2 edge·entropy &nbsp;·&nbsp; '
                            'R3 azimuth coherence. None of the three reads a ground-truth height.</p>')

    @staticmethod
    def _readout(value: str, unit: str, tone: str) -> str:
        return (f'<div class="sh-readout"><span class="v {tone}">{value}</span>'
                f'<span class="u">{unit}</span></div>')

    # ------------------------------------------------------------------ #
    # TRAIN
    # ------------------------------------------------------------------ #
    def _panel_train(self) -> None:
        with ui.row().classes("w-full no-wrap gap-4 p-6 items-start"):
            with ui.column().classes("gap-4").style("width:340px;min-width:340px"):
                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">stage 1 · zone policy</p>')
                    ui.html('<p class="sh-caption">PPO or DQN over the proxy reward, on randomised '
                            'synthetic cities. No labels are consumed.</p><hr class="sh-rule">')
                    self.rl_algo = ui.select(["PPO", "DQN"], value="PPO", label="ALGORITHM").classes("w-full")
                    self.rl_steps = ui.number(label="TIMESTEPS", value=20000, min=512, step=1000).classes("w-full")
                    ui.button("TRAIN POLICY", on_click=self.train_rl) \
                        .classes("sh-btn sh-btn--primary w-full mt-3").props("flat no-caps")

                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">stage 2 · height regressor</p>')
                    ui.html('<p class="sh-caption">CNN over RGB + shadow mask. Solar elevation and GSD '
                            'enter as physics; the head predicts a residual around h = L·tan θ.</p>'
                            '<hr class="sh-rule">')
                    self.cnn_scenes = ui.number(label="SCENES", value=24, min=1, max=400).classes("w-full")
                    self.cnn_epochs = ui.number(label="EPOCHS", value=20, min=1, max=500).classes("w-full")
                    ui.button("TRAIN REGRESSOR", on_click=self.train_cnn) \
                        .classes("sh-btn sh-btn--ghost w-full mt-3").props("flat no-caps")

                with ui.element("div").classes("sh-card sh-rise"):
                    ui.button("ABORT RUN", on_click=self.abort) \
                        .classes("sh-btn sh-btn--danger w-full").props("flat no-caps")
                    self.job_line = ui.html('<p class="sh-mono" style="margin-top:10px">idle</p>')
                    self.job_progress = ui.linear_progress(value=0, show_value=False).classes("mt-2")

            with ui.column().classes("grow gap-4"):
                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">episode return · policy</p>')
                    self.spark_rl = ui.html(svg.sparkline([], tone="solar"))
                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">validation MAE (metres) · regressor</p>')
                    self.spark_cnn = ui.html(svg.sparkline([], tone="shadow"))
                with ui.element("div").classes("sh-card sh-rise"):
                    ui.html('<p class="sh-section">telemetry</p>')
                    self.log_box = ui.html('<div class="sh-log"></div>')

    # ------------------------------------------------------------------ #
    # ARCHIVE
    # ------------------------------------------------------------------ #
    def _panel_archive(self) -> None:
        with ui.column().classes("w-full gap-4 p-6"):
            with ui.element("div").classes("sh-card sh-rise w-full"):
                with ui.row().classes("w-full items-center"):
                    ui.html('<p class="sh-section">checkpoints</p>')
                    ui.element("div").classes("grow")
                    ui.button("REFRESH", on_click=self.refresh_archive) \
                        .classes("sh-btn sh-btn--ghost").props("flat no-caps")
                self.art_table = ui.table(
                    columns=[{"name": c, "label": c.upper(), "field": c, "align": "left"}
                             for c in ("name", "kind", "size", "modified")],
                    rows=[], row_key="name").classes("w-full")
                self.art_table.on("rowClick", self._activate_artifact)
                ui.html('<p class="sh-caption" style="margin-top:8px">'
                        'Click a row to activate that checkpoint for inference.</p>')

            with ui.element("div").classes("sh-card sh-rise w-full"):
                ui.html('<p class="sh-section">recent estimates</p>')
                self.hist_table = ui.table(
                    columns=[{"name": c, "label": c.upper(), "field": c, "align": "left"}
                             for c in ("time", "scene", "policy", "height", "sigma", "score")],
                    rows=[], row_key="time").classes("w-full")

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _sun(self) -> dict[str, float]:
        return {"azimuth_deg": float(self.in_azim.value),
                "elevation_deg": float(self.in_elev.value), "gsd_m": 0.5}

    async def _first_run(self) -> None:
        await self._sync_health()
        self._refresh_dial()
        await self.run_hunt()
        await self.refresh_archive()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.viewport.classes(add="busy") if busy else self.viewport.classes(remove="busy")
        self.pill_state.set_content(self._pill("hunting" if busy else "nominal",
                                               "warn" if busy else "live"))

    async def run_hunt(self) -> None:
        if self.busy:
            return
        self._set_busy(True)
        try:
            payload = await asyncio.to_thread(
                self.client.analyze,
                size=int(self.in_size.value), buildings=int(self.in_buildings.value),
                sun=self._sun(), policy=self.in_policy.value,
                max_steps=int(self.in_steps.value or 48),
            )
        except DeckError as exc:
            ui.notify(str(exc), type="negative")
            self._set_busy(False)
            return
        self._set_busy(False)
        self.result = payload
        self._render_hunt(payload)

    def _render_hunt(self, p: dict[str, Any]) -> None:
        self.scene_img.set_content(
            f'<img src="{svg.data_uri(p.get("overlay_png"))}" alt="annotated satellite tile">')
        h = p["height"]
        self.out_height.set_content(self._readout(f"{h['fused_m']:.1f}", "metres above ground", "solar"))
        self.out_floors.set_content(self._readout(str(h["floors"]), "storeys", ""))
        self.out_sigma.set_content(self._readout(f"{h['sigma_m']:.2f}", "1σ metres", "shadow"))
        self.out_conf.set_content(self._readout(f"{h['confidence'] * 100:.0f}%", "confidence", "signal"))

        m = p["metrics"]
        self.out_metrics.set_content("".join([
            svg.metric_row("shadow contrast", m["contrast"]),
            svg.metric_row("blob isolation", m["isolation"]),
            svg.metric_row("edge coherence", m["edge_coherence"], "solar"),
            svg.metric_row("azimuth alignment", m["axis_alignment"], "solar"),
            svg.metric_row("occlusion", m["occlusion"], "shadow", invert=True),
            svg.metric_row("truncation", m["truncation"], "shadow", invert=True),
        ]))
        b = p["breakdown"]
        self.out_reward.set_content("".join([
            svg.metric_row("R1 contrast · isolation", b["r1_contrast"], "violet"),
            svg.metric_row("R2 edge · entropy", b["r2_structure"], "violet"),
            svg.metric_row("R3 azimuth coherence", b["r3_azimuth"], "violet"),
        ]))

        cnn = h["cnn_m"]
        self.scene_note.set_content(
            f'<p class="sh-mono">policy <b style="color:{theme.c("ink")}">{p["policy"]}</b> · '
            f'{p["steps"]} steps · score {p["score"]:+.3f} · zone {p["box"]} · '
            f'geometric {h["geometric_m"]:.1f} m · '
            f'cnn {f"{cnn:.1f} m" if cnn else "not loaded"} · {p["elapsed_ms"]:.0f} ms</p>')

        sun = p["scene"]["sun"]
        self._refresh_dial(sun["azimuth_deg"], sun["elevation_deg"])

    async def run_sweep(self) -> None:
        if self.busy:
            return
        self._set_busy(True)
        try:
            p = await asyncio.to_thread(
                self.client.sweep, size=int(self.in_size.value),
                buildings=int(self.in_buildings.value), sun=self._sun(),
                policy=self.in_policy.value, limit=int(self.in_buildings.value))
        except DeckError as exc:
            ui.notify(str(exc), type="negative")
            self._set_busy(False)
            return
        self._set_busy(False)
        self.scene_img.set_content(
            f'<img src="{svg.data_uri(p.get("overlay_png"))}" alt="swept satellite tile">')
        self.out_height.set_content(self._readout(f"{p['mae_m']:.2f}", "MAE across the tile, metres", "solar"))
        self.out_sigma.set_content(self._readout(f"{p['rmse_m']:.2f}", "RMSE metres", "shadow"))
        self.out_floors.set_content(self._readout(str(len(p["items"])), "buildings", ""))
        self.out_conf.set_content(self._readout(f"{p['mean_score'] * 100:.0f}%", "mean score", "signal"))
        self.scene_note.set_content(
            f'<p class="sh-mono">swept {len(p["items"])} buildings in {p["elapsed_ms"]:.0f} ms</p>')

    def _refresh_dial(self, azimuth: float | None = None, elevation: float | None = None) -> None:
        az = float(self.in_azim.value if azimuth is None else azimuth)
        el = float(self.in_elev.value if elevation is None else elevation)
        sun = SunGeometry(az, el, 0.5)
        q = quality_of_geometry(sun)
        self.dial.set_content(svg.sun_dial(az, el, q))
        verdict = ("Long, clean shadows — ideal metrology." if q > 0.66 else
                   "Usable geometry." if q > 0.33 else
                   "Poor geometry — shadows are too short or merging.")
        self.dial_note.set_content(
            f'<p class="sh-caption">Shadow cast toward {sun.shadow_bearing_deg:.0f}°. {verdict}</p>')

    async def train_rl(self) -> None:
        self.rl_series.clear()
        await self._start(self.client.train_rl, algo=self.rl_algo.value,
                          total_timesteps=int(self.rl_steps.value), tag="shadow_hunter")

    async def train_cnn(self) -> None:
        self.cnn_series.clear()
        await self._start(self.client.train_cnn, scenes=int(self.cnn_scenes.value),
                          epochs=int(self.cnn_epochs.value), tag="height_cnn")

    async def _start(self, fn, **kwargs) -> None:
        try:
            job = await asyncio.to_thread(lambda: fn(**kwargs))
        except DeckError as exc:
            ui.notify(str(exc), type="negative")
            return
        self.active_job = job["id"]
        ui.notify(f"job {job['id']} queued", type="positive")

    async def abort(self) -> None:
        if not self.active_job:
            return
        await asyncio.to_thread(self.client.abort, self.active_job)
        ui.notify("abort requested")

    async def refresh_archive(self) -> None:
        try:
            arts = await asyncio.to_thread(self.client.artifacts)
            hist = await asyncio.to_thread(self.client.history, 25)
        except DeckError:
            return
        self.art_table.rows = [{"name": a["name"], "kind": a["kind"],
                                "size": f"{a['size_bytes'] / 1e6:.2f} MB",
                                "modified": a["modified"][:19].replace("T", " ")} for a in arts]
        self.hist_table.rows = [{"time": h["created_at"][11:19], "scene": h["scene"],
                                 "policy": h["policy"], "height": f"{h['height_m']:.1f}",
                                 "sigma": f"{h['sigma_m']:.2f}", "score": f"{h['score']:+.3f}"}
                                for h in hist]

    async def _activate_artifact(self, event) -> None:
        try:
            row = event.args[1]
            result = await asyncio.to_thread(self.client.load_artifact, row["name"])
            ui.notify(f"activated {result['loaded']}", type="positive")
            await self._sync_health()
        except Exception as exc:
            ui.notify(str(exc), type="negative")

    # ------------------------------------------------------------------ #
    # Polling
    # ------------------------------------------------------------------ #
    async def _tick(self) -> None:
        try:
            events = await asyncio.to_thread(self.client.telemetry, None, 40)
        except DeckError:
            self.pill_state.set_content(self._pill("backend down", "warn"))
            return

        touched_rl = touched_cnn = False
        lines: list[str] = []
        for e in events[-24:]:
            topic = e.get("topic", "")
            if topic == "ping":
                continue
            if topic == "rl.progress" and e.get("ep_reward_mean") is not None:
                self.rl_series.append(float(e["ep_reward_mean"]))
                touched_rl = True
            if topic == "cnn.progress":
                self.cnn_series.append(float(e["val_mae"]))
                touched_cnn = True
            klass = "e" if "fail" in topic else "w" if topic.startswith("job") else "k"
            body = " ".join(f"{k}={v}" for k, v in e.items()
                            if k not in {"topic", "ts"} and not isinstance(v, (dict, list)))
            lines.append(f'<div><span class="t">{e.get("ts", "")[11:19]}</span> '
                         f'<span class="{klass}">{topic}</span> {body[:150]}</div>')

        if touched_rl:
            self.spark_rl.set_content(svg.sparkline(self.rl_series[-220:], tone="solar"))
        if touched_cnn:
            self.spark_cnn.set_content(svg.sparkline(self.cnn_series[-220:], tone="shadow"))
        if lines:
            self.log_box.set_content(f'<div class="sh-log">{"".join(reversed(lines))}</div>')

        if self.active_job:
            try:
                job = await asyncio.to_thread(self.client.job, self.active_job)
            except DeckError:
                return
            self.job_line.set_content(
                f'<p class="sh-mono">{job["id"]} · {job["kind"]} · '
                f'<b style="color:{theme.c("solar")}">{job["state"]}</b> · {job["message"]}</p>')
            self.job_progress.set_value(job["progress"])
            if job["state"] in {"done", "failed", "aborted"}:
                self.active_job = None
                await self._sync_health()
                await self.refresh_archive()

    async def _sync_health(self) -> None:
        try:
            h = await asyncio.to_thread(self.client.health)
        except DeckError:
            self.pill_state.set_content(self._pill("backend down", "warn"))
            return
        policy = "policy ✓" if h["policy_loaded"] else "policy · greedy"
        cnn = "cnn ✓" if h["cnn_loaded"] else "cnn · geometric"
        self.pill_state.set_content(self._pill("nominal", "live"))
        self.pill_device.set_content(self._pill(
            f'<b>{"CUDA" if h["cuda"] else "CPU"}</b> · torch {h["torch"]} · {policy} · {cnn}'))


# --------------------------------------------------------------------------- #
def main(base_url: str | None = None, port: int = 8080, show: bool = True) -> int:
    if base_url:
        client = DeckClient(base_url=base_url)
    else:
        from ...services.supervisor import serve_in_thread

        client = serve_in_thread()

    @ui.page("/")
    def index() -> None:
        ui.query("body").style(f"background:{theme.c('void')}")
        Observatory(client).build()

    nicegui_app.on_shutdown(lambda: log.info("observatory closed"))
    ui.run(title="Shadow Hunter · Observatory", port=port, show=show,
           dark=True, favicon="🛰", reload=False, uvicorn_logging_level="warning")
    return 0


if __name__ in {"__main__", "__mp_main__"}:
    main()
