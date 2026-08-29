"""The Shadow Hunter instrument deck - one page, everything on it.

Operating loop, left to right:

    find a place  ->  draw an AOI on the map  ->  SURVEY (imagery + sun + count)
                  ->  MEASURE (hunt each shadow, height per structure)

Layout rules this file is built to obey, because the previous version broke
all three: the side rails scroll instead of clipping, the centre is a splitter
the operator controls, and the window works down to 1180x720.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from PySide6.QtCore import (QDate, QDateTime, QObject, QRunnable, Qt, QThreadPool, QTime,
                            QTimer, Signal, Slot)
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QDateTimeEdit,
                               QDialog, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMainWindow, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
                               QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ....core.geo import SunGeometry, quality_of_geometry
from ....services.client import DeckClient, DeckError, TelemetryStream, decode_png
from .instruments import CopilotPane, HistogramStrip, PipelineStrip, SpectraStrip, TimelineView
from .mapview import AOIView, MapCanvas
from .scene3dview import Field3DDialog, Field3DView
from .widgets import MetricBar, Readout, Sparkline, StatusDot, SunCompass


# --------------------------------------------------------------------------- #
# Threading
# --------------------------------------------------------------------------- #
class WorkerSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    """Run any callable off the GUI thread and deliver the result as a signal."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.done.emit(self.fn(*self.args, **self.kwargs))
        except DeckError as exc:
            self.signals.failed.emit(str(exc))
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}\n"
                                     f"{traceback.format_exc(limit=2)}")


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def panel(name: str = "Panel") -> QFrame:
    frame = QFrame()
    frame.setObjectName(name)
    return frame


def section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("SectionLabel")
    return label


def caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Caption")
    label.setWordWrap(True)
    return label


def rule(vertical: bool = False) -> QFrame:
    frame = QFrame()
    frame.setObjectName("RuleV" if vertical else "Rule")
    frame.setFrameShape(QFrame.VLine if vertical else QFrame.HLine)
    return frame


class Rail(QScrollArea):
    """A fixed-width column that scrolls vertically and never clips sideways.

    Two failure modes are being defended against here, both of which the
    earlier deck suffered from:

    * Vertical: content taller than the window used to be squeezed until the
      readouts collapsed into empty grey bars and the lower controls fell off
      the bottom. Now it scrolls.
    * Horizontal: a fixed-width rail whose children ask for more width than it
      has does not shrink them - it lets them overflow, and with the
      horizontal scrollbar switched off the overflow is simply cut away. So
      the body is pinned to the viewport width on every resize, which forces
      the layout to honour the rail's width instead of exceeding it.
    """

    def __init__(self, width: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFixedWidth(width)

        self.body = QWidget()
        self.body.setObjectName("RailBody")
        self.column = QVBoxLayout(self.body)
        self.column.setContentsMargins(14, 14, 14, 16)
        self.column.setSpacing(10)
        self.setWidget(self.body)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.body.setMaximumWidth(self.viewport().width())


def scroll_rail(width: int) -> tuple[Rail, QVBoxLayout]:
    rail = Rail(width)
    return rail, rail.column


class LabeledSlider(QWidget):
    """Slider with a mono readout above it."""

    def __init__(self, label: str, lo: int, hi: int, value: int, suffix: str = "",
                 scale: float = 1.0) -> None:
        super().__init__()
        self.scale, self.suffix = scale, suffix
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        name = QLabel(label.upper())
        name.setObjectName("SectionLabel")
        self.value = QLabel("")
        self.value.setObjectName("MonoValue")
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        head.addWidget(name)
        head.addStretch(1)
        head.addWidget(self.value)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._sync)

        layout.addLayout(head)
        layout.addWidget(self.slider)
        self._sync(value)

    def _sync(self, raw: int) -> None:
        self.value.setText(f"{raw * self.scale:g}{self.suffix}")

    def get(self) -> float:
        return self.slider.value() * self.scale


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class DeckWindow(QMainWindow):
    def __init__(self, client: DeckClient) -> None:
        super().__init__()
        self.client = client
        self.pool = QThreadPool.globalInstance()
        self.busy = False
        self.active_job: str | None = None
        self.survey_data: dict[str, Any] | None = None
        self.measured: dict[int, dict[str, Any]] = {}
        self.selected: int | None = None
        self._best_when: str | None = None
        self._sun_autoshift = False
        self._aoi_bbox: tuple[float, float, float, float] | None = None
        self._inspect_payload: dict[str, Any] | None = None
        self._construct_payload: dict[str, Any] | None = None
        self._suggestions: list[dict[str, Any]] = []
        self._suggest_seq = 0
        self._suggest_timer = QTimer(self)
        self._suggest_timer.setSingleShot(True)
        self._suggest_timer.setInterval(900)
        self._suggest_timer.timeout.connect(self._request_suggestions)

        self.setWindowTitle("Shadow Hunter — Instrument Deck")
        self.resize(1600, 980)
        self.setMinimumSize(1180, 720)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())
        outer.addWidget(rule())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_left_rail())
        body.addWidget(rule(vertical=True))
        body.addWidget(self._build_centre(), 1)
        body.addWidget(rule(vertical=True))
        body.addWidget(self._build_right_rail())
        outer.addLayout(body, 1)

        self.setStatusBar(QStatusBar())
        self.status("deck idle — LOCATE a place for auto-suggest, or DRAW AOI")

        self.stream = TelemetryStream(self.client, self._on_telemetry)
        self.stream.start()
        self.job_timer = QTimer(self)
        self.job_timer.setInterval(900)
        self.job_timer.timeout.connect(self._poll_job)

        QTimer.singleShot(150, self._refresh_health)
        QTimer.singleShot(250, lambda: self._sync_sun(*self.map.aoi_centre_or_view()))
        QTimer.singleShot(800, lambda: self._schedule_suggestions(force=True))

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    def _build_header(self) -> QWidget:
        head = panel("PanelFlush")
        head.setFixedHeight(58)
        layout = QHBoxLayout(head)
        layout.setContentsMargins(18, 0, 16, 0)
        layout.setSpacing(12)

        mark = QLabel("SHADOW")
        mark.setObjectName("Wordmark")
        mark2 = QLabel("HUNTER")
        mark2.setObjectName("WordmarkAccent")
        layout.addWidget(mark)
        layout.addWidget(mark2)
        layout.addSpacing(6)
        layout.addWidget(rule(vertical=True))
        layout.addSpacing(6)

        self.search = QLineEdit()
        self.search.setPlaceholderText("place name, or  lat, lon   →  Enter")
        self.search.setMinimumWidth(260)
        self.search.setMaximumWidth(420)
        self.search.returnPressed.connect(self.go_to_place)
        layout.addWidget(self.search, 1)

        go = QPushButton("LOCATE")
        go.setObjectName("Ghost")
        go.clicked.connect(self.go_to_place)
        layout.addWidget(go)
        layout.addStretch(1)

        self.health_label = QLabel("connecting…")
        self.health_label.setObjectName("Mono")
        self.status_dot = StatusDot()
        layout.addWidget(self.health_label)
        layout.addWidget(self.status_dot)
        return head

    # ------------------------------------------------------------------ #
    # Left rail - the operating sequence, top to bottom
    # ------------------------------------------------------------------ #
    def _build_left_rail(self) -> QWidget:
        rail, layout = scroll_rail(298)

        layout.addWidget(section("1 · area of interest"))
        layout.addWidget(caption("LOCATE یا جابه‌جایی نقشه → پیشنهاد خودکار سازه. "
                                 "روی کادر خط‌چین کلیک کنید یا ACCEPT. "
                                 "DRAW AOI فقط اگر پیشنهاد کافی نبود."))

        self.btn_aoi_mode = QPushButton("DRAW AOI")
        self.btn_aoi_mode.setObjectName("Ghost")
        self.btn_aoi_mode.setCheckable(True)
        self.btn_aoi_mode.toggled.connect(self._toggle_aoi_mode)

        btn_clear = QPushButton("CLEAR")
        btn_clear.setObjectName("Ghost")
        btn_clear.clicked.connect(self._clear_aoi)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self.btn_aoi_mode, 2)
        row.addWidget(btn_clear, 1)
        layout.addLayout(row)

        self.btn_accept_suggest = QPushButton("ACCEPT SUGGESTION")
        self.btn_accept_suggest.setObjectName("Ghost")
        self.btn_accept_suggest.setEnabled(False)
        self.btn_accept_suggest.setToolTip("بهترین پیشنهاد را به‌عنوان AOI بپذیر")
        self.btn_accept_suggest.clicked.connect(self.accept_suggestion)
        layout.addWidget(self.btn_accept_suggest)

        self.aoi_label = QLabel("no area selected — locate a place to auto-suggest")
        self.aoi_label.setObjectName("Mono")
        self.aoi_label.setWordWrap(True)
        layout.addWidget(self.aoi_label)

        self.btn_field = QPushButton("RUN FIELD")
        self.btn_field.setObjectName("Primary")
        self.btn_field.setToolTip("Survey + seasonal sun + spectra + measure + 3D + copilot")
        self.btn_field.clicked.connect(self.run_field)
        layout.addWidget(self.btn_field)

        self.btn_save_place = QPushButton("SAVE PLACE")
        self.btn_save_place.setObjectName("Ghost")
        self.btn_save_place.clicked.connect(self.save_place)
        layout.addWidget(self.btn_save_place)

        layout.addWidget(caption("favorites"))
        self.fav_box = QComboBox()
        self.fav_box.setEditable(False)
        self.fav_box.currentIndexChanged.connect(self._on_favorite_picked)
        layout.addWidget(self.fav_box)

        layout.addWidget(caption("history"))
        self.hist_box = QComboBox()
        self.hist_box.currentIndexChanged.connect(self._on_history_picked)
        layout.addWidget(self.hist_box)

        self.count_label = QLabel("—")
        self.count_label.setObjectName("Mono")
        self.count_label.setWordWrap(True)
        layout.addWidget(self.count_label)

        layout.addWidget(rule())
        self.btn_advanced = QPushButton("ADVANCED")
        self.btn_advanced.setObjectName("Ghost")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.btn_advanced)

        self.advanced = QWidget()
        adv = QVBoxLayout(self.advanced)
        adv.setContentsMargins(0, 8, 0, 0)
        adv.setSpacing(8)

        self.provider = QComboBox()
        self.provider.addItem("Esri World Imagery (map)", "esri")
        self.provider.addItem("Carto Dark Matter (map only)", "carto_dark")
        self.provider.addItem("OpenStreetMap (map only)", "osm")
        self.provider.currentIndexChanged.connect(
            lambda: self.map.set_provider(self.provider.currentData()))
        adv.addWidget(section("map style"))
        adv.addWidget(self.provider)
        adv.addWidget(caption("Survey always fetches Esri satellite tiles. "
                              "OSM/Carto are for orientation only — they have no shadows."))

        self.tile_budget = LabeledSlider("tile budget", 4, 100, 36)
        adv.addWidget(self.tile_budget)

        adv.addWidget(section("illumination"))
        self.when = QDateTimeEdit(QDateTime.currentDateTimeUtc())
        self.when.setDisplayFormat("yyyy-MM-dd  HH:mm 'UTC'")
        self.when.setCalendarPopup(True)
        self.when.dateTimeChanged.connect(
            lambda: self._sync_sun(*self.map.aoi_centre_or_view()))
        adv.addWidget(self.when)

        btn_best = QPushButton("JUMP TO BEST HOUR")
        btn_best.setObjectName("Ghost")
        btn_best.setToolTip("The time of day whose solar elevation gives the most "
                            "reliable shadow length — not noon, which gives the worst")
        btn_best.clicked.connect(self._use_best_hour)
        adv.addWidget(btn_best)

        self.sun_note = caption("")
        adv.addWidget(self.sun_note)

        self.auto_sun = QPushButton("USE SHADOWS IN IMAGE")
        self.auto_sun.setObjectName("Ghost")
        self.auto_sun.setCheckable(True)
        self.auto_sun.setChecked(True)
        self.auto_sun.setToolTip("Basemap mosaics carry no capture time. This reads the "
                                 "azimuth off the shadows themselves and uses it when "
                                 "the estimate is confident.")
        adv.addWidget(self.auto_sun)

        adv.addWidget(section("structures"))
        self.min_size = LabeledSlider("min footprint", 4, 60, 10, " m")
        adv.addWidget(self.min_size)

        self.btn_survey = QPushButton("SURVEY AREA")
        self.btn_survey.setObjectName("Ghost")
        self.btn_survey.clicked.connect(self.run_survey)
        adv.addWidget(self.btn_survey)

        adv.addWidget(section("measure heights"))
        self.policy = QComboBox()
        self.policy.addItems(["auto", "learned", "greedy"])
        self.policy.setToolTip("auto uses the trained policy when one is loaded;\n"
                               "greedy is the training-free hill-climbing baseline")
        adv.addWidget(self.policy)

        self.measure_limit = LabeledSlider("structures per run", 1, 80, 16)
        adv.addWidget(self.measure_limit)

        self.btn_measure = QPushButton("MEASURE ALL")
        self.btn_measure.setObjectName("Ghost")
        self.btn_measure.setEnabled(False)
        self.btn_measure.clicked.connect(self.run_measure)
        adv.addWidget(self.btn_measure)

        self.btn_measure_one = QPushButton("MEASURE SELECTED")
        self.btn_measure_one.setObjectName("Ghost")
        self.btn_measure_one.setEnabled(False)
        self.btn_measure_one.clicked.connect(self.run_measure_selected)
        adv.addWidget(self.btn_measure_one)

        adv.addWidget(section("training"))
        train_row = QHBoxLayout()
        train_row.setSpacing(6)
        self.btn_train_rl = QPushButton("POLICY")
        self.btn_train_rl.setObjectName("Ghost")
        self.btn_train_rl.clicked.connect(self.train_rl)
        self.btn_train_cnn = QPushButton("REGRESSOR")
        self.btn_train_cnn.setObjectName("Ghost")
        self.btn_train_cnn.clicked.connect(self.train_cnn)
        train_row.addWidget(self.btn_train_rl)
        train_row.addWidget(self.btn_train_cnn)
        adv.addLayout(train_row)

        self.spark = Sparkline("solar")
        self.spark.setFixedHeight(56)
        adv.addWidget(self.spark)

        self.job_label = QLabel("no run active")
        self.job_label.setObjectName("Mono")
        self.job_label.setWordWrap(True)
        adv.addWidget(self.job_label)

        self.btn_abort = QPushButton("ABORT RUN")
        self.btn_abort.setObjectName("Danger")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self.abort_job)
        adv.addWidget(self.btn_abort)

        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)

        layout.addStretch(1)
        QTimer.singleShot(400, self._reload_places)
        return rail

    # ------------------------------------------------------------------ #
    # Centre - map | AOI over the structure table
    # ------------------------------------------------------------------ #
    def _build_centre(self) -> QWidget:
        self.map = MapCanvas(self.client.base_url)
        self.map.aoiChanged.connect(self._on_aoi)
        self.map.viewChanged.connect(self._on_view)
        self.map.tilesPending.connect(self._on_tiles)
        self.map.suggestionChosen.connect(self._on_suggestion_chosen)
        # convenience used by the sun sync before any AOI exists
        self.map.aoi_centre_or_view = lambda: (
            ((self.map.aoi[1] + self.map.aoi[3]) / 2, (self.map.aoi[0] + self.map.aoi[2]) / 2)
            if self.map.aoi else (self.map.center_lat, self.map.center_lon))

        self.aoi_view = AOIView()
        self.aoi_view.structureSelected.connect(self._on_structure_selected)

        views = QSplitter(Qt.Horizontal)
        views.setChildrenCollapsible(True)
        for widget, title in ((self.map, "WORLD  ·  AUTO-SUGGEST / DRAW AOI"),
                              (self.aoi_view, "SURVEYED AREA  ·  CLICK A STRUCTURE")):
            holder = panel()
            holder_layout = QVBoxLayout(holder)
            holder_layout.setContentsMargins(10, 8, 10, 10)
            holder_layout.setSpacing(6)
            holder_layout.addWidget(section(title))
            holder_layout.addWidget(widget, 1)
            views.addWidget(holder)
        views.setSizes([620, 620])

        table_holder = panel()
        table_layout = QVBoxLayout(table_holder)
        table_layout.setContentsMargins(10, 8, 10, 10)
        table_layout.setSpacing(6)

        head = QHBoxLayout()
        head.addWidget(section("detected structures"))
        head.addStretch(1)
        self.table_note = QLabel("")
        self.table_note.setObjectName("Mono")
        head.addWidget(self.table_note)
        table_layout.addLayout(head)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["#", "SCORE", "SHADOW", "WIDTH m", "HEIGHT m", "±σ", "FLOORS", "LAT, LON"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        table_layout.addWidget(self.table)

        ops = QTabWidget()
        ops.setDocumentMode(True)
        self.timeline = TimelineView()
        self.histogram = HistogramStrip()
        self.copilot = CopilotPane()
        self.copilot.setMinimumHeight(140)

        time_page = QWidget()
        time_l = QVBoxLayout(time_page)
        time_l.setContentsMargins(8, 6, 8, 8)
        time_l.addWidget(caption("ساعت خورشیدی محلی ۱۰، ۱۲، ۱۳، ۱۴، ۱۵، ۱۶ — "
                                 "روی موزاییک بدون تاریخ این افمریس است نه پیکسل برداشت‌شده."))
        self.timeline.setMinimumHeight(168)
        time_l.addWidget(self.timeline, 1)
        self.capture_label = QLabel("پس از RUN FIELD منحنی ارتفاع خورشید و ساعات برداشت این‌جا می‌آید.")
        self.capture_label.setObjectName("Mono")
        self.capture_label.setWordWrap(True)
        time_l.addWidget(self.capture_label)
        ops.addTab(time_page, "SHADOW HOURS")

        rgb_page = QWidget()
        rgb_l = QVBoxLayout(rgb_page)
        rgb_l.setContentsMargins(8, 6, 8, 8)
        self.inspect_label = QLabel("no raster yet")
        self.inspect_label.setObjectName("Mono")
        self.inspect_label.setWordWrap(True)
        rgb_l.addWidget(self.histogram)
        self.spectra = SpectraStrip()
        self.spectra.setMinimumHeight(120)
        rgb_l.addWidget(self.spectra, 1)
        rgb_l.addWidget(self.inspect_label)
        self._rgb_tab = ops.addTab(rgb_page, "RGB / DEPTH")

        copilot_page = QWidget()
        copilot_l = QVBoxLayout(copilot_page)
        copilot_l.setContentsMargins(4, 4, 4, 4)
        copilot_l.addWidget(self.copilot)
        brief_row = QHBoxLayout()
        self.btn_brief = QPushButton("ASK COPILOT")
        self.btn_brief.setObjectName("Ghost")
        self.btn_brief.clicked.connect(self.run_brief)
        self.btn_intel = QPushButton("WEB INTEL")
        self.btn_intel.setObjectName("Ghost")
        self.btn_intel.clicked.connect(self.run_intel)
        self.btn_scene3d = QPushButton("BUILD 3D")
        self.btn_scene3d.setObjectName("Primary")
        self.btn_scene3d.clicked.connect(self.run_scene3d)
        brief_row.addWidget(self.btn_brief)
        brief_row.addWidget(self.btn_intel)
        brief_row.addWidget(self.btn_scene3d)
        copilot_l.addLayout(brief_row)
        ops.addTab(copilot_page, "COPILOT")

        self.field3d = Field3DView()
        self.field3d_window = Field3DDialog(self)

        field_page = QWidget()
        field_l = QVBoxLayout(field_page)
        field_l.setContentsMargins(4, 4, 4, 4)
        field_l.addWidget(caption("اگر سایهٔ فشرده تشخیص شود، حجم با بافت سقف بیرون می‌زند. "
                                 "اگر حجم قابل اعتماد نباشد، نقش برجستهٔ همان تصویر نمایش داده می‌شود. "
                                 "طیف رنگ فقط در تب RGB / DEPTH است."))
        field_l.addWidget(self.field3d, 1)
        self.field_note = caption("")
        field_l.addWidget(self.field_note)
        pop_row = QHBoxLayout()
        self.btn_pop3d = QPushButton("OPEN 3D WINDOW")
        self.btn_pop3d.setObjectName("Ghost")
        self.btn_pop3d.clicked.connect(self._pop_field3d)
        pop_row.addWidget(self.btn_pop3d)
        pop_row.addStretch(1)
        field_l.addLayout(pop_row)

        self.ops = ops
        ops.addTab(field_page, "FIELD 3D")
        self._field3d_tab = ops.count() - 1

        centre = QSplitter(Qt.Vertical)
        centre.addWidget(views)
        centre.addWidget(table_holder)
        centre.addWidget(ops)
        centre.setStretchFactor(0, 3)
        centre.setStretchFactor(1, 1)
        centre.setStretchFactor(2, 1)
        centre.setSizes([520, 200, 220])
        return centre

    # ------------------------------------------------------------------ #
    # Right rail - the numbers
    # ------------------------------------------------------------------ #
    def _build_right_rail(self) -> QWidget:
        rail, layout = scroll_rail(320)

        self.r_height = Readout("estimated height", "metres above ground", "solar")
        self.r_floors = Readout("storeys", "at 3.2 m per floor")
        self.r_sigma = Readout("uncertainty", "1σ, metres", "shadow")
        self.r_conf = Readout("zone confidence", "occlusion · truncation · sun", "signal")
        for widget in (self.r_height, self.r_floors, self.r_sigma, self.r_conf):
            widget.setMinimumHeight(84)

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.addWidget(self.r_height, 0, 0, 1, 2)
        grid.addWidget(self.r_floors, 1, 0)
        grid.addWidget(self.r_sigma, 1, 1)
        grid.addWidget(self.r_conf, 2, 0, 1, 2)
        layout.addLayout(grid)

        layout.addWidget(rule())
        layout.addWidget(section("measurement honesty"))
        self.honesty = QLabel("heights are not claimed until a survey runs")
        self.honesty.setObjectName("Honesty")
        self.honesty.setWordWrap(True)
        layout.addWidget(self.honesty)

        layout.addWidget(rule())
        layout.addWidget(section("pipeline  ·  detect → RL → CNN → 3D"))
        self.pipeline = PipelineStrip()
        layout.addWidget(self.pipeline)
        self.pipeline_note = caption("کادر سازه، شکار PPO/greedy، CNN، ارتفاع اعلامی، حجم.")
        layout.addWidget(self.pipeline_note)

        layout.addWidget(rule())
        layout.addWidget(section("area summary"))
        self.summary = QLabel("no survey yet")
        self.summary.setObjectName("Mono")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        layout.addWidget(rule())
        layout.addWidget(section("solar geometry"))
        self.compass = SunCompass()
        layout.addWidget(self.compass)
        self.compass_note = caption("")
        layout.addWidget(self.compass_note)

        layout.addWidget(rule())
        layout.addWidget(section("detection metrics"))
        self.bars = {
            "roof_score": MetricBar("roof score", "shadow"),
            "shadow_support": MetricBar("shadow support", "shadow"),
            "shadow_len": MetricBar("shadow length (norm.)", "solar"),
            "score": MetricBar("detection score", "solar"),
            "occlusion": MetricBar("occlusion", "signal", invert=True),
            "truncation": MetricBar("truncation", "signal", invert=True),
        }
        for bar in self.bars.values():
            layout.addWidget(bar)

        layout.addWidget(rule())
        layout.addWidget(section("proxy reward"))
        self.reward_bars = {
            "r1_contrast": MetricBar("R1 contrast·isolation", "violet"),
            "r2_structure": MetricBar("R2 edge·entropy", "violet"),
            "r3_azimuth": MetricBar("R3 azimuth coherence", "violet"),
        }
        for bar in self.reward_bars.values():
            layout.addWidget(bar)
        layout.addWidget(caption("None of the three reads a ground-truth height."))

        layout.addWidget(rule())
        self.attribution = caption("")
        layout.addWidget(self.attribution)
        layout.addStretch(1)
        return rail

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    def status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _submit(self, fn: Callable[..., Any], on_done, *args: Any, **kwargs: Any) -> None:
        worker = Worker(fn, *args, **kwargs)
        worker.signals.done.connect(on_done)
        worker.signals.failed.connect(self._on_failed)
        self.pool.start(worker)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.busy = busy
        for name in ("btn_survey", "btn_field", "btn_save_place"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(not busy)
        self.btn_measure.setEnabled(not busy and bool(self.survey_data))
        self.btn_measure_one.setEnabled(not busy and self.selected is not None)
        for name in ("btn_brief", "btn_intel", "btn_scene3d"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(not busy)
        self.aoi_view.set_busy(busy)
        self.status_dot.set_state("solar" if busy else "signal", pulsing=busy)
        if message:
            self.status(message)

    def _when_iso(self) -> str:
        return self.when.dateTime().toPython().replace(tzinfo=timezone.utc).isoformat()

    def _set_when_utc(self, moment: datetime, *, notify: bool = True) -> None:
        """PySide 6.5+ QDateTime needs seconds — five ints is a TypeError."""
        if moment.tzinfo is not None:
            moment = moment.astimezone(timezone.utc)
        qdt = QDateTime(
            QDate(moment.year, moment.month, moment.day),
            QTime(moment.hour, moment.minute, moment.second),
            Qt.TimeSpec.UTC,
        )
        if notify:
            self.when.setDateTime(qdt)
            return
        previous = self.when.blockSignals(True)
        self.when.setDateTime(qdt)
        self.when.blockSignals(previous)

    # ------------------------------------------------------------------ #
    # Map interaction
    # ------------------------------------------------------------------ #
    def _toggle_aoi_mode(self, on: bool) -> None:
        self.map.set_aoi_mode(on)
        self.btn_aoi_mode.setText("DRAWING…" if on else "DRAW AOI")

    def _clear_aoi(self) -> None:
        self.map.clear_aoi()
        self._aoi_bbox = None
        self.aoi_label.setText("no area selected — locate a place to auto-suggest")
        self.status("area cleared")
        self._schedule_suggestions(force=True)

    @Slot(float, float, float, float)
    def _on_aoi(self, west: float, south: float, east: float, north: float) -> None:
        self.btn_aoi_mode.setChecked(False)
        self._aoi_bbox = (west, south, east, north)
        self.map.clear_suggestions()
        self._suggestions = []
        if hasattr(self, "btn_accept_suggest"):
            self.btn_accept_suggest.setEnabled(False)
        self._submit(self.client.map_plan, self._on_plan,
                     (west, south, east, north), int(self.tile_budget.get()))
        self._sync_sun((south + north) / 2, (west + east) / 2)

    @Slot(float, float, float, float)
    def _on_suggestion_chosen(self, west: float, south: float, east: float, north: float) -> None:
        self.status("suggestion accepted as AOI — press RUN FIELD")
        self._on_aoi(west, south, east, north)

    @Slot()
    def accept_suggestion(self) -> None:
        best = (self._suggestions[0] if self._suggestions else None)
        if not best or not best.get("bbox"):
            self.status("no suggestion yet — LOCATE or wait for scan")
            return
        west, south, east, north = (float(v) for v in best["bbox"][:4])
        self.map.set_aoi((west, south, east, north))
        self.map.clear_suggestions()
        self._on_aoi(west, south, east, north)
        name = best.get("name_fa") or best.get("name") or "structure"
        self.status(f"accepted suggestion · {name}")

    def _schedule_suggestions(self, force: bool = False) -> None:
        if self.map.aoi and not force:
            return
        if self.busy and not force:
            return
        self._suggest_timer.start()

    @Slot()
    def _request_suggestions(self) -> None:
        if self.map.aoi or self.busy:
            return
        if not hasattr(self.client, "suggest"):
            return
        lat, lon = self.map.center_lat, self.map.center_lon
        if abs(lat) > 85:
            return
        self._suggest_seq += 1
        seq = self._suggest_seq
        span = 220.0 if self.map.zoom >= 17 else 320.0 if self.map.zoom >= 15 else 480.0
        self.status(f"scanning for structures near {lat:.5f}, {lon:.5f}…")
        self._submit(self.client.suggest, lambda payload, s=seq: self._on_suggestions(payload, s),
                     lat, lon, span_m=span, when=self._when_iso())

    @Slot(object, int)
    def _on_suggestions(self, payload: dict[str, Any], seq: int) -> None:
        if seq != self._suggest_seq or self.map.aoi:
            return
        items = list(payload.get("items") or [])
        self._suggestions = items
        self.map.set_suggestions(items)
        if hasattr(self, "btn_accept_suggest"):
            self.btn_accept_suggest.setEnabled(bool(items))
        if not items:
            self.aoi_label.setText(
                f"no structure suggested near {payload.get('lat'):.5f}, {payload.get('lon'):.5f}\n"
                "DRAW AOI manually or move the map")
            self.status("no auto-suggestion — draw an AOI")
            return
        best = items[0]
        name = best.get("name_fa") or best.get("name") or "structure"
        w = best.get("width_m")
        d = best.get("depth_m")
        h = best.get("stated_height_m")
        line = f"پیشنهاد: {name}"
        if w and d:
            line += f"  ·  {float(w):.0f}×{float(d):.0f} m"
        if h:
            line += f"  ·  H {float(h):.0f}"
        line += f"\n{len(items)} candidate(s) — click dashed box or ACCEPT"
        self.aoi_label.setText(line)
        self.status(f"suggested {name} · click dashed box or ACCEPT SUGGESTION")

    @Slot(object)
    def _on_plan(self, plan: dict[str, Any]) -> None:
        verdict = "" if plan["affordable"] else "  ·  will drop a zoom level to fit"
        lat = lon = None
        if self._aoi_bbox:
            west, south, east, north = self._aoi_bbox
            lat, lon = (south + north) / 2, (west + east) / 2
        geo = f"\ncentre {lat:.5f}, {lon:.5f}" if lat is not None else ""
        self.aoi_label.setText(
            f"{plan['width_m']:,.0f} × {plan['height_m']:,.0f} m{geo}\n"
            f"z{plan['zoom']}  ·  {plan['gsd_m']:.2f} m/px  ·  "
            f"{plan['tiles']} tiles ({plan['width_px']}×{plan['height_px']} px){verdict}")

    @Slot(float, float, int)
    def _on_view(self, lon: float, lat: float, zoom: int) -> None:
        self.status(f"view {lat:+.5f}, {lon:+.5f}  ·  zoom {zoom}  ·  "
                    f"{self.map.gsd_m:.2f} m/px")
        self._schedule_suggestions()

    @Slot(int)
    def _on_tiles(self, pending: int) -> None:
        if pending:
            self.status_dot.set_state("shadow", pulsing=True)
        elif not self.busy:
            self.status_dot.set_state("signal")

    @Slot()
    def go_to_place(self) -> None:
        query = self.search.text().strip()
        if not query:
            return
        parts = query.replace(";", ",").split(",")
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                self.map.clear_aoi()
                self._aoi_bbox = None
                self.map.set_center(lon, lat, max(self.map.zoom, 17))
                self.status(f"centred on {lat:+.5f}, {lon:+.5f} — auto-suggest…")
                self._sync_sun(lat, lon)
                self._schedule_suggestions(force=True)
                return
            except ValueError:
                pass
        self.status(f"searching for “{query}”…")
        self._submit(self.client.geocode, self._on_geocode, query, 1)

    @Slot(object)
    def _on_geocode(self, hits: list[dict[str, Any]]) -> None:
        if not hits:
            self.status("no match — try coordinates as  lat, lon")
            return
        hit = hits[0]
        self.map.clear_aoi()
        self._aoi_bbox = None
        if hit.get("bbox"):
            west, south, east, north = hit["bbox"]
            # A city bbox is far larger than a sensible AOI; frame it, do not select it.
            self.map.set_center((west + east) / 2, (south + north) / 2, 17)
        else:
            self.map.set_center(hit["lon"], hit["lat"], 17)
        self.status(f"found {hit['name'][:90]} — auto-suggest…")
        self._sync_sun(hit["lat"], hit["lon"])
        self._reload_places()
        self._schedule_suggestions(force=True)

    # ------------------------------------------------------------------ #
    # Sun
    # ------------------------------------------------------------------ #
    def _sync_sun(self, lat: float, lon: float) -> None:
        self._submit(self.client.sun, self._on_sun, lat, lon, self._when_iso(), True)

    @Slot(object)
    def _on_sun(self, sun: dict[str, Any]) -> None:
        self._best_when = sun.get("best_when")
        if (not sun.get("is_daylight") and sun.get("best_when")
                and not self._sun_autoshift):
            self._sun_autoshift = True
            self.status("sun is below the horizon — jumping to today's best shadow hour")
            self._use_best_hour()
            return
        self._sun_autoshift = False
        geometry = SunGeometry(sun["azimuth_deg"], max(sun["elevation_deg"], 0.1), 0.5)
        self.compass.set_sun(geometry.azimuth_deg, geometry.elevation_deg, sun["quality"])

        if not sun["is_daylight"]:
            verdict = "sun is below the horizon — no shadows at this time"
        elif sun["quality"] > 0.66:
            verdict = "long, clean shadows — ideal metrology"
        elif sun["quality"] > 0.33:
            verdict = "usable geometry"
        else:
            verdict = "poor geometry — shadows too short or merging"
        self.compass_note.setText(
            f"Shadow cast toward {sun['shadow_bearing_deg']:.0f}°. {verdict}.")
        self.sun_note.setText(
            f"elev {sun['elevation_deg']:.1f}°  ·  az {sun['azimuth_deg']:.1f}°  ·  "
            f"Q {sun['quality']:.2f}\nbest today {(sun.get('best_when') or '—')[11:16]} UTC  ·  "
            f"daylight {(sun.get('sunrise') or '—')[11:16]}–{(sun.get('sunset') or '—')[11:16]}")
        seasons = sun.get("seasons") or []
        if seasons:
            bits = [f"{s.get('season','?')[0].upper()} {float(s.get('quality') or 0):.2f}"
                    for s in seasons if isinstance(s, dict)]
            pick = sun.get("season") or ""
            self.sun_note.setText(self.sun_note.text() + f"\nyear {pick}: " + "  ".join(bits))

    def _use_best_hour(self) -> None:
        best = getattr(self, "_best_when", None)
        if not best:
            self.status("best hour not known yet — pick a location first")
            return
        moment = datetime.fromisoformat(best.replace("Z", "+00:00"))
        self._set_when_utc(moment)
        self.status(f"acquisition time set to {best[11:16]} UTC — the best geometry today")

    def _toggle_advanced(self, on: bool) -> None:
        self.advanced.setVisible(on)
        self.btn_advanced.setText("ADVANCED ▾" if on else "ADVANCED")

    def _reload_places(self) -> None:
        if not hasattr(self.client, "favorites"):
            return
        self._submit(lambda: {
            "favorites": self.client.favorites(),
            "history": self.client.place_history(),
        }, self._on_places_loaded)

    @Slot(object)
    def _on_places_loaded(self, payload: dict[str, Any]) -> None:
        favs = payload.get("favorites") or []
        hist = payload.get("history") or []
        self.fav_box.blockSignals(True)
        self.fav_box.clear()
        self.fav_box.addItem("— favorites —", None)
        for item in favs:
            label = f"{item.get('name', 'place')}  {item.get('lat', 0):.4f},{item.get('lon', 0):.4f}"
            self.fav_box.addItem(label[:72], item)
        self.fav_box.blockSignals(False)
        self.hist_box.blockSignals(True)
        self.hist_box.clear()
        self.hist_box.addItem("— history —", None)
        for item in hist:
            kind = item.get("kind") or "event"
            name = item.get("name") or item.get("query") or ""
            when = str(item.get("when") or "")[11:16]
            label = f"{kind} {when} {name}".strip()
            self.hist_box.addItem(label[:72], item)
        self.hist_box.blockSignals(False)

    @Slot(int)
    def _on_favorite_picked(self, index: int) -> None:
        item = self.fav_box.itemData(index)
        if not isinstance(item, dict):
            return
        self._goto_saved(item)

    @Slot(int)
    def _on_history_picked(self, index: int) -> None:
        item = self.hist_box.itemData(index)
        if not isinstance(item, dict):
            return
        self._goto_saved(item)

    def _goto_saved(self, item: dict[str, Any]) -> None:
        lat, lon = item.get("lat"), item.get("lon")
        if lat is None or lon is None:
            return
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            self.map.set_center((float(bbox[0]) + float(bbox[2])) / 2,
                                (float(bbox[1]) + float(bbox[3])) / 2, 16)
        else:
            self.map.set_center(float(lon), float(lat), 16)
        if item.get("name"):
            self.search.setText(str(item["name"])[:90])
        self._sync_sun(float(lat), float(lon))
        self.status(f"centred on {item.get('name') or ''} {float(lat):+.5f}, {float(lon):+.5f}")

    @Slot()
    def save_place(self) -> None:
        if not hasattr(self.client, "add_favorite"):
            self.status("places endpoint not loaded")
            return
        lat, lon = self.map.aoi_centre_or_view()
        name = self.search.text().strip() or f"{lat:.4f}, {lon:.4f}"
        bbox = list(self.map.aoi) if self.map.aoi else None
        self._submit(self.client.add_favorite, self._on_place_saved, name, lat, lon, bbox)

    @Slot(object)
    def _on_place_saved(self, item: dict[str, Any]) -> None:
        self.status(f"saved {item.get('name', 'place')}")
        self._reload_places()

    @Slot()
    def run_field(self) -> None:
        if self.busy:
            return
        if not self.map.aoi:
            if self._suggestions and self._suggestions[0].get("bbox"):
                self.accept_suggestion()
            if not self.map.aoi:
                self.status("no AOI — LOCATE for auto-suggest, or DRAW AOI")
                return
        if not hasattr(self.client, "run_field"):
            self.status("field endpoint not loaded — restart the deck")
            return
        query = self.search.text().strip() or None
        self._set_busy(True, "RUN FIELD — فصل‌ها، طیف رنگ، اندازه‌گیری، ۳بعدی…")
        self._submit(self.client.run_field, self._on_field, self.map.aoi,
                     locale="fa", query=query)

    @Slot(object)
    def _on_field(self, payload: dict[str, Any]) -> None:
        survey = payload.get("survey") or payload
        self._on_survey(survey, follow=False)
        if payload.get("inspect"):
            self._on_inspect(payload["inspect"])
        if payload.get("timeline"):
            self._on_timeline(payload["timeline"])
        measures = payload.get("measures") or {}
        if isinstance(measures, dict) and measures.get("items"):
            self._on_measure(measures)
        brief = dict(payload.get("brief") or {})
        construct = payload.get("construct") or {}
        if construct:
            brief.update({k: v for k, v in construct.items() if k not in {"tools"}})
            brief["construct"] = construct
            self._construct_payload = construct
        if brief:
            self._on_brief(brief)
        if payload.get("intel"):
            self._on_intel(payload["intel"])
        mosaic = decode_png(survey.get("image_png") or survey.get("overlay_png"))
        self._bind_field3d(mosaic, payload.get("inspect"), construct or None)
        if payload.get("measures") and hasattr(self, "pipeline"):
            self.pipeline.set_trace({
                **(payload.get("measures") or {}),
                "survey": survey,
                "viable": self.field3d.viable(),
            })
        plan = payload.get("plan") or {}
        season = (survey.get("sun") or {}).get("season") or survey.get("season") or "—"
        self.status(
            f"field ready · plan {plan.get('source', 'heuristic')} · season {season} · "
            f"{survey.get('count', 0)} structures"
            + ("" if self.field3d.viable() else "  ·  3D relief (spectra on RGB / DEPTH)")
        )
        if hasattr(self, "ops"):
            tab = (self._field3d_tab if self.field3d.viable() else 0)
            self.ops.setCurrentIndex(tab)
        self._reload_places()

    # ------------------------------------------------------------------ #
    # Survey / measure
    # ------------------------------------------------------------------ #
    @Slot()
    def run_survey(self) -> None:
        if self.busy:
            return
        if not self.map.aoi:
            self.status("draw an area on the map first — press DRAW AOI, then drag")
            return
        self._set_busy(True, "fetching imagery, placing the sun, counting structures…")
        self._submit(self.client.survey, self._on_survey, self.map.aoi,
                     provider="esri",
                     max_tiles=int(self.tile_budget.get()),
                     when=self._when_iso(),
                     auto_sun=self.auto_sun.isChecked(),
                     min_size_m=self.min_size.get(),
                     max_structures=80)

    @Slot(object)
    def _on_survey(self, payload: dict[str, Any], follow: bool = True) -> None:
        self._set_busy(False)
        self.survey_data = payload
        self.measured.clear()
        self.selected = None

        image = decode_png(payload.get("overlay_png") or payload.get("image_png"))
        self.aoi_view.set_image(image, payload["structures"])

        scene, sun = payload["scene"], payload["sun"]
        used = sun.get("when")
        if used:
            try:
                moment = datetime.fromisoformat(str(used).replace("Z", "+00:00"))
                self._set_when_utc(moment, notify=False)
            except (ValueError, TypeError):
                pass
        centre = scene.get("center") or []
        geo_line = ""
        if len(centre) >= 2:
            geo_line = f"\ncentre {float(centre[1]):.5f}, {float(centre[0]):.5f}"
        snap = "  ·  clock was night, using best hour" if sun.get("snapped_to_best_hour") else ""
        season = sun.get("season") or payload.get("season") or ""
        if season:
            snap += f"  ·  {season}"
        self.count_label.setText(
            f"{payload['count']} structures  ·  {payload['with_shadow']} with a usable shadow\n"
            f"mean detection score {payload['mean_score']:.2f}")
        self.summary.setText(
            f"{scene['width_m']:,.0f} × {scene['height_m']:,.0f} m  ·  z{scene['zoom']}  ·  "
            f"{scene['gsd_m']:.2f} m/px{geo_line}\n"
            f"tiles {scene['tiles']}  ·  "
            f"shadow {payload['shadow_coverage'] * 100:.0f}%  ·  "
            f"water {payload['water_coverage'] * 100:.0f}%\n"
            f"survey took {payload['elapsed_ms']:.0f} ms{snap}")
        self.attribution.setText(f"Imagery: {scene['attribution']}. "
                                 f"Geocoding: OpenStreetMap Nominatim.")

        geometry = SunGeometry(sun["azimuth_deg"], max(sun["elevation_deg"], 0.1), sun["gsd_m"])
        self.compass.set_sun(geometry.azimuth_deg, geometry.elevation_deg, sun["quality"])
        if sun.get("snapped_to_best_hour"):
            self.compass_note.setText(
                f"Shadow cast toward {sun['shadow_bearing_deg']:.0f}°. "
                "Clock was night — using today's best shadow hour.")
        elif sun.get("is_daylight"):
            self.compass_note.setText(
                f"Shadow cast toward {sun['shadow_bearing_deg']:.0f}°. usable geometry.")

        estimate = payload.get("sun_estimate") or {}
        if estimate:
            applied = "APPLIED" if estimate.get("applied") else "not applied"
            self.status(f"{payload['count']} structures  ·  image-derived shadow bearing "
                        f"{estimate.get('shadow_bearing_deg', 0):.0f}° "
                        f"(confidence {estimate.get('confidence', 0):.2f}, {applied})")
        else:
            self.status(f"{payload['count']} structures detected")

        self._fill_table()
        self.btn_measure.setEnabled(True)
        landmark = payload.get("landmark") or {}
        stated = None
        structs = payload.get("structures") or []
        for s in structs:
            if s.get("stated_height_m") is not None:
                stated = s["stated_height_m"]
                break
        if stated is None and landmark.get("height_m") is not None:
            stated = landmark["height_m"]
        self.r_height.set_value(str(payload["count"]), "structures detected in this area")
        self.r_floors.set_value("—", "measure to populate")
        self.r_sigma.set_value("—", "1σ, metres")
        self.r_conf.set_value(f"{payload['mean_score'] * 100:.0f}%", "mean detection score")
        if hasattr(self, "pipeline"):
            self.pipeline.set_trace({"survey": payload, "structures": structs})
            span = ""
            if structs:
                span = (f"{float(structs[0].get('width_m') or 0):.0f}×"
                        f"{float(structs[0].get('height_m') or 0):.0f} m")
            self.pipeline_note.setText(
                f"DETECT {span or '—'}  ·  {payload['count']} box"
                + (f"  ·  {landmark.get('name')} {float(stated):.0f} m stated" if stated else "")
            )
        self._apply_honesty(payload.get("honesty") or {})
        if follow:
            self._follow_survey(payload)

    def _apply_honesty(self, honesty: dict[str, Any]) -> None:
        verdict = honesty.get("verdict", "undated_basemap")
        indicative = honesty.get("height_indicative", True)
        line = "INDICATIVE metres — undated mosaic" if indicative else "dated scene — elevation trusted"
        warns = honesty.get("warnings") or []
        fa = next((w for w in warns if any(ord(c) > 127 for c in w)), warns[0] if warns else "")
        self.honesty.setText(f"{verdict}  ·  {line}\n{fa}")

    def _follow_survey(self, payload: dict[str, Any]) -> None:
        aoi_id = payload.get("aoi_id")
        sun = payload.get("sun") or {}
        scene = payload.get("scene") or {}
        center = scene.get("center") or [self.map.center_lon, self.map.center_lat]
        lon, lat = float(center[0]), float(center[1])
        if hasattr(self.client, "timeline"):
            self._submit(self.client.timeline, self._on_timeline,
                         lat, lon, sun.get("when") or self._when_iso())
        if aoi_id and hasattr(self.client, "inspect_aoi"):
            self._submit(self.client.inspect_aoi, self._on_inspect, aoi_id)
        if aoi_id and hasattr(self.client, "brief"):
            self._submit(self.client.brief, self._on_brief, aoi_id)

    @Slot()
    def run_brief(self) -> None:
        if not self.survey_data:
            self.status("survey an area first")
            return
        fn = getattr(self.client, "brief", None)
        if fn is None:
            self.status("copilot endpoint not loaded yet")
            return
        self.status("copilot writing a brief…")
        self._submit(fn, self._on_brief, self.survey_data["aoi_id"])

    @Slot()
    def run_intel(self) -> None:
        if not self.map.aoi:
            self.status("draw an area first")
            return
        fn = getattr(self.client, "intel", None)
        if fn is None:
            self.status("intel endpoint not loaded yet")
            return
        self.status("querying OSM / Wikipedia…")
        self._submit(fn, self._on_intel, self.map.aoi)

    def _bind_field3d(self, mosaic, inspect: dict[str, Any] | None = None,
                      construct: dict[str, Any] | None = None) -> None:
        if not self.survey_data:
            return
        measures = list(self.measured.values())
        inspect_payload = inspect if inspect is not None else getattr(self, "_inspect_payload", None)
        construct_payload = construct if construct is not None else getattr(self, "_construct_payload", None)
        self.field3d.load_survey(self.survey_data, measures, mosaic,
                                 inspect=inspect_payload, construct=construct_payload)
        self.field3d_window.load_survey(self.survey_data, measures, mosaic,
                                        inspect=inspect_payload, construct=construct_payload)
        fallback = not self.field3d.viable()
        if hasattr(self, "field_note"):
            if fallback:
                self.field_note.setText(
                    "حجم سه‌بعدی از این سایه قابل اعتماد نیست. "
                    "طیف رنگ در تب RGB / DEPTH است؛ ساعت خورشیدی در SHADOW HOURS.")
            else:
                kind = "prism"
                for b in (self.field3d._scene or {}).get("buildings") or []:
                    if b.get("massing_kind"):
                        kind = str(b["massing_kind"])
                        break
                self.field_note.setText(
                    f"عامل‌ها: massing={kind} · ارتفاع اعلامی روی بنای شناخته‌شده · "
                    "شکار RL/CNN فقط INDICATIVE. طیف در RGB / DEPTH.")

    @Slot()
    def run_scene3d(self) -> None:
        if not self.survey_data:
            self.status("survey an area first")
            return
        mosaic = decode_png(self.survey_data.get("image_png") or self.survey_data.get("overlay_png"))
        self._bind_field3d(mosaic)
        if hasattr(self, "ops"):
            tab = (self._field3d_tab if self.field3d.viable()
                   else getattr(self, "_rgb_tab", self._field3d_tab))
            self.ops.setCurrentIndex(tab)
        self.field3d_window.show()
        self.field3d_window.raise_()
        self.field3d_window.activateWindow()
        if self.field3d.viable():
            self.status("live 3D in FIELD 3D tab and reconstruction window")
        else:
            self.status("3D volume not trusted — mosaic relief in FIELD 3D, spectra on RGB / DEPTH")
        fn = getattr(self.client, "scene3d", None)
        if fn is None:
            return
        self._submit(fn, self._on_scene3d, self.survey_data["aoi_id"],
                     list(self.measured.values()), False)

    def _pop_field3d(self) -> None:
        self.field3d_window.show()
        self.field3d_window.raise_()
        self.field3d_window.activateWindow()

    @Slot(object)
    def _on_scene3d(self, payload: dict[str, Any]) -> None:
        n = payload.get("buildings", 0)
        folder = payload.get("dir") or payload.get("html") or ""
        self.status(f"workspace saved ({n} volumes)  ·  {folder}")

    @Slot(object)
    def _on_brief(self, brief: dict[str, Any]) -> None:
        self.copilot.set_brief(brief)
        if brief.get("honesty"):
            self._apply_honesty(brief["honesty"])
        self.status("copilot brief ready")

    @Slot(object)
    def _on_timeline(self, payload: dict[str, Any]) -> None:
        samples = payload.get("samples") or payload.get("hours") or []
        if not samples:
            calendar = payload.get("calendar") or {}
            seasons = calendar.get("seasons") if isinstance(calendar, dict) else []
            for season in seasons or []:
                samples.extend(season.get("slots") or [])
        strip = decode_png(payload.get("strip_png") or payload.get("png"))
        if strip is None and samples:
            try:
                from ....models.timeline import render_timeline_strip
                strip = render_timeline_strip(samples)
            except Exception:
                strip = None
        note = ""
        if payload.get("best_when"):
            note = f"best hour {str(payload['best_when'])[11:16]} UTC"
        captures = payload.get("captures") or []
        if not captures:
            calendar = payload.get("calendar") or {}
            seasons = calendar.get("seasons") if isinstance(calendar, dict) else []
            for season in seasons or []:
                captures.extend(season.get("slots") or [])
        self.timeline.set_strip(strip, samples, note, captures=captures)
        calendar = payload.get("calendar")
        bits: list[str] = []
        if captures:
            hours = sorted({int(c.get("local_hour", 0)) for c in captures if c.get("local_hour")})
            if hours:
                bits.append("local " + ", ".join(f"{h:02d}:00" for h in hours))
            day = sum(1 for c in captures if c.get("is_daylight"))
            bits.append(f"{day}/{len(captures)} daylight")
        if isinstance(calendar, dict) and calendar.get("seasons"):
            bits.append(f"{len(calendar['seasons'])} seasons")
        if hasattr(self, "capture_label"):
            self.capture_label.setText("  ·  ".join(bits) if bits else note or "solar clock")

    @Slot(object)
    def _on_inspect(self, payload: dict[str, Any]) -> None:
        self._inspect_payload = payload
        stats = payload.get("inspect") or payload
        self.histogram.set_from_inspect(stats)
        vis = payload if payload.get("depth_png") or payload.get("false_png") else stats
        if hasattr(self, "spectra"):
            self.spectra.set_from_inspect(vis)
        depth = stats.get("depth_proxy") or payload.get("depth_proxy") or {}
        clip = stats.get("clipping") or {}
        clip_hi = 0.0
        if isinstance(clip, dict):
            clip_hi = float(clip.get("at_255") or clip.get("hi") or 0)
        spectra = stats.get("spectra") or {}
        spec_note = ""
        if isinstance(spectra, dict) and spectra:
            spec_note = "  ·  spectra " + "/".join(spectra.keys())
        self.inspect_label.setText(
            f"layout {stats.get('layout', 'BGR')}  ·  {stats.get('bit_depth', '?')}-bit  ·  "
            f"{stats.get('channels', '?')} ch  ·  "
            f"range {float(stats.get('dynamic_range') or 0):.2f}  ·  clip {clip_hi:.1%}\n"
            f"depth proxy {depth.get('method', 'shadow_luma')}  ·  "
            f"mean {float(depth.get('mean') or 0):.2f}  (not a LiDAR sensor){spec_note}"
        )

    @Slot(object)
    def _on_intel(self, payload: dict[str, Any]) -> None:
        osm = payload.get("overpass") or payload.get("osm") or payload
        wiki = payload.get("wikipedia") or payload.get("wiki") or []
        n = osm.get("count", len(osm.get("items") or [])) if isinstance(osm, dict) else 0
        titles = ", ".join((w.get("title") or "") for w in (wiki if isinstance(wiki, list) else [])[:3])
        extra = f"OSM buildings {n}"
        if titles:
            extra += f"  ·  wiki {titles}"
        place = (payload.get("reverse") or {}).get("name") or (payload.get("place") or "")
        if place:
            extra = f"{place}\n{extra}"
        self.summary.setText(self.summary.text() + "\n" + extra)
        self.status(extra.replace("\n", "  ·  "))

    @Slot()
    def run_measure(self) -> None:
        if self.busy or not self.survey_data:
            return
        self._set_busy(True, "hunting each structure's shadow…")
        self._submit(self.client.aoi_analyze, self._on_measure,
                     self.survey_data["aoi_id"], policy=self.policy.currentText(),
                     limit=int(self.measure_limit.get()), max_steps=40)

    @Slot()
    def run_measure_selected(self) -> None:
        if self.busy or not self.survey_data or self.selected is None:
            return
        self._set_busy(True, f"measuring structure #{self.selected + 1}…")
        self._submit(self.client.aoi_analyze, self._on_measure,
                     self.survey_data["aoi_id"], indices=[self.selected],
                     policy=self.policy.currentText(), limit=1, max_steps=48)

    @Slot(object)
    def _on_measure(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        for item in payload["items"]:
            self.measured[item["index"]] = item

        image = decode_png(payload.get("overlay_png"))
        if image is not None and self.survey_data:
            self.aoi_view.set_image(image, self.survey_data["structures"])
            if self.selected is not None:
                self.aoi_view.select(self.selected)

        tallest = payload.get("tallest")
        if tallest:
            stated = tallest.get("stated_height_m")
            self.r_height.set_value(
                f"{tallest['height_m']:.1f}",
                f"INDICATIVE ±{tallest['sigma_m']:.2f} m · fused hunt")
            self.r_floors.set_value(str(tallest["floors"]), "storeys, tallest structure")
            self.r_sigma.set_value(f"{tallest['sigma_m']:.2f}", "1σ, metres")
            self.r_conf.set_value(f"{tallest['confidence'] * 100:.0f}%", "zone confidence")
            hunt_box = tallest.get("box")
            if hunt_box:
                self.aoi_view.set_hunt(hunt_box, tallest.get("trajectory"))
            if hasattr(self, "pipeline"):
                self.pipeline.set_trace({
                    **payload,
                    "survey": self.survey_data,
                    "viable": self.field3d.viable() if hasattr(self, "field3d") else False,
                })
                cnn = tallest.get("cnn_m")
                geom_m = tallest.get("geometric_m")
                pol = tallest.get("policy") or "—"
                g_txt = "—" if geom_m is None else f"{float(geom_m):.1f} m"
                cnn_txt = "off" if cnn is None else f"{float(cnn):.1f} m"
                self.pipeline_note.setText(
                    f"RL {pol} · {tallest.get('steps') or 0} steps  ·  "
                    f"g {g_txt} / cnn {cnn_txt}  ·  "
                    f"fuse {tallest['height_m']:.1f} m INDICATIVE"
                    + (f"  ·  stated {float(stated):.0f} m" if stated else "")
                )

        self.summary.setText(
            f"{len(payload['items'])} measured  ·  mean {payload['mean_height_m']:.1f} m\n"
            f"tallest {tallest['height_m']:.1f} m at "
            f"{tallest['lat']:.5f}, {tallest['lon']:.5f}\n"
            f"{payload['total_floors']} storeys in total  ·  "
            f"{payload['elapsed_ms']:.0f} ms" if tallest else "nothing measured")

        self._fill_table()
        self.status(f"measured {len(payload['items'])} structures  ·  "
                    f"mean {payload['mean_height_m']:.1f} m  ·  "
                    f"{payload['elapsed_ms']:.0f} ms")

    # ------------------------------------------------------------------ #
    # Table / selection
    # ------------------------------------------------------------------ #
    def _fill_table(self) -> None:
        structures = (self.survey_data or {}).get("structures", [])
        self.table.setRowCount(len(structures))
        for row, structure in enumerate(structures):
            measured = self.measured.get(row)
            src = measured or structure
            lat, lon = src.get("lat"), src.get("lon")
            coord = (f"{float(lat):.5f}, {float(lon):.5f}"
                     if lat is not None and lon is not None else "—")
            values = [
                str(row + 1),
                f"{structure['score']:.2f}",
                f"{structure['shadow_support']:.2f}",
                f"{structure['width_m']:.0f}",
                f"{measured['height_m']:.1f}" if measured else f"{structure.get('quick_height_m', 0):.0f}",
                f"{measured['sigma_m']:.2f}" if measured else "—",
                str(measured["floors"]) if measured else "—",
                coord,
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table_note.setText(f"{len(self.measured)} of {len(structures)} measured")

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self._select_structure(rows[0].row(), from_table=True)

    @Slot(int)
    def _on_structure_selected(self, index: int) -> None:
        self._select_structure(index, from_table=False)

    def _select_structure(self, index: int, from_table: bool) -> None:
        self.selected = index
        self.btn_measure_one.setEnabled(not self.busy and self.survey_data is not None)
        self.aoi_view.select(index)
        if not from_table and 0 <= index < self.table.rowCount():
            self.table.selectRow(index)

        structures = (self.survey_data or {}).get("structures", [])
        if not 0 <= index < len(structures):
            return
        structure = structures[index]
        measured = self.measured.get(index)

        if measured:
            self.r_height.set_value(
                f"{measured['height_m']:.1f}",
                f"INDICATIVE ±{measured['sigma_m']:.2f} m · fused hunt")
            self.r_floors.set_value(str(measured["floors"]), "at 3.2 m per floor")
            self.r_sigma.set_value(f"{measured['sigma_m']:.2f}", "1σ, metres")
            self.r_conf.set_value(f"{measured['confidence'] * 100:.0f}%",
                                  "occlusion · truncation · sun")
            self.bars["occlusion"].set_value(measured["occlusion"])
            self.bars["truncation"].set_value(float(measured.get("truncation") or 0.0))
            self.status(f"structure #{index + 1}  ·  {measured['height_m']:.1f} m INDICATIVE "
                        f"±{measured['sigma_m']:.2f}  ·  {measured['lat']:.5f}, "
                        f"{measured['lon']:.5f}")
            hunt_box = measured.get("box")
            if hunt_box:
                self.aoi_view.set_hunt(hunt_box, measured.get("trajectory"))
        else:
            self.r_height.set_value(f"{structure['quick_height_m']:.0f}",
                                    "metres — rough strip estimate, not measured")
            self.r_floors.set_value("—", "press MEASURE SELECTED")
            self.r_sigma.set_value("—", "1σ, metres")
            self.status(f"structure #{index + 1}  ·  {structure['width_m']:.0f} m wide  ·  "
                        f"detection score {structure['score']:.2f}  ·  not measured yet")

        self.bars["roof_score"].set_value(structure["roof_score"])
        self.bars["shadow_support"].set_value(structure["shadow_support"])
        self.bars["shadow_len"].set_value(min(1.0, structure["shadow_len_px"] / 120.0))
        self.bars["score"].set_value(structure["score"])

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    @Slot()
    def train_rl(self) -> None:
        self.spark.clear()
        self._submit(self.client.train_rl, self._on_job_started,
                     algo="PPO", total_timesteps=20_000, tag="shadow_hunter")

    @Slot()
    def train_cnn(self) -> None:
        self.spark.clear()
        self._submit(self.client.train_cnn, self._on_job_started,
                     scenes=24, epochs=20, tag="height_cnn")

    @Slot(object)
    def _on_job_started(self, job: dict[str, Any]) -> None:
        self.active_job = job["id"]
        self.btn_abort.setEnabled(True)
        self.job_label.setText(f"{job['id']} · {job['kind']} · queued")
        self.job_timer.start()

    @Slot()
    def abort_job(self) -> None:
        if self.active_job:
            self._submit(self.client.abort, lambda _: self.status("abort requested"),
                         self.active_job)

    def _poll_job(self) -> None:
        if not self.active_job:
            self.job_timer.stop()
            return
        self._submit(self.client.job, self._on_job_status, self.active_job)

    @Slot(object)
    def _on_job_status(self, job: dict[str, Any]) -> None:
        self.job_label.setText(f"{job['kind']} · {job['state']} · "
                               f"{job['progress'] * 100:.0f}%\n{job['message'][:60]}")
        if job["state"] in {"done", "failed", "aborted"}:
            self.job_timer.stop()
            self.active_job = None
            self.btn_abort.setEnabled(False)
            self.status(f"job {job['id']} {job['state']} — {job['message']}")
            self._refresh_health()

    def _on_telemetry(self, event: dict[str, Any]) -> None:
        topic = event.get("topic")
        if topic == "rl.progress" and event.get("ep_reward_mean") is not None:
            QTimer.singleShot(0, lambda v=event["ep_reward_mean"]: self.spark.push(v))
        elif topic == "cnn.progress":
            QTimer.singleShot(0, lambda v=event["val_mae"]: self.spark.push(v))

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def _refresh_health(self) -> None:
        self._submit(self.client.health, self._on_health)

    @Slot(object)
    def _on_health(self, health: dict[str, Any]) -> None:
        device = "CUDA" if health.get("cuda") else "CPU"
        policy = "policy ✓" if health.get("policy_loaded") else "policy · greedy"
        cnn = "cnn ✓" if health.get("cnn_loaded") else "cnn · geometric"
        self.health_label.setText(f"{device}  ·  torch {health.get('torch', '?')}  ·  "
                                  f"{policy}  ·  {cnn}")
        if not self.busy:
            self.status_dot.set_state("signal")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.status_dot.set_state("alert")
        self.status(message.splitlines()[0][:240])

    def closeEvent(self, event) -> None:
        self.stream.stop()
        super().closeEvent(event)
