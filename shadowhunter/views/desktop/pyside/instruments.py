"""Copilot, RGB histogram and timeline instruments for the flagship deck."""
from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .widgets import bgr_to_qimage, col, mono
from ....services.client import decode_png


class HistogramStrip(QWidget):
    """Per-channel RGB histogram of the surveyed mosaic."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(92)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._r: list[float] = []
        self._g: list[float] = []
        self._b: list[float] = []
        self._caption = "NO RASTER"

    def set_from_inspect(self, inspect: dict[str, Any] | None) -> None:
        if not inspect:
            self._r = self._g = self._b = []
            self._caption = "NO RASTER"
            self.update()
            return
        hist = inspect.get("histogram") or {}
        self._r = [float(v) for v in hist.get("r") or hist.get("R") or []]
        self._g = [float(v) for v in hist.get("g") or hist.get("G") or []]
        self._b = [float(v) for v in hist.get("b") or hist.get("B") or []]
        mean = inspect.get("rgb_mean") or [0, 0, 0]
        depth = inspect.get("bit_depth", 8)
        ch = inspect.get("channels", 3)
        layout = inspect.get("layout", "BGR")
        self._caption = (
            f"{inspect.get('width', 0)}×{inspect.get('height', 0)}  ·  "
            f"{layout} {ch}ch  {depth}-bit  ·  "
            f"RGB {mean[0]:.0f}/{mean[1]:.0f}/{mean[2]:.0f}"
        )
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), col("void"))
        p.setFont(mono(9))
        p.setPen(col("inkFaint"))
        p.drawText(self.rect().adjusted(6, 4, -6, 0), Qt.AlignLeft | Qt.AlignTop, self._caption)

        series = (
            (self._r, QColor(255, 90, 90, 180)),
            (self._g, QColor(90, 220, 120, 180)),
            (self._b, QColor(80, 180, 255, 180)),
        )
        if not any(series[i][0] for i in range(3)):
            return
        peak = max(max(s) if s else 1.0 for s, _ in series) or 1.0
        pad_top, pad_bot = 22, 8
        h = max(1, self.height() - pad_top - pad_bot)
        w = self.width()
        for values, colour in series:
            if len(values) < 2:
                continue
            p.setPen(QPen(colour, 1.4))
            prev = None
            for i, v in enumerate(values):
                x = i / (len(values) - 1) * (w - 8) + 4
                y = pad_top + h - (v / peak) * h
                pt = (x, y)
                if prev:
                    p.drawLine(int(prev[0]), int(prev[1]), int(pt[0]), int(pt[1]))
                prev = pt


class SpectraStrip(QWidget):
    """Four relative-depth previews: shadow depth, false colour, LAB L*, boost."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(88)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._cells: dict[str, QLabel] = {}
        for key, title in (
            ("depth_png", "DEPTH"),
            ("false_png", "FALSE"),
            ("lab_png", "LAB L*"),
            ("boost_png", "BOOST"),
        ):
            cell = QLabel(title)
            cell.setObjectName("Mono")
            cell.setAlignment(Qt.AlignCenter)
            cell.setMinimumHeight(80)
            cell.setStyleSheet("background:#0e1210; border:1px solid #243028;")
            cell.setScaledContents(True)
            self._cells[key] = cell
            row.addWidget(cell, 1)

    def set_from_inspect(self, inspect: dict[str, Any] | None) -> None:
        payload = inspect or {}
        for key, cell in self._cells.items():
            blob = payload.get(key)
            bgr = decode_png(blob if isinstance(blob, str) else None)
            if bgr is None:
                cell.setPixmap(QPixmap())
                cell.setText(key.replace("_png", "").upper())
                continue
            cell.setPixmap(QPixmap.fromImage(bgr_to_qimage(bgr)))
            cell.setText("")


class TimelineView(QWidget):
    """Day elevation curve plus local civil capture hours 10, 12, 13, 14, 15, 16."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(148)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._image = None
        self._samples: list[dict[str, Any]] = []
        self._captures: list[dict[str, Any]] = []
        self._note = "shadow hours — survey an area first"

    def set_strip(self, bgr: np.ndarray | None, samples: list[dict[str, Any]] | None = None,
                  note: str = "", captures: list[dict[str, Any]] | None = None) -> None:
        self._image = None if bgr is None else bgr_to_qimage(np.ascontiguousarray(bgr))
        if samples is not None:
            self._samples = list(samples)
        if captures is not None:
            self._captures = list(captures)
        if note:
            self._note = note
        elif self._samples:
            day = [s for s in self._samples if s.get("is_daylight")]
            self._note = f"{len(self._samples)} samples  ·  {len(day)} daylight"
        elif self._captures:
            self._note = f"{len(self._captures)} local capture hours"
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), col("void"))
        rows = self._samples or self._captures
        if rows:
            self._paint_samples(p, overlay=False)
            return
        if self._image is not None:
            p.drawImage(self.rect(), self._image)
            p.setFont(mono(9))
            p.setPen(col("inkFaint"))
            p.drawText(self.rect().adjusted(6, 3, -6, 0), Qt.AlignLeft | Qt.AlignTop, self._note)
            return
        p.setFont(mono(9))
        p.setPen(col("inkFaint"))
        p.drawText(self.rect(), Qt.AlignCenter, self._note)

    def _paint_samples(self, p: QPainter, overlay: bool) -> None:
        samples = self._samples or self._captures
        n = len(samples)
        if n < 1:
            return
        w, h = self.width(), self.height()
        pad_top, pad_bot, pad_x = 22, 18, 8
        usable = max(1, h - pad_top - pad_bot)
        span = max(1.0, float(w - 2 * pad_x))

        def sx(i: int) -> float:
            return pad_x + (i / max(n - 1, 1)) * span

        def sy(elev: float) -> float:
            e = max(0.0, min(90.0, float(elev)))
            return pad_top + usable - (e / 90.0) * usable

        night = QColor(0, 0, 0, 50 if overlay else 150)
        col_w = span / n
        for i, sample in enumerate(samples):
            if sample.get("is_daylight"):
                continue
            x0 = pad_x + i * col_w
            p.fillRect(int(x0), 0, max(1, int(col_w + 1)), h, night)

        pts: list[tuple[float, float]] = []
        best_i, best_q = 0, -1.0
        for i, sample in enumerate(samples):
            pts.append((sx(i), sy(float(sample.get("elevation_deg") or 0.0))))
            q = float(sample.get("quality") or 0.0)
            if q > best_q:
                best_i, best_q = i, q

        p.setPen(QPen(col("solar"), 2.0))
        prev = None
        for pt in pts:
            if prev is not None:
                p.drawLine(int(prev[0]), int(prev[1]), int(pt[0]), int(pt[1]))
            prev = pt

        bx, by = pts[best_i]
        p.setPen(QPen(col("shadow"), 1.2))
        p.drawLine(int(bx), pad_top // 2, int(bx), h - pad_bot // 2)
        p.setBrush(col("shadow"))
        p.drawEllipse(int(bx) - 5, int(by) - 5, 10, 10)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(col("shadow"), 1.4))
        p.drawEllipse(int(bx) - 8, int(by) - 8, 16, 16)

        p.setFont(mono(8))
        for cap in self._captures:
            frac = self._utc_frac(cap)
            if frac is None:
                hour = cap.get("local_hour")
                if hour is None:
                    continue
                frac = float(hour) / 24.0
            x = pad_x + frac * span
            day = bool(cap.get("is_daylight"))
            p.setPen(QPen(QColor(63, 211, 228) if day else QColor(90, 72, 48), 1.1))
            p.drawLine(int(x), pad_top // 2, int(x), h - pad_bot)
            p.setBrush(QColor(63, 211, 228) if day else QColor(90, 72, 48))
            elev = float(cap.get("elevation_deg") or 0.0)
            p.drawEllipse(int(x) - 4, int(sy(elev)) - 4, 8, 8)
            p.setPen(QColor(63, 211, 228) if day else col("inkFaint"))
            hour = cap.get("local_hour")
            if hour is not None:
                p.drawText(int(x) - 16, h - 16, 32, 14,
                           Qt.AlignHCenter | Qt.AlignBottom, f"{int(hour):02d}:00")

        if self._note:
            p.setFont(mono(9))
            p.setPen(col("inkFaint"))
            p.drawText(self.rect().adjusted(6, 3, -6, 0), Qt.AlignLeft | Qt.AlignTop, self._note)

    @staticmethod
    def _utc_frac(row: dict[str, Any]) -> float | None:
        when = str(row.get("when") or "")
        if "T" not in when or len(when) < 16:
            return None
        try:
            return (int(when[11:13]) + int(when[14:16]) / 60.0) / 24.0
        except ValueError:
            return None


class PipelineStrip(QWidget):
    """DETECT → RL hunt → CNN → fuse → stated height → 3D extrusion."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._stages: list[tuple[str, str, bool]] = [
            ("DETECT", "کادر سازه", False),
            ("RL", "شکار سایه", False),
            ("CNN", "رگرسور", False),
            ("FUSE", "ارتفاع", False),
            ("STATED", "۴۳ م اعلامی", False),
            ("3D", "حجم", False),
        ]

    def set_trace(self, payload: dict[str, Any] | None) -> None:
        data = payload or {}
        item = data.get("tallest") or {}
        if not item and isinstance(data.get("items"), list) and data["items"]:
            item = data["items"][0]
        survey = data.get("survey") or data
        structures = survey.get("structures") if isinstance(survey, dict) else None
        n = len(structures or data.get("structures") or [])
        box = item.get("detect_box") or item.get("box_geom", {}).get("px") or item.get("box")
        stated = item.get("stated_height_m")
        if stated is None:
            for s in structures or []:
                if s.get("stated_height_m") is not None:
                    stated = s["stated_height_m"]
                    break
        cnn = item.get("cnn_m")
        geom_m = item.get("geometric_m")
        fused = item.get("fused_m")
        policy = str(item.get("policy") or data.get("policy") or "—")
        steps = item.get("steps")
        geom = item.get("box_geom") or {}
        w_m = geom.get("width_m")
        d_m = geom.get("depth_m")
        if not w_m and structures:
            w_m = structures[0].get("width_m")
            d_m = structures[0].get("height_m")
        detect_ok = bool(box) or n > 0
        rl_ok = item.get("policy") is not None or (steps or 0) > 0
        cnn_ok = cnn is not None
        measured_ok = item.get("policy") is not None or item.get("fused_m") is not None
        if fused is None and measured_ok:
            fused = item.get("height_m")
        fuse_ok = measured_ok and fused is not None
        stated_ok = stated is not None
        vol_ok = data.get("viable") is True

        def _m(value: Any) -> str:
            try:
                return f"{float(value):.1f} m"
            except (TypeError, ValueError):
                return "—"

        span = ""
        if w_m and d_m:
            span = f"{float(w_m):.0f}×{float(d_m):.0f} m"
        cnn_detail = "off"
        if cnn_ok or geom_m is not None:
            cnn_detail = f"g {_m(geom_m)} · cnn {_m(cnn) if cnn_ok else 'off'}"
        self._stages = [
            ("DETECT", span or (f"{n} box" if n else "no box"), detect_ok),
            ("RL", f"{policy}" + (f" · {int(steps)} step" if steps else ""), rl_ok),
            ("CNN", cnn_detail, cnn_ok),
            ("FUSE", _m(fused) if fuse_ok else "—", fuse_ok),
            ("STATED", _m(stated) if stated_ok else "none", stated_ok),
            ("3D", "volume" if vol_ok else "relief", vol_ok),
        ]
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), col("void"))
        n = len(self._stages)
        if n < 1:
            return
        pad = 8
        gap = 6
        w = (self.width() - 2 * pad - gap * (n - 1)) / n
        h = self.height() - 10
        p.setFont(mono(8, 700))
        for i, (title, detail, ok) in enumerate(self._stages):
            x = pad + i * (w + gap)
            tone = col("signal") if ok else col("hairline")
            p.setPen(QPen(tone, 1))
            p.setBrush(col("panel") if ok else col("void"))
            p.drawRoundedRect(int(x), 6, max(8, int(w)), int(h), 4, 4)
            p.setPen(col("solar") if ok else col("inkFaint"))
            p.drawText(int(x) + 6, 20, title)
            p.setFont(mono(8))
            p.setPen(col("ink"))
            p.drawText(QRect(int(x) + 6, 38, max(8, int(w) - 10), 28),
                       Qt.TextWordWrap, str(detail)[:28])
            p.setFont(mono(8, 700))
            if i < n - 1:
                p.setPen(col("hairline"))
                nx = x + w
                p.drawLine(int(nx), int(self.height() / 2), int(nx + gap), int(self.height() / 2))


class CopilotPane(QFrame):
    """Human-readable operator brief. RTL-friendly for Persian reports."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Elevated")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        head = QHBoxLayout()
        title = QLabel("COPILOT")
        title.setObjectName("SectionLabel")
        self.badge = QLabel("offline")
        self.badge.setObjectName("Mono")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.badge)
        layout.addLayout(head)

        self.body = QLabel("پس از SURVEY، راهنما هندسه، صداقت اندازه‌گیری و گام بعدی را می‌نویسد.")
        self.body.setObjectName("CopilotBody")
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setAlignment(Qt.AlignRight | Qt.AlignTop)
        layout.addWidget(self.body, 1)

        self.actions = QLabel("")
        self.actions.setObjectName("Caption")
        self.actions.setWordWrap(True)
        layout.addWidget(self.actions)

    def set_brief(self, brief: dict[str, Any] | None) -> None:
        if not brief:
            self.badge.setText("idle")
            return
        provider = brief.get("provider") or "offline"
        model = brief.get("model") or "heuristic"
        self.badge.setText(f"{provider} · {model}")
        text = brief.get("report_fa") or brief.get("report_en") or ""
        build = brief.get("build_fa") or (brief.get("construct") or {}).get("build_fa") or ""
        if build:
            text = f"{text}\n\n—— دستور ساخت ——\n{build}".strip() if text else build
        self.body.setText(text)
        honesty = brief.get("honesty") or {}
        dossier = brief.get("dossier") or {}
        flag = "تخمینی" if honesty.get("height_indicative") else "قابل اندازه‌گیری"
        acts = brief.get("actions") or []
        labels = "  ·  ".join(a.get("label", a.get("id", "")) for a in acts[:3])
        warn = (brief.get("warnings") or [None])[0]
        bits = [flag]
        if dossier.get("place"):
            bits.append(str(dossier["place"])[:80])
        lat = dossier.get("lat", brief.get("lat"))
        lon = dossier.get("lon", brief.get("lon"))
        if lat is not None and lon is not None:
            bits.append(f"{float(lat):.5f}, {float(lon):.5f}")
        if brief.get("coords") and brief.get("coords") != "—":
            bits.append(str(brief["coords"]))
        if dossier.get("season"):
            bits.append(str(dossier["season"]))
        if dossier.get("when"):
            bits.append(str(dossier["when"])[:16])
        ident = dossier.get("identity") or {}
        wiki = ident.get("wikipedia") or []
        if wiki:
            bits.append(" / ".join(str(t) for t in wiki[:2]))
        extra = "  ·  ".join(bits)
        if labels:
            extra += f"\n{labels}"
        if warn:
            extra += f"\n{warn}"
        tools = [t.get("id") for t in (dossier.get("tools") or []) if t.get("ready")]
        if tools:
            extra += "\nابزار: " + ", ".join(tools[:6])
        self.actions.setText(extra)
