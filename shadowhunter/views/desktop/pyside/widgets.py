"""Custom-painted instruments for the Qt deck.

QSS can style boxes; it cannot make a compass, a reticle or a reward
sparkline. Everything with real character in this UI is painted here, using
the same tokens the stylesheet uses.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QLinearGradient,
                           QPainter, QPainterPath, QPen, QRadialGradient)
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ....templates import theme


def col(name: str, alpha: int = 255) -> QColor:
    r, g, b = theme.rgb_tuple(name)
    return QColor(r, g, b, alpha)


def mono(size: int, weight: int = QFont.Normal) -> QFont:
    f = QFont(theme.font_first("mono"), size)
    f.setStyleHint(QFont.Monospace)
    f.setWeight(QFont.Weight(weight) if isinstance(weight, int) else weight)
    return f


def display(size: int, weight: QFont.Weight = QFont.DemiBold) -> QFont:
    f = QFont(theme.font_first("display"), size)
    f.setWeight(weight)
    return f


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Zero-copy-ish conversion; the caller must keep ``bgr`` alive."""
    h, w = bgr.shape[:2]
    arr = np.ascontiguousarray(bgr)
    return QImage(arr.data, w, h, arr.strides[0], QImage.Format_BGR888).copy()


# --------------------------------------------------------------------------- #
# Scene canvas
# --------------------------------------------------------------------------- #
class SceneCanvas(QWidget):
    """The main viewport: scene, shadow tint, search trail, reticle, HUD grid.

    Clicking seeds the hunt at that point - the operator can say "start here"
    and watch the agent walk off from their pixel.
    """

    seedRequested = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(460, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self._image: QImage | None = None
        self._buffer: np.ndarray | None = None
        self._box: tuple[int, int, int, int] | None = None
        self._trail: list[tuple[int, int]] = []
        self._label = ""
        self._hover = QPointF(-1, -1)
        self._seed: tuple[int, int] | None = None
        self._scan = 0.0
        self._scanning = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(16)

    # ------------------------------------------------------------------ api
    def set_scene(self, bgr: np.ndarray | None, box=None, trail=None, label: str = "") -> None:
        self._buffer = None if bgr is None else np.ascontiguousarray(bgr)
        self._image = None if bgr is None else bgr_to_qimage(self._buffer)
        self._box = tuple(box) if box else None
        self._trail = [(t["box"][0] + t["box"][2] // 2, t["box"][1] + t["box"][3] // 2)
                       for t in (trail or [])]
        self._label = label
        self.update()

    def set_scanning(self, on: bool) -> None:
        self._scanning = on
        if on:
            self._scan = 0.0
            self._timer.start()
        else:
            self._timer.stop()
            self.update()

    def _tick(self) -> None:
        self._scan = (self._scan + 0.012) % 1.0
        self.update()

    # --------------------------------------------------------------- events
    def mouseMoveEvent(self, event) -> None:
        self._hover = event.position()
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = QPointF(-1, -1)
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._image is None or event.button() != Qt.LeftButton:
            return
        rect = self._target_rect()
        if not rect.contains(event.position()):
            return
        sx = (event.position().x() - rect.x()) / rect.width() * self._image.width()
        sy = (event.position().y() - rect.y()) / rect.height() * self._image.height()
        self._seed = (int(sx), int(sy))
        self.seedRequested.emit(int(sx), int(sy))
        self.update()

    # -------------------------------------------------------------- painting
    def _target_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        pad = 14
        avail = self.rect().adjusted(pad, pad, -pad, -pad)
        iw, ih = self._image.width(), self._image.height()
        k = min(avail.width() / iw, avail.height() / ih)
        w, h = iw * k, ih * k
        return QRectF(avail.x() + (avail.width() - w) / 2,
                      avail.y() + (avail.height() - h) / 2, w, h)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Ground: a dusk radial field, not a flat fill.
        grad = QRadialGradient(self.width() * 0.15, -self.height() * 0.1,
                               max(self.width(), self.height()) * 1.3)
        grad.setColorAt(0.0, col("elevated"))
        grad.setColorAt(0.45, col("panel"))
        grad.setColorAt(1.0, col("void"))
        p.fillRect(self.rect(), grad)
        self._paint_grid(p)

        if self._image is None:
            self._paint_empty(p)
            return

        rect = self._target_rect()
        p.drawImage(rect, self._image)
        p.setPen(QPen(col("hairlineHot"), 1))
        p.drawRect(rect)

        sx = rect.width() / self._image.width()
        sy = rect.height() / self._image.height()

        self._paint_trail(p, rect, sx, sy)
        if self._seed:
            self._paint_seed(p, rect, sx, sy)
        if self._box:
            self._paint_reticle(p, rect, sx, sy)
        if self._scanning:
            self._paint_scan(p, rect)
        if self._hover.x() >= 0:
            self._paint_crosshair(p, rect)

    def _paint_grid(self, p: QPainter) -> None:
        p.setPen(QPen(col("hairline", 90), 1))
        step = 40
        for x in range(0, self.width(), step):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            p.drawLine(0, y, self.width(), y)

    def _paint_empty(self, p: QPainter) -> None:
        p.setPen(col("inkFaint"))
        p.setFont(mono(11))
        p.drawText(self.rect(), Qt.AlignCenter, "NO SCENE LOADED\n\nSYNTHESISE A TILE OR RUN A HUNT")

    def _paint_trail(self, p: QPainter, rect: QRectF, sx: float, sy: float) -> None:
        if len(self._trail) < 2:
            return
        for i in range(1, len(self._trail)):
            t = i / max(len(self._trail) - 1, 1)
            p.setPen(QPen(col("violet", int(40 + 150 * t)), 1.2))
            a = self._trail[i - 1]
            b = self._trail[i]
            p.drawLine(QPointF(rect.x() + a[0] * sx, rect.y() + a[1] * sy),
                       QPointF(rect.x() + b[0] * sx, rect.y() + b[1] * sy))
        p.setBrush(col("violet", 190))
        p.setPen(Qt.NoPen)
        first = self._trail[0]
        p.drawEllipse(QPointF(rect.x() + first[0] * sx, rect.y() + first[1] * sy), 3, 3)

    def _paint_seed(self, p: QPainter, rect: QRectF, sx: float, sy: float) -> None:
        cx = rect.x() + self._seed[0] * sx
        cy = rect.y() + self._seed[1] * sy
        p.setPen(QPen(col("signal", 200), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), 7, 7)
        p.drawLine(QPointF(cx - 11, cy), QPointF(cx - 3, cy))
        p.drawLine(QPointF(cx + 3, cy), QPointF(cx + 11, cy))

    def _paint_reticle(self, p: QPainter, rect: QRectF, sx: float, sy: float) -> None:
        x, y, w, h = self._box
        r = QRectF(rect.x() + x * sx, rect.y() + y * sy, w * sx, h * sy)

        p.setPen(QPen(col("solar", 70), 1, Qt.DashLine))
        p.drawRect(r)

        arm = min(18.0, r.width() * 0.3)
        p.setPen(QPen(col("solar"), 2))
        for (px, py, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                                 (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
            p.drawLine(QPointF(px, py), QPointF(px + dx * arm, py))
            p.drawLine(QPointF(px, py), QPointF(px, py + dy * arm))

        if self._label:
            p.setFont(mono(11, 600))
            fm = QFontMetrics(p.font())
            tw = fm.horizontalAdvance(self._label) + 14
            tag = QRectF(r.left(), max(rect.top(), r.top() - 24), tw, 19)
            p.setPen(Qt.NoPen)
            p.setBrush(col("solar"))
            p.drawRect(tag)
            p.setPen(col("void"))
            p.drawText(tag, Qt.AlignCenter, self._label)

    def _paint_scan(self, p: QPainter, rect: QRectF) -> None:
        y = rect.top() + rect.height() * self._scan
        grad = QLinearGradient(rect.left(), y - 26, rect.left(), y + 4)
        grad.setColorAt(0.0, col("shadow", 0))
        grad.setColorAt(1.0, col("shadow", 70))
        p.fillRect(QRectF(rect.left(), y - 26, rect.width(), 30), grad)
        p.setPen(QPen(col("shadow", 190), 1))
        p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    def _paint_crosshair(self, p: QPainter, rect: QRectF) -> None:
        if not rect.contains(self._hover):
            return
        p.setPen(QPen(col("inkFaint", 110), 1, Qt.DotLine))
        p.drawLine(QPointF(rect.left(), self._hover.y()), QPointF(rect.right(), self._hover.y()))
        p.drawLine(QPointF(self._hover.x(), rect.top()), QPointF(self._hover.x(), rect.bottom()))

        if self._image is not None:
            ix = int((self._hover.x() - rect.x()) / rect.width() * self._image.width())
            iy = int((self._hover.y() - rect.y()) / rect.height() * self._image.height())
            p.setFont(mono(10))
            p.setPen(col("inkMuted"))
            p.drawText(QPointF(self._hover.x() + 9, self._hover.y() - 8), f"{ix},{iy}")


# --------------------------------------------------------------------------- #
# Sun compass
# --------------------------------------------------------------------------- #
class SunCompass(QWidget):
    """Azimuth dial + elevation arc. The single most useful widget in the app:
    it tells the operator at a glance whether the geometry is even worth
    trusting today."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(152)
        self.azimuth = 148.0
        self.elevation = 41.0
        self.quality = 0.7

    def set_sun(self, azimuth: float, elevation: float, quality: float) -> None:
        self.azimuth, self.elevation, self.quality = azimuth, elevation, quality
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        side = min(self.width(), self.height()) - 22
        cx, cy = self.width() / 2, self.height() / 2
        r = side / 2

        # dial face
        p.setPen(QPen(col("hairlineHot"), 1))
        p.setBrush(col("void"))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # ticks
        for deg in range(0, 360, 15):
            major = deg % 90 == 0
            p.setPen(QPen(col("inkFaint") if major else col("hairlineHot"), 1))
            a = math.radians(deg - 90)
            r0 = r - (9 if major else 4)
            p.drawLine(QPointF(cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                       QPointF(cx + math.cos(a) * r, cy + math.sin(a) * r))

        p.setFont(mono(9))
        p.setPen(col("inkFaint"))
        for deg, letter in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            a = math.radians(deg - 90)
            p.drawText(QRectF(cx + math.cos(a) * (r - 21) - 7, cy + math.sin(a) * (r - 21) - 7, 14, 14),
                       Qt.AlignCenter, letter)

        # elevation arc - how high the sun stands
        span = int(self.elevation / 90.0 * 90 * 16)
        p.setPen(QPen(col("shadowDeep"), 3, Qt.SolidLine, Qt.FlatCap))
        p.drawArc(QRectF(cx - r + 12, cy - r + 12, (r - 12) * 2, (r - 12) * 2), 0, span)

        # sun needle
        a = math.radians(self.azimuth - 90)
        tip = QPointF(cx + math.cos(a) * (r - 12), cy + math.sin(a) * (r - 12))
        p.setPen(QPen(col("solar"), 2))
        p.drawLine(QPointF(cx, cy), tip)
        p.setBrush(col("solar"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(tip, 4, 4)

        # shadow needle - where the shadow actually falls
        a2 = math.radians(self.azimuth + 180 - 90)
        p.setPen(QPen(col("shadow", 170), 2, Qt.DashLine))
        p.drawLine(QPointF(cx, cy), QPointF(cx + math.cos(a2) * (r - 20), cy + math.sin(a2) * (r - 20)))

        # hub + quality readout
        p.setPen(Qt.NoPen)
        p.setBrush(col("panel"))
        p.drawEllipse(QPointF(cx, cy), 22, 22)
        p.setPen(QPen(col("hairlineHot"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), 22, 22)

        q = self.quality
        tone = "signal" if q > 0.66 else ("solar" if q > 0.33 else "alert")
        p.setFont(mono(13, 700))
        p.setPen(col(tone))
        p.drawText(QRectF(cx - 22, cy - 12, 44, 14), Qt.AlignCenter, f"{self.elevation:.0f}°")
        p.setFont(mono(8))
        p.setPen(col("inkFaint"))
        p.drawText(QRectF(cx - 22, cy + 2, 44, 12), Qt.AlignCenter, f"Q {q:.2f}")


# --------------------------------------------------------------------------- #
# Metric bar
# --------------------------------------------------------------------------- #
class MetricBar(QWidget):
    """One label-free metric as a labelled bar. Inverted metrics (occlusion,
    truncation) are drawn in alert red so 'bad' reads instantly."""

    def __init__(self, label: str, tone: str = "shadow", invert: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(34)
        self.label = label
        self.tone = tone
        self.invert = invert
        self._value = 0.0
        self._shown = 0.0
        self._anim = QTimer(self)
        self._anim.timeout.connect(self._ease)
        self._anim.setInterval(16)

    def set_value(self, value: float) -> None:
        self._value = float(np.clip(value, 0.0, 1.0))
        if not self._anim.isActive():
            self._anim.start()

    def _ease(self) -> None:
        delta = self._value - self._shown
        if abs(delta) < 0.002:
            self._shown = self._value
            self._anim.stop()
        else:
            self._shown += delta * 0.18
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.setFont(mono(10))
        p.setPen(col("inkMuted"))
        p.drawText(QRect(0, 0, self.width() - 46, 15), Qt.AlignLeft | Qt.AlignVCenter, self.label.upper())

        tone = "alert" if (self.invert and self._shown > 0.35) else self.tone
        p.setFont(mono(11, 600))
        p.setPen(col(tone))
        p.drawText(QRect(self.width() - 46, 0, 46, 15), Qt.AlignRight | Qt.AlignVCenter, f"{self._shown:.3f}")

        track = QRectF(0, 21, self.width(), 3)
        p.setPen(Qt.NoPen)
        p.setBrush(col("hairline"))
        p.drawRoundedRect(track, 1.5, 1.5)

        fill = QRectF(0, 21, self.width() * self._shown, 3)
        grad = QLinearGradient(fill.topLeft(), fill.topRight())
        grad.setColorAt(0.0, col(tone, 110))
        grad.setColorAt(1.0, col(tone))
        p.setBrush(grad)
        p.drawRoundedRect(fill, 1.5, 1.5)


# --------------------------------------------------------------------------- #
# Sparkline
# --------------------------------------------------------------------------- #
class Sparkline(QWidget):
    """Rolling reward / loss trace with a filled area under the curve."""

    def __init__(self, tone: str = "solar", capacity: int = 220, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.tone = tone
        self.values: deque[float] = deque(maxlen=capacity)
        self.caption = ""

    def push(self, value: float) -> None:
        self.values.append(float(value))
        self.update()

    def clear(self) -> None:
        self.values.clear()
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), col("void"))

        p.setPen(QPen(col("hairline"), 1))
        for i in range(1, 4):
            y = self.height() * i / 4
            p.drawLine(0, int(y), self.width(), int(y))

        if len(self.values) < 2:
            p.setFont(mono(10))
            p.setPen(col("inkFaint"))
            p.drawText(self.rect(), Qt.AlignCenter, "AWAITING TELEMETRY")
            return

        vals = list(self.values)
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 1e-6)
        pad = 8
        h = self.height() - pad * 2
        pts = [QPointF(i / (len(vals) - 1) * self.width(),
                       pad + h - (v - lo) / span * h) for i, v in enumerate(vals)]

        area = QPainterPath(QPointF(pts[0].x(), self.height()))
        for pt in pts:
            area.lineTo(pt)
        area.lineTo(pts[-1].x(), self.height())
        area.closeSubpath()
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, col(self.tone, 62))
        grad.setColorAt(1.0, col(self.tone, 0))
        p.fillPath(area, grad)

        path = QPainterPath(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)
        p.strokePath(path, QPen(col(self.tone), 1.6))

        p.setBrush(col(self.tone))
        p.setPen(Qt.NoPen)
        p.drawEllipse(pts[-1], 2.6, 2.6)

        p.setFont(mono(9))
        p.setPen(col("inkFaint"))
        p.drawText(QRect(4, 2, 120, 12), Qt.AlignLeft, f"max {hi:+.2f}")
        p.drawText(QRect(4, self.height() - 14, 120, 12), Qt.AlignLeft, f"min {lo:+.2f}")
        if self.caption:
            p.setPen(col("inkMuted"))
            p.drawText(QRect(0, 2, self.width() - 6, 12), Qt.AlignRight, self.caption)


# --------------------------------------------------------------------------- #
# Readout tile
# --------------------------------------------------------------------------- #
class Readout(QFrame):
    """Big number + unit + caption. The deck's primary information surface."""

    def __init__(self, caption: str, unit: str = "", tone: str = "ink",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Elevated")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 11, 14, 12)
        lay.setSpacing(2)

        # Captions and units wrap. Without this their sizeHint is the full
        # single-line text width, every Readout in a side rail demands that
        # width, and the rail's contents overflow its fixed width and clip.
        self.caption = QLabel(caption.upper())
        self.caption.setObjectName("SectionLabel")
        self.caption.setWordWrap(True)

        self.value = QLabel("--")
        self.value.setObjectName({"solar": "ReadoutSolar", "shadow": "ReadoutShadow",
                                  "signal": "ReadoutSignal"}.get(tone, "Readout"))
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.unit = QLabel(unit)
        self.unit.setObjectName("Unit")
        self.unit.setWordWrap(True)

        # Let the frame shrink with its rail rather than pushing it wider.
        self.setMinimumWidth(96)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        lay.addWidget(self.caption)
        lay.addWidget(self.value)
        lay.addWidget(self.unit)

    def set_value(self, text: str, unit: str | None = None) -> None:
        self.value.setText(text)
        if unit is not None:
            self.unit.setText(unit)


class StatusDot(QWidget):
    """Pulsing state indicator - green nominal, amber busy, red fault."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(QSize(12, 12))
        self._tone = "inkFaint"
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_state(self, tone: str, pulsing: bool = False) -> None:
        self._tone = tone
        if pulsing:
            self._timer.start(40)
        else:
            self._timer.stop()
            self._phase = 0.0
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.06) % (2 * math.pi)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        glow = 0.5 + 0.5 * math.sin(self._phase) if self._timer.isActive() else 1.0
        p.setPen(Qt.NoPen)
        p.setBrush(col(self._tone, int(50 * glow)))
        p.drawEllipse(QPointF(6, 6), 6, 6)
        p.setBrush(col(self._tone, int(255 * (0.55 + 0.45 * glow))))
        p.drawEllipse(QPointF(6, 6), 3, 3)
