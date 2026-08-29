"""A slippy map, painted by hand.

No QtWebEngine, no Leaflet, no extra dependency: XYZ tiles fetched through the
deck's own caching proxy and drawn straight onto a QWidget. That keeps the map
inside the same design system as the rest of the deck - hairline graticule,
amber reticle, mono readouts - instead of embedding a browser that looks like
somebody else's product.

Interaction, in the order an operator reaches for it:

    drag             pan
    wheel            zoom about the cursor
    Ctrl+drag        draw the area of interest
    AOI mode + drag  the same, without holding a modifier
    double-click     zoom in one level
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any
from urllib import request as urlrequest

from PySide6.QtCore import QObject, QPointF, QRect, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ....models.vision.tiles import (TILE_PX, bbox_span_m, ground_resolution,
                                     lonlat_to_tile, tile_to_lonlat)
from ....templates import theme

MIN_ZOOM, MAX_ZOOM = 2, 19


def col(name: str, alpha: int = 255) -> QColor:
    r, g, b = theme.rgb_tuple(name)
    return QColor(r, g, b, alpha)


def mono(size: int, weight: int = 400) -> QFont:
    f = QFont(theme.font_first("mono"), size)
    f.setStyleHint(QFont.Monospace)
    f.setWeight(QFont.Weight(weight))
    return f


# --------------------------------------------------------------------------- #
# Tile loading
# --------------------------------------------------------------------------- #
class _TileSignals(QObject):
    ready = Signal(str, bytes)          # cache key, raw image bytes


class _TileJob(QRunnable):
    def __init__(self, url: str, key: str, signals: _TileSignals) -> None:
        super().__init__()
        self.url, self.key, self.signals = url, key, signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            request = urlrequest.Request(self.url, headers={"User-Agent": "ShadowHunter/1.0"})
            with urlrequest.urlopen(request, timeout=20) as response:
                blob = response.read()
        except Exception:
            return                                    # a missing tile just stays void
        if blob:
            self.signals.ready.emit(self.key, blob)


class TileStore:
    """Bounded pixmap cache in front of the deck's tile proxy."""

    def __init__(self, base_url: str, capacity: int = 900) -> None:
        self.base_url = base_url.rstrip("/")
        self.capacity = capacity
        self._pixmaps: OrderedDict[str, QPixmap] = OrderedDict()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self.signals = _TileSignals()
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(6)
        self.signals.ready.connect(self._store)

    @staticmethod
    def key(provider: str, z: int, x: int, y: int) -> str:
        return f"{provider}/{z}/{x}/{y}"

    def get(self, provider: str, z: int, x: int, y: int) -> QPixmap | None:
        k = self.key(provider, z, x, y)
        pixmap = self._pixmaps.get(k)
        if pixmap is not None:
            self._pixmaps.move_to_end(k)
        return pixmap

    def request(self, provider: str, z: int, x: int, y: int) -> None:
        k = self.key(provider, z, x, y)
        with self._lock:
            if k in self._pixmaps or k in self._pending:
                return
            self._pending.add(k)
        self.pool.start(_TileJob(f"{self.base_url}/api/map/tile/{k}", k, self.signals))

    def _store(self, key: str, blob: bytes) -> None:
        pixmap = QPixmap()
        if pixmap.loadFromData(blob):
            self._pixmaps[key] = pixmap
            while len(self._pixmaps) > self.capacity:
                self._pixmaps.popitem(last=False)
        with self._lock:
            self._pending.discard(key)

    @property
    def busy(self) -> int:
        with self._lock:
            return len(self._pending)


# --------------------------------------------------------------------------- #
# The map
# --------------------------------------------------------------------------- #
class MapCanvas(QWidget):
    """Pan/zoom world map with rectangle selection."""

    aoiChanged = Signal(float, float, float, float)     # west, south, east, north
    viewChanged = Signal(float, float, int)             # lon, lat, zoom
    tilesPending = Signal(int)
    suggestionChosen = Signal(float, float, float, float)

    def __init__(self, base_url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.OpenHandCursor)

        self.store = TileStore(base_url)
        self.provider = "esri"
        self.zoom = 16
        self.center_lon, self.center_lat = 55.1375, 25.0795   # Dubai Marina, a good demo
        self.aoi: tuple[float, float, float, float] | None = None
        self.aoi_mode = False
        self.suggestions: list[dict[str, Any]] = []
        self._suggest_hover: int | None = None

        self._drag_from: QPointF | None = None
        self._drag_kind = ""                # "pan" | "aoi"
        self._aoi_px: tuple[QPointF, QPointF] | None = None
        self._hover = QPointF(-1, -1)

        # Tiles land asynchronously; one modest repaint timer beats a repaint
        # per tile when sixty of them arrive at once.
        self._repaint = QTimer(self)
        self._repaint.setInterval(120)
        self._repaint.timeout.connect(self._on_tick)
        self._repaint.start()

    # ------------------------------------------------------------------ api
    def set_provider(self, provider: str) -> None:
        self.provider = provider
        self.update()

    def set_center(self, lon: float, lat: float, zoom: int | None = None) -> None:
        self.center_lon = float(lon)
        self.center_lat = max(-85.0, min(85.0, float(lat)))
        if zoom is not None:
            self.zoom = int(max(MIN_ZOOM, min(MAX_ZOOM, zoom)))
        self._emit_view()
        self.update()

    def set_aoi(self, bbox: tuple[float, float, float, float] | None) -> None:
        self.aoi = bbox
        self.update()

    def fit_bbox(self, west: float, south: float, east: float, north: float) -> None:
        """Frame a bbox and select it - used by the search box and presets."""
        self.center_lon = (west + east) / 2
        self.center_lat = (south + north) / 2
        for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
            x0, y0 = lonlat_to_tile(west, north, zoom)
            x1, y1 = lonlat_to_tile(east, south, zoom)
            if ((x1 - x0) * TILE_PX <= self.width() * 0.85
                    and (y1 - y0) * TILE_PX <= self.height() * 0.85):
                self.zoom = zoom
                break
        self.aoi = (west, south, east, north)
        self._emit_view()
        self.aoiChanged.emit(west, south, east, north)
        self.update()

    def set_aoi_mode(self, on: bool) -> None:
        self.aoi_mode = on
        self.setCursor(Qt.CrossCursor if on else Qt.OpenHandCursor)

    def clear_aoi(self) -> None:
        self.aoi = None
        self._aoi_px = None
        self.update()

    def set_suggestions(self, items: list[dict[str, Any]] | None) -> None:
        self.suggestions = list(items or [])
        self._suggest_hover = None
        self.update()

    def clear_suggestions(self) -> None:
        self.suggestions = []
        self._suggest_hover = None
        self.update()

    def _suggestion_rects(self) -> list[tuple[int, QRectF, dict[str, Any]]]:
        out: list[tuple[int, QRectF, dict[str, Any]]] = []
        for i, item in enumerate(self.suggestions):
            box = item.get("bbox")
            if not (isinstance(box, (list, tuple)) and len(box) >= 4):
                continue
            west, south, east, north = (float(v) for v in box[:4])
            top_left = self.lonlat_to_screen(west, north)
            bottom_right = self.lonlat_to_screen(east, south)
            rect = QRectF(top_left, bottom_right).normalized()
            if rect.width() >= 4 and rect.height() >= 4:
                out.append((i, rect, item))
        return out

    def _hit_suggestion(self, pos: QPointF) -> int | None:
        for i, rect, _item in self._suggestion_rects():
            if rect.contains(pos):
                return i
        return None

    @property
    def gsd_m(self) -> float:
        return ground_resolution(self.center_lat, self.zoom)

    # -------------------------------------------------------- coordinates
    def _origin(self) -> tuple[float, float]:
        """World-pixel coordinate under the widget's top-left corner."""
        cx, cy = lonlat_to_tile(self.center_lon, self.center_lat, self.zoom)
        return cx * TILE_PX - self.width() / 2, cy * TILE_PX - self.height() / 2

    def screen_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self._origin()
        return tile_to_lonlat((ox + x) / TILE_PX, (oy + y) / TILE_PX, self.zoom)

    def lonlat_to_screen(self, lon: float, lat: float) -> QPointF:
        ox, oy = self._origin()
        tx, ty = lonlat_to_tile(lon, lat, self.zoom)
        return QPointF(tx * TILE_PX - ox, ty * TILE_PX - oy)

    def _emit_view(self) -> None:
        self.viewChanged.emit(self.center_lon, self.center_lat, self.zoom)

    def _on_tick(self) -> None:
        pending = self.store.busy
        self.tilesPending.emit(pending)
        if pending:
            self.update()

    # ------------------------------------------------------------- events
    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        before = self.screen_to_lonlat(event.position().x(), event.position().y())
        self.zoom = int(max(MIN_ZOOM, min(MAX_ZOOM, self.zoom + (1 if steps > 0 else -1))))
        after = self.screen_to_lonlat(event.position().x(), event.position().y())
        # keep the point under the cursor fixed
        self.center_lon += before[0] - after[0]
        self.center_lat += before[1] - after[1]
        self.center_lat = max(-85.0, min(85.0, self.center_lat))
        self._emit_view()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        drawing = self.aoi_mode or bool(event.modifiers() & Qt.ControlModifier)
        if not drawing and not self.aoi:
            hit = self._hit_suggestion(event.position())
            if hit is not None:
                item = self.suggestions[hit]
                box = item.get("bbox")
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    west, south, east, north = (float(v) for v in box[:4])
                    self.aoi = (west, south, east, north)
                    self.clear_suggestions()
                    self.suggestionChosen.emit(west, south, east, north)
                    self.aoiChanged.emit(west, south, east, north)
                    self.update()
                    return
        self._drag_from = event.position()
        self._drag_kind = "aoi" if drawing else "pan"
        if drawing:
            self._aoi_px = (event.position(), event.position())
        else:
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        self._hover = event.position()
        if self._drag_from is None:
            self._suggest_hover = self._hit_suggestion(event.position()) if not self.aoi else None
            if self._suggest_hover is not None:
                self.setCursor(Qt.PointingHandCursor)
            elif not self.aoi_mode:
                self.setCursor(Qt.OpenHandCursor)
            self.update()
            return

        if self._drag_kind == "aoi":
            self._aoi_px = (self._drag_from, event.position())
        else:
            delta = event.position() - self._drag_from
            self._drag_from = event.position()
            ox, oy = self._origin()
            lon, lat = tile_to_lonlat((ox + self.width() / 2 - delta.x()) / TILE_PX,
                                      (oy + self.height() / 2 - delta.y()) / TILE_PX,
                                      self.zoom)
            self.center_lon, self.center_lat = lon, max(-85.0, min(85.0, lat))
            self._emit_view()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_kind == "aoi" and self._aoi_px:
            a, b = self._aoi_px
            if abs(a.x() - b.x()) > 8 and abs(a.y() - b.y()) > 8:
                lon0, lat0 = self.screen_to_lonlat(a.x(), a.y())
                lon1, lat1 = self.screen_to_lonlat(b.x(), b.y())
                bbox = (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))
                self.aoi = bbox
                self.aoiChanged.emit(*bbox)
            self._aoi_px = None
        self._drag_from = None
        self._drag_kind = ""
        self.setCursor(Qt.CrossCursor if self.aoi_mode else Qt.OpenHandCursor)
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if self.zoom < MAX_ZOOM:
            before = self.screen_to_lonlat(event.position().x(), event.position().y())
            self.zoom += 1
            after = self.screen_to_lonlat(event.position().x(), event.position().y())
            self.center_lon += before[0] - after[0]
            self.center_lat += before[1] - after[1]
            self._emit_view()
            self.update()

    def keyPressEvent(self, event) -> None:
        step = 40
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.set_center(self.center_lon, self.center_lat, self.zoom + 1)
        elif event.key() == Qt.Key_Minus:
            self.set_center(self.center_lon, self.center_lat, self.zoom - 1)
        elif event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            dx = -step if event.key() == Qt.Key_Left else step if event.key() == Qt.Key_Right else 0
            dy = -step if event.key() == Qt.Key_Up else step if event.key() == Qt.Key_Down else 0
            lon, lat = self.screen_to_lonlat(self.width() / 2 + dx, self.height() / 2 + dy)
            self.set_center(lon, lat)
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = QPointF(-1, -1)
        self.update()

    # ----------------------------------------------------------- painting
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), col("void"))

        self._paint_tiles(p)
        self._paint_graticule(p)
        self._paint_suggestions(p)
        self._paint_aoi(p)
        self._paint_scale(p)
        self._paint_hud(p)

    def _paint_suggestions(self, p: QPainter) -> None:
        if self.aoi or not self.suggestions:
            return
        for i, rect, item in self._suggestion_rects():
            hot = i == self._suggest_hover or i == 0
            tone = "signal" if hot else "hairlineHot"
            p.setPen(QPen(col(tone), 2.0 if hot else 1.2, Qt.DashLine))
            p.setBrush(col(tone, 28 if hot else 14))
            p.drawRect(rect)
            label = str(item.get("name_fa") or item.get("name") or f"#{i + 1}")
            if item.get("stated_height_m"):
                label += f"  H {float(item['stated_height_m']):.0f}"
            p.setFont(mono(9, 700))
            p.setPen(Qt.NoPen)
            p.setBrush(col(tone))
            tw = 7.2 * len(label) + 10
            p.drawRect(QRectF(rect.left(), rect.top() - 16, min(tw, max(rect.width(), 40)), 15))
            p.setPen(col("void"))
            p.drawText(QRectF(rect.left() + 4, rect.top() - 16, tw, 15),
                       Qt.AlignVCenter, label)

    def _paint_tiles(self, p: QPainter) -> None:
        ox, oy = self._origin()
        x0 = math.floor(ox / TILE_PX)
        y0 = math.floor(oy / TILE_PX)
        x1 = math.floor((ox + self.width()) / TILE_PX)
        y1 = math.floor((oy + self.height()) / TILE_PX)
        span = 2 ** self.zoom

        for ty in range(y0, y1 + 1):
            if ty < 0 or ty >= span:
                continue
            for tx in range(x0, x1 + 1):
                wrapped = tx % span
                px = tx * TILE_PX - ox
                py = ty * TILE_PX - oy
                pixmap = self.store.get(self.provider, self.zoom, wrapped, ty)
                if pixmap is None:
                    self.store.request(self.provider, self.zoom, wrapped, ty)
                    # Fall back to the parent tile so panning never flashes black.
                    parent = self.store.get(self.provider, self.zoom - 1, wrapped // 2, ty // 2)
                    if parent is not None:
                        src = QRect((wrapped % 2) * TILE_PX // 2, (ty % 2) * TILE_PX // 2,
                                    TILE_PX // 2, TILE_PX // 2)
                        p.drawPixmap(QRectF(px, py, TILE_PX, TILE_PX), parent, QRectF(src))
                    else:
                        p.fillRect(QRectF(px, py, TILE_PX, TILE_PX), col("panel"))
                        p.setPen(QPen(col("hairline"), 1))
                        p.drawRect(QRectF(px, py, TILE_PX, TILE_PX))
                    continue
                p.drawPixmap(QRectF(px, py, TILE_PX, TILE_PX), pixmap,
                             QRectF(0, 0, pixmap.width(), pixmap.height()))

    def _paint_graticule(self, p: QPainter) -> None:
        p.setPen(QPen(col("shadow", 26), 1))
        for x in range(0, self.width(), 64):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 64):
            p.drawLine(0, y, self.width(), y)

        p.setPen(QPen(col("solar", 130), 1))
        cx, cy = self.width() / 2, self.height() / 2
        p.drawLine(QPointF(cx - 9, cy), QPointF(cx - 3, cy))
        p.drawLine(QPointF(cx + 3, cy), QPointF(cx + 9, cy))
        p.drawLine(QPointF(cx, cy - 9), QPointF(cx, cy - 3))
        p.drawLine(QPointF(cx, cy + 3), QPointF(cx, cy + 9))

    def _paint_aoi(self, p: QPainter) -> None:
        rect = None
        if self._aoi_px:
            a, b = self._aoi_px
            rect = QRectF(a, b).normalized()
        elif self.aoi:
            west, south, east, north = self.aoi
            top_left = self.lonlat_to_screen(west, north)
            bottom_right = self.lonlat_to_screen(east, south)
            rect = QRectF(top_left, bottom_right).normalized()
        if rect is None or rect.width() < 1:
            return

        # Dim everything outside the selection - the AOI is the subject now.
        shade = col("void", 118)
        p.fillRect(QRectF(0, 0, self.width(), rect.top()), shade)
        p.fillRect(QRectF(0, rect.bottom(), self.width(), self.height() - rect.bottom()), shade)
        p.fillRect(QRectF(0, rect.top(), rect.left(), rect.height()), shade)
        p.fillRect(QRectF(rect.right(), rect.top(), self.width() - rect.right(), rect.height()), shade)

        p.setPen(QPen(col("solar"), 1, Qt.DashLine))
        p.drawRect(rect)
        arm = min(20.0, rect.width() / 3, rect.height() / 3)
        p.setPen(QPen(col("solar"), 2))
        for (px, py, dx, dy) in ((rect.left(), rect.top(), 1, 1), (rect.right(), rect.top(), -1, 1),
                                 (rect.left(), rect.bottom(), 1, -1),
                                 (rect.right(), rect.bottom(), -1, -1)):
            p.drawLine(QPointF(px, py), QPointF(px + dx * arm, py))
            p.drawLine(QPointF(px, py), QPointF(px, py + dy * arm))

        bbox = self.aoi
        if self._aoi_px:
            a, b = self._aoi_px
            lon0, lat0 = self.screen_to_lonlat(a.x(), a.y())
            lon1, lat1 = self.screen_to_lonlat(b.x(), b.y())
            bbox = (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))
        if bbox:
            width_m, height_m = bbox_span_m(bbox)
            text = f"{width_m:,.0f} x {height_m:,.0f} m"
            p.setFont(mono(10, 600))
            p.setPen(Qt.NoPen)
            p.setBrush(col("solar"))
            p.drawRect(QRectF(rect.left(), rect.top() - 19, 9.0 * len(text), 17))
            p.setPen(col("void"))
            p.drawText(QRectF(rect.left() + 5, rect.top() - 19, 9.0 * len(text), 17),
                       Qt.AlignVCenter, text)

    def _paint_scale(self, p: QPainter) -> None:
        metres_per_px = self.gsd_m
        for candidate in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000):
            width = candidate / metres_per_px
            if 60 <= width <= 190:
                break
        else:
            candidate, width = 100, 100 / metres_per_px

        x, y = 14, self.height() - 22
        p.setPen(QPen(col("ink"), 2))
        p.drawLine(QPointF(x, y), QPointF(x + width, y))
        p.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))
        p.drawLine(QPointF(x + width, y - 4), QPointF(x + width, y + 4))
        p.setFont(mono(9))
        p.setPen(col("inkMuted"))
        label = f"{candidate} m" if candidate < 1000 else f"{candidate // 1000} km"
        p.drawText(QPointF(x + width + 7, y + 4), label)

    def _paint_hud(self, p: QPainter) -> None:
        p.setFont(mono(10))
        if self._hover.x() >= 0:
            lon, lat = self.screen_to_lonlat(self._hover.x(), self._hover.y())
            p.setPen(QPen(col("inkFaint", 120), 1, Qt.DotLine))
            p.drawLine(QPointF(0, self._hover.y()), QPointF(self.width(), self._hover.y()))
            p.drawLine(QPointF(self._hover.x(), 0), QPointF(self._hover.x(), self.height()))
            p.setPen(col("ink"))
            p.drawText(QPointF(self._hover.x() + 10, self._hover.y() - 9),
                       f"{lat:+.5f}, {lon:+.5f}")

        badge = f"Z{self.zoom}  {self.gsd_m:.2f} m/px"
        p.setPen(Qt.NoPen)
        p.setBrush(col("void", 200))
        p.drawRect(QRectF(10, 10, 9.0 * len(badge) + 12, 20))
        p.setPen(col("shadow"))
        p.drawText(QRectF(16, 10, 9.0 * len(badge), 20), Qt.AlignVCenter, badge)

        if self.aoi_mode:
            hint = "AOI MODE - DRAG TO SELECT"
            p.setPen(Qt.NoPen)
            p.setBrush(col("solar"))
            p.drawRect(QRectF(self.width() - 9.0 * len(hint) - 22, 10, 9.0 * len(hint) + 12, 20))
            p.setPen(col("void"))
            p.setFont(mono(10, 700))
            p.drawText(QRectF(self.width() - 9.0 * len(hint) - 16, 10, 9.0 * len(hint), 20),
                       Qt.AlignVCenter, hint)


# --------------------------------------------------------------------------- #
# AOI result view
# --------------------------------------------------------------------------- #
class AOIView(QWidget):
    """The surveyed mosaic with its green structure boxes; click one to select."""

    structureSelected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._image = None
        self._buffer = None
        self._structures: list[dict[str, Any]] = []
        self._selected: int | None = None
        self._hover: int | None = None
        self._hunt_box: tuple[int, int, int, int] | None = None
        self._trail: list[tuple[int, int]] = []
        self._busy = False
        self._scan = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ api
    def set_image(self, bgr, structures: list[dict[str, Any]] | None = None) -> None:
        from .widgets import bgr_to_qimage

        if bgr is None:
            self._image = self._buffer = None
        else:
            import numpy as np

            self._buffer = np.ascontiguousarray(bgr)
            self._image = bgr_to_qimage(self._buffer)
        self._structures = structures or []
        self._selected = None
        self._hunt_box = None
        self._trail = []
        self.update()

    def set_hunt(self, box, trail=None) -> None:
        self._hunt_box = tuple(box) if box else None
        self._trail = [(t["box"][0] + t["box"][2] // 2, t["box"][1] + t["box"][3] // 2)
                       for t in (trail or [])]
        self.update()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._scan = 0.0
        self._timer.start() if busy else self._timer.stop()
        self.update()

    def select(self, index: int | None) -> None:
        self._selected = index
        self.update()

    def _tick(self) -> None:
        self._scan = (self._scan + 0.012) % 1.0
        self.update()

    # -------------------------------------------------------------- events
    def _target_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        pad = 10
        avail = self.rect().adjusted(pad, pad, -pad, -pad)
        k = min(avail.width() / self._image.width(), avail.height() / self._image.height())
        w, h = self._image.width() * k, self._image.height() * k
        return QRectF(avail.x() + (avail.width() - w) / 2,
                      avail.y() + (avail.height() - h) / 2, w, h)

    def _hit(self, pos: QPointF) -> int | None:
        if self._image is None:
            return None
        rect = self._target_rect()
        if not rect.contains(pos):
            return None
        sx = (pos.x() - rect.x()) / rect.width() * self._image.width()
        sy = (pos.y() - rect.y()) / rect.height() * self._image.height()
        best, best_area = None, None
        for i, s in enumerate(self._structures):
            x, y, w, h = s["box"]
            if x <= sx <= x + w and y <= sy <= y + h:
                area = w * h
                if best_area is None or area < best_area:   # innermost box wins
                    best, best_area = i, area
        return best

    def mouseMoveEvent(self, event) -> None:
        hover = self._hit(event.position())
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.PointingHandCursor if hover is not None else Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, event) -> None:
        index = self._hit(event.position())
        if index is not None:
            self._selected = index
            self.structureSelected.emit(index)
            self.update()

    # ----------------------------------------------------------- painting
    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QRadialGradient

        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        grad = QRadialGradient(self.width() * 0.2, -self.height() * 0.1,
                               max(self.width(), self.height()) * 1.3)
        grad.setColorAt(0.0, col("elevated"))
        grad.setColorAt(1.0, col("void"))
        p.fillRect(self.rect(), grad)

        if self._image is None:
            p.setFont(mono(11))
            p.setPen(col("inkFaint"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "NO AREA SURVEYED\n\nDRAW A RECTANGLE ON THE MAP, THEN PRESS SURVEY")
            return

        rect = self._target_rect()
        p.drawImage(rect, self._image)
        p.setPen(QPen(col("hairlineHot"), 1))
        p.drawRect(rect)

        sx = rect.width() / self._image.width()
        sy = rect.height() / self._image.height()

        for i, s in enumerate(self._structures):
            x, y, w, h = s["box"]
            r = QRectF(rect.x() + x * sx, rect.y() + y * sy, w * sx, h * sy)
            selected = i == self._selected
            hovered = i == self._hover
            tone = "solar" if selected else ("shadow" if hovered else "signal")
            alpha = 255 if (selected or hovered) else int(120 + 135 * s["score"])

            p.setPen(QPen(col(tone, alpha), 2 if selected else 1.4))
            p.drawRect(r)
            outline = s.get("outline_px")
            if isinstance(outline, list) and len(outline) >= 3:
                poly = QPolygonF([
                    QPointF(rect.x() + float(px) * sx, rect.y() + float(py) * sy)
                    for px, py in outline
                ])
                p.setPen(QPen(col(tone), 2.0 if selected else 1.6))
                p.setBrush(Qt.NoBrush)
                p.drawPolygon(poly)
            arm = min(18.0, min(r.width(), r.height()) * 0.28)
            p.setPen(QPen(col(tone), 2.2))
            for (px, py, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                                     (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
                p.drawLine(QPointF(px, py), QPointF(px + dx * arm, py))
                p.drawLine(QPointF(px, py), QPointF(px, py + dy * arm))
            w_m = float(s.get("width_m") or 0)
            d_m = float(s.get("height_m") or 0)
            stated = s.get("stated_height_m")
            tag = f"{i + 1}  {w_m:.0f}×{d_m:.0f} m"
            if stated:
                tag += f"  H {float(stated):.0f}"
            p.setFont(mono(9, 700))
            p.setPen(Qt.NoPen)
            p.setBrush(col(tone))
            tw = 8.0 * len(tag) + 10
            p.drawRect(QRectF(r.left(), r.top() - 16, min(tw, r.width() + 40), 15))
            p.setPen(col("void"))
            p.drawText(QRectF(r.left() + 4, r.top() - 16, tw, 15),
                       Qt.AlignVCenter, tag)

        if self._trail and len(self._trail) > 1:
            for i in range(1, len(self._trail)):
                t = i / max(len(self._trail) - 1, 1)
                p.setPen(QPen(col("violet", int(40 + 160 * t)), 1.2))
                a, b = self._trail[i - 1], self._trail[i]
                p.drawLine(QPointF(rect.x() + a[0] * sx, rect.y() + a[1] * sy),
                           QPointF(rect.x() + b[0] * sx, rect.y() + b[1] * sy))

        if self._hunt_box:
            x, y, w, h = self._hunt_box
            r = QRectF(rect.x() + x * sx, rect.y() + y * sy, w * sx, h * sy)
            p.setPen(QPen(col("solar"), 2))
            arm = min(16.0, r.width() * 0.3)
            for (px, py, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                                     (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
                p.drawLine(QPointF(px, py), QPointF(px + dx * arm, py))
                p.drawLine(QPointF(px, py), QPointF(px, py + dy * arm))

        if self._busy:
            from PySide6.QtGui import QLinearGradient

            y = rect.top() + rect.height() * self._scan
            sweep = QLinearGradient(rect.left(), y - 26, rect.left(), y + 4)
            sweep.setColorAt(0.0, col("shadow", 0))
            sweep.setColorAt(1.0, col("shadow", 70))
            p.fillRect(QRectF(rect.left(), y - 26, rect.width(), 30), sweep)
            p.setPen(QPen(col("shadow", 190), 1))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
