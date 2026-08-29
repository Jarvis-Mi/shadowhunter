"""Live field reconstruction inside the Qt deck — mosaic, roof crop, sun shadow.

If a compact shadow volume cannot be trusted (whole-AOI seed, no shadow),
the view falls back to mosaic ground plus luminance relief.
Colour spectra belong on the RGB/DEPTH tab, not in this widget.
No browser, no file://, no CDN.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import (QColor, QImage, QMouseEvent, QPainter, QPainterPath,
                           QPen, QPolygonF, QTransform, QVector3D, QWheelEvent, QMatrix4x4)
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .widgets import bgr_to_qimage, col, mono


def _textured_quad(p: QPainter, dst: list[QPointF], image: QImage | None) -> bool:
    if image is None or image.isNull() or len(dst) != 4:
        return False
    src = QPolygonF([
        QPointF(0, 0),
        QPointF(image.width(), 0),
        QPointF(image.width(), image.height()),
        QPointF(0, image.height()),
    ])
    xform = QTransform()
    if not QTransform.quadToQuad(src, QPolygonF(dst), xform):
        return False
    clip = QPainterPath()
    clip.addPolygon(QPolygonF(dst))
    p.save()
    p.setClipPath(clip)
    p.setTransform(xform, True)
    p.drawImage(0, 0, image)
    p.restore()
    return True


class Field3DView(QWidget):
    """Orbitable reconstruction: textured ground, coloured extrusions, cast shadows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._mosaic: QImage | None = None
        self._mosaic_np: np.ndarray | None = None
        self._scene: dict[str, Any] = {}
        self._yaw = 0.55
        self._pitch = 0.62
        self._dist = 140.0
        self._drag: QPoint | None = None
        self._note = "SURVEY then RUN FIELD — reconstruction stays inside this deck"

    def viable(self) -> bool:
        buildings = (self._scene or {}).get("buildings") or []
        if any(not b.get("cover") for b in buildings):
            return True
        if "viable" in (self._scene or {}):
            return bool(self._scene.get("viable"))
        return False

    def load(self, mosaic_bgr: np.ndarray | None, scene: dict[str, Any]) -> None:
        self._scene = scene or {}
        self._mosaic_np = None if mosaic_bgr is None else np.ascontiguousarray(mosaic_bgr)
        self._mosaic = None if mosaic_bgr is None else bgr_to_qimage(self._mosaic_np)
        ground = (scene or {}).get("ground") or {}
        span = max(float(ground.get("width_m") or 80), float(ground.get("height_m") or 80), 20.0)
        buildings = (scene or {}).get("buildings") or []
        max_h = max(
            (float(b.get("height_m") or 0) for b in buildings if not b.get("cover")),
            default=12.0,
        )
        self._dist = max(span * 1.15, max_h * 2.6, 36.0)
        self._yaw, self._pitch = 0.82, 0.78
        if self.viable():
            n = len([b for b in buildings if not b.get("cover")])
            vol = next((b for b in buildings if not b.get("cover")), None)
            extra = ""
            if vol:
                extra = (f"  ·  {float(vol.get('height_m') or 0):.0f} m "
                         f"{vol.get('height_source') or ''}"
                         f"  ·  {vol.get('massing_kind') or 'prism'}")
            self._note = f"{n} volumes{extra}  ·  drag to orbit  ·  agent massing"
        else:
            self._note = "volume not trusted — mosaic ground + luminance relief"
        self.update()

    def load_survey(self, survey: dict[str, Any], measures: list[dict[str, Any]] | None,
                    mosaic_bgr: np.ndarray | None, *,
                    inspect: dict[str, Any] | None = None,
                    construct: dict[str, Any] | None = None) -> None:
        from ....models.reconstruct import compose_field

        if mosaic_bgr is None:
            self._note = "no mosaic — survey an area first"
            self._scene = {}
            self.update()
            return
        sun = survey.get("sun") or {}
        gsd = float((survey.get("scene") or {}).get("gsd_m") or sun.get("gsd_m") or 0.5)
        meta = dict(survey.get("scene") or {})
        if survey.get("landmark"):
            meta["landmark"] = survey["landmark"]
        if inspect and inspect.get("vlm"):
            meta["vlm"] = inspect["vlm"]
        if construct:
            meta["construct"] = construct
        scene = compose_field(
            mosaic_bgr, survey.get("structures") or [], measures, sun, gsd, meta,
        )
        self.load(mosaic_bgr, scene)

    def set_spectra(self, inspect: dict[str, Any] | None) -> None:
        """No-op: spectra live on the RGB/DEPTH tab, not in the 3D field."""
        return

    def _mvp(self) -> QMatrix4x4:
        ground = self._scene.get("ground") or {}
        w = max(float(ground.get("width_m") or 80.0), 8.0)
        d = max(float(ground.get("height_m") or 80.0), 8.0)
        cx, cy = w * 0.5, -d * 0.5
        pitch = min(max(self._pitch, 0.18), 1.35)
        dist = max(self._dist, 12.0)
        eye = QVector3D(
            cx + dist * math.cos(pitch) * math.sin(self._yaw),
            cy - dist * math.cos(pitch) * math.cos(self._yaw),
            dist * math.sin(pitch),
        )
        view = QMatrix4x4()
        view.lookAt(eye, QVector3D(cx, cy, 0.0), QVector3D(0.0, 0.0, 1.0))
        proj = QMatrix4x4()
        aspect = max(self.width() / max(self.height(), 1), 0.2)
        proj.perspective(42.0, aspect, 0.8, dist * 18.0)
        return proj * view

    def _project(self, x: float, y: float, z: float, mvp: QMatrix4x4) -> tuple[QPointF, float] | None:
        p = mvp.map(QVector3D(float(x), float(y), float(z)))
        sx = (p.x() * 0.5 + 0.5) * self.width()
        sy = (-p.y() * 0.5 + 0.5) * self.height()
        return QPointF(sx, sy), float(p.z())

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), col("void"))
        if not self._scene:
            p.setPen(col("inkFaint"))
            p.setFont(mono(10))
            p.drawText(self.rect(), Qt.AlignCenter, self._note)
            return

        mvp = self._mvp()
        ground = self._scene.get("ground") or {}
        gw = float(ground.get("width_m") or 80.0)
        gd = float(ground.get("height_m") or 80.0)

        self._paint_ground(p, mvp, gw, gd)
        if self.viable():
            self._paint_shadows(p, mvp)
            self._paint_buildings(p, mvp)
        else:
            self._paint_relief(p, mvp, gw, gd)

        p.setPen(col("inkFaint"))
        p.setFont(mono(9))
        p.drawText(self.rect().adjusted(10, 8, -10, -8), Qt.AlignLeft | Qt.AlignTop,
                   "FIELD 3D  ·  " + self._note)
        sun = self._scene.get("sun") or {}
        p.drawText(self.rect().adjusted(10, 8, -10, -10), Qt.AlignLeft | Qt.AlignBottom,
                   f"sun {float(sun.get('elevation_deg') or 0):.0f}°  "
                   f"az {float(sun.get('azimuth_deg') or 0):.0f}°  ·  "
                   f"gsd {self._scene.get('gsd_m', '—')} m")

    def _paint_ground(self, p: QPainter, mvp: QMatrix4x4, gw: float, gd: float) -> None:
        corners = [(0.0, 0.0, 0.0), (gw, 0.0, 0.0), (gw, -gd, 0.0), (0.0, -gd, 0.0)]
        dst: list[QPointF] = []
        for c in corners:
            pr = self._project(*c, mvp)
            if pr is None:
                return
            dst.append(pr[0])
        poly = QPolygonF(dst)
        p.setPen(QPen(col("hairline"), 1))
        p.setBrush(col("panel"))
        p.drawPolygon(poly)
        _textured_quad(p, dst, self._mosaic)

    def _paint_relief(self, p: QPainter, mvp: QMatrix4x4, gw: float, gd: float) -> None:
        """Lift the mosaic by luminance so the monument reads without a fake box."""
        img = self._mosaic_np
        if img is None or img.size == 0:
            return
        h, w = img.shape[:2]
        gx, gy = 20, 20
        luma = img.mean(axis=2).astype(np.float32) / 255.0
        zscale = min(gw, gd) * 0.22
        faces: list[tuple[float, list[QPointF], QColor]] = []
        for iy in range(gy):
            for ix in range(gx):
                u0, u1 = ix / gx, (ix + 1) / gx
                v0, v1 = iy / gy, (iy + 1) / gy
                pts_uv = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
                quad: list[QPointF] = []
                zmean = 0.0
                skip = False
                for u, v in pts_uv:
                    px = min(w - 1, int(u * (w - 1)))
                    py = min(h - 1, int(v * (h - 1)))
                    z = float(luma[py, px]) * zscale
                    pr = self._project(u * gw, -v * gd, z, mvp)
                    if pr is None:
                        skip = True
                        break
                    quad.append(pr[0])
                    zmean += pr[1]
                if skip or len(quad) < 4:
                    continue
                px = min(w - 1, int(((u0 + u1) / 2) * (w - 1)))
                py = min(h - 1, int(((v0 + v1) / 2) * (h - 1)))
                b, g, r = (int(img[py, px, k]) for k in range(3))
                faces.append((zmean / 4.0, quad, QColor(r, g, b)))
        faces.sort(key=lambda item: item[0], reverse=True)
        p.setPen(Qt.NoPen)
        for _depth, pts, colour in faces:
            p.setBrush(colour)
            p.drawPolygon(QPolygonF(pts))

    def _paint_shadows(self, p: QPainter, mvp: QMatrix4x4) -> None:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(6, 8, 11, 120))
        for b in self._scene.get("buildings") or []:
            if b.get("cover"):
                continue
            sdx, sdy = float(b.get("shadow_dx_m") or 0), float(b.get("shadow_dy_m") or 0)
            if math.hypot(sdx, sdy) <= 0.8:
                continue
            outline = b.get("outline_m")
            if isinstance(outline, list) and len(outline) >= 3:
                hull: list[QPointF] = []
                for x, y in outline:
                    pr = self._project(float(x) + sdx, float(y) + sdy, 0.03, mvp)
                    if pr:
                        hull.append(pr[0])
                if len(hull) >= 3:
                    p.drawPolygon(QPolygonF(hull))
                continue
            hw = float(b.get("width_m") or 1) * 0.5
            hd = float(b.get("depth_m") or 1) * 0.5
            cx, cy = float(b.get("x_m") or 0), float(b.get("y_m") or 0)
            extra = [
                (cx - hw, cy - hd, 0.03),
                (cx + hw, cy - hd, 0.03),
                (cx + hw + sdx, cy - hd + sdy, 0.03),
                (cx - hw + sdx, cy - hd + sdy, 0.03),
            ]
            hull = []
            for x, y, z in extra:
                pr = self._project(x, y, z, mvp)
                if pr:
                    hull.append(pr[0])
            if len(hull) >= 3:
                p.drawPolygon(QPolygonF(hull))

    def _roof_crop(self, building: dict[str, Any]) -> QImage | None:
        if self._mosaic is None:
            return None
        box = building.get("box") or []
        if len(box) < 4:
            return None
        x, y, bw, bh = (int(v) for v in box[:4])
        if bw < 2 or bh < 2:
            return None
        rect = QRect(x, y, bw, bh).intersected(
            QRect(0, 0, self._mosaic.width(), self._mosaic.height()))
        if rect.width() < 2 or rect.height() < 2:
            return None
        return self._mosaic.copy(rect)

    def _append_box_faces(
        self,
        faces: list[tuple[float, list[QPointF], QColor, QImage | None, QPainterPath | None]],
        *,
        cx: float, cy: float, zc: float, hw: float, hd: float, hh: float,
        yaw: float, roof: QColor, side: QColor, dark: QColor,
        crop: QImage | None = None,
        mvp: QMatrix4x4,
    ) -> None:
        c, s = math.cos(yaw), math.sin(yaw)

        def local(lx: float, ly: float, z: float) -> tuple[float, float, float]:
            return cx + c * lx - s * ly, cy + s * lx + c * ly, z

        z0, z1 = zc - hh * 0.5, zc + hh * 0.5
        corners = {
            "flb": local(-hw, -hd, z0),
            "frb": local(hw, -hd, z0),
            "brb": local(hw, hd, z0),
            "blb": local(-hw, hd, z0),
            "flt": local(-hw, -hd, z1),
            "frt": local(hw, -hd, z1),
            "brt": local(hw, hd, z1),
            "blt": local(-hw, hd, z1),
        }
        projected: dict[str, tuple[QPointF, float]] = {}
        for key, xyz in corners.items():
            pr = self._project(*xyz, mvp)
            if pr is None:
                return
            projected[key] = pr
        quads = [
            (("flt", "frt", "brt", "blt"), roof, crop),
            (("flb", "frb", "frt", "flt"), side, None),
            (("frb", "brb", "brt", "frt"), dark, None),
            (("brb", "blb", "blt", "brt"), side, None),
            (("blb", "flb", "flt", "blt"), dark, None),
        ]
        for keys, colour, tex in quads:
            pts = [projected[k][0] for k in keys]
            depth = sum(projected[k][1] for k in keys) / 4.0
            faces.append((depth, pts, colour, tex, None))

    def _paint_buildings(self, p: QPainter, mvp: QMatrix4x4) -> None:
        faces: list[tuple[float, list[QPointF], QColor, QImage | None, QPainterPath | None]] = []
        for b in self._scene.get("buildings") or []:
            if b.get("cover"):
                continue
            hh = max(float(b.get("height_m") or 8), 0.8)
            rgb = b.get("rgb") or [255, 176, 32]
            roof = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            side = QColor(max(0, roof.red() - 40), max(0, roof.green() - 35), max(0, roof.blue() - 25))
            dark = QColor(max(0, roof.red() - 70), max(0, roof.green() - 60), max(0, roof.blue() - 45))
            crop = self._roof_crop(b)
            parts = b.get("parts") or []
            if isinstance(parts, list) and len(parts) >= 2:
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    pr = part.get("rgb") or rgb
                    proof = QColor(int(pr[0]), int(pr[1]), int(pr[2]))
                    pside = QColor(max(0, proof.red() - 40), max(0, proof.green() - 35),
                                   max(0, proof.blue() - 25))
                    pdark = QColor(max(0, proof.red() - 70), max(0, proof.green() - 60),
                                   max(0, proof.blue() - 45))
                    self._append_box_faces(
                        faces,
                        cx=float(part.get("x_m") or 0),
                        cy=float(part.get("y_m") or 0),
                        zc=float(part.get("z_m") or float(part.get("height_m") or 1) * 0.5),
                        hw=max(float(part.get("width_m") or 1), 0.5) * 0.5,
                        hd=max(float(part.get("depth_m") or 1), 0.5) * 0.5,
                        hh=max(float(part.get("height_m") or 1), 0.5),
                        yaw=float(part.get("yaw") or 0),
                        roof=proof, side=pside, dark=pdark,
                        crop=crop if part.get("role") in {"vault", "mass"} else None,
                        mvp=mvp,
                    )
                continue
            outline = b.get("outline_m")
            if isinstance(outline, list) and len(outline) >= 3:
                rings = [outline] + [h for h in (b.get("holes_m") or []) if isinstance(h, list) and len(h) >= 3]
                for ring in rings:
                    n = len(ring)
                    for i in range(n):
                        x0, y0 = float(ring[i][0]), float(ring[i][1])
                        x1, y1 = float(ring[(i + 1) % n][0]), float(ring[(i + 1) % n][1])
                        corners = ((x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, hh), (x0, y0, hh))
                        projected: list[tuple[QPointF, float]] = []
                        skip = False
                        for xyz in corners:
                            pr = self._project(*xyz, mvp)
                            if pr is None:
                                skip = True
                                break
                            projected.append(pr)
                        if skip:
                            continue
                        pts = [pr[0] for pr in projected]
                        depth = sum(pr[1] for pr in projected) / 4.0
                        colour = side if (i % 2 == 0) else dark
                        faces.append((depth, pts, colour, None, None))
                roof_path = QPainterPath()
                roof_path.setFillRule(Qt.OddEvenFill)
                roof_pts: list[QPointF] = []
                roof_z = 0.0
                skip_roof = False
                for i, (x, y) in enumerate(outline):
                    pr = self._project(float(x), float(y), hh, mvp)
                    if pr is None:
                        skip_roof = True
                        break
                    roof_pts.append(pr[0])
                    roof_z += pr[1]
                    if i == 0:
                        roof_path.moveTo(pr[0])
                    else:
                        roof_path.lineTo(pr[0])
                if not skip_roof and len(roof_pts) >= 3:
                    roof_path.closeSubpath()
                    for hole in b.get("holes_m") or []:
                        if not isinstance(hole, list) or len(hole) < 3:
                            continue
                        for j, (x, y) in enumerate(hole):
                            pr = self._project(float(x), float(y), hh, mvp)
                            if pr is None:
                                break
                            if j == 0:
                                roof_path.moveTo(pr[0])
                            else:
                                roof_path.lineTo(pr[0])
                        else:
                            roof_path.closeSubpath()
                    tex_pts = roof_pts[:4]
                    box = b.get("box") or []
                    gsd = float(self._scene.get("gsd_m") or 0.5)
                    if len(box) >= 4 and gsd > 0:
                        bx, by, bw, bh = (float(v) for v in box[:4])
                        quad3 = [
                            (bx * gsd, -by * gsd, hh),
                            ((bx + bw) * gsd, -by * gsd, hh),
                            ((bx + bw) * gsd, -(by + bh) * gsd, hh),
                            (bx * gsd, -(by + bh) * gsd, hh),
                        ]
                        proj_quad: list[QPointF] = []
                        okq = True
                        for xyz in quad3:
                            pr = self._project(*xyz, mvp)
                            if pr is None:
                                okq = False
                                break
                            proj_quad.append(pr[0])
                        if okq:
                            tex_pts = proj_quad
                    faces.append((roof_z / len(roof_pts), tex_pts, roof, crop, roof_path))
                continue
            hw = float(b.get("width_m") or 1) * 0.5
            hd = float(b.get("depth_m") or 1) * 0.5
            cx, cy = float(b.get("x_m") or 0), float(b.get("y_m") or 0)
            corners = {
                "flb": (cx - hw, cy - hd, 0.0),
                "frb": (cx + hw, cy - hd, 0.0),
                "brb": (cx + hw, cy + hd, 0.0),
                "blb": (cx - hw, cy + hd, 0.0),
                "flt": (cx - hw, cy - hd, hh),
                "frt": (cx + hw, cy - hd, hh),
                "brt": (cx + hw, cy + hd, hh),
                "blt": (cx - hw, cy + hd, hh),
            }
            projected: dict[str, tuple[QPointF, float]] = {}
            skip = False
            for key, xyz in corners.items():
                pr = self._project(*xyz, mvp)
                if pr is None:
                    skip = True
                    break
                projected[key] = pr
            if skip:
                continue
            quads = [
                (("flt", "frt", "brt", "blt"), roof, crop),
                (("flb", "frb", "frt", "flt"), side, None),
                (("frb", "brb", "brt", "frt"), dark, None),
                (("brb", "blb", "blt", "brt"), side, None),
                (("blb", "flb", "flt", "blt"), dark, None),
            ]
            for keys, colour, tex in quads:
                pts = [projected[k][0] for k in keys]
                depth = sum(projected[k][1] for k in keys) / 4.0
                faces.append((depth, pts, colour, tex, None))
        faces.sort(key=lambda item: item[0], reverse=True)
        for _depth, pts, colour, tex, clip in faces:
            if clip is not None:
                p.save()
                p.setClipPath(clip)
                if tex is not None and _textured_quad(p, pts[:4] if len(pts) >= 4 else pts, tex):
                    p.restore()
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(63, 211, 228, 110), 1.0))
                    p.drawPath(clip)
                    continue
                p.setBrush(colour)
                p.setPen(Qt.NoPen)
                p.drawPath(clip)
                p.restore()
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(63, 211, 228, 110), 1.0))
                p.drawPath(clip)
                continue
            if tex is not None and _textured_quad(p, pts, tex):
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(63, 211, 228, 90), 1.0))
                p.drawPolygon(QPolygonF(pts))
                continue
            path = QPainterPath()
            path.addPolygon(QPolygonF(pts))
            p.setBrush(colour)
            p.setPen(QPen(QColor(63, 211, 228, 90), 1.0))
            p.drawPath(path)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None:
            return
        pos = event.position().toPoint()
        dx, dy = pos.x() - self._drag.x(), pos.y() - self._drag.y()
        self._yaw += dx * 0.008
        self._pitch = min(max(self._pitch + dy * 0.006, 0.18), 1.35)
        self._drag = pos
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 0.92 if delta > 0 else 1.08
        self._dist = min(max(self._dist * factor, 8.0), 4000.0)
        self.update()


class Field3DDialog(QDialog):
    """Detached reconstruction window — same live scene, larger orbit."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Shadow Hunter · field reconstruction")
        self.resize(1100, 720)
        self.view = Field3DView()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 8)
        label = QLabel("mosaic on ground  ·  roof crop  ·  luminance relief when volume is not trusted")
        label.setObjectName("Caption")
        bar.addWidget(label)
        bar.addStretch(1)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)

    def load_survey(self, survey: dict[str, Any], measures: list[dict[str, Any]] | None,
                    mosaic_bgr: np.ndarray | None, *,
                    inspect: dict[str, Any] | None = None,
                    construct: dict[str, Any] | None = None) -> None:
        self.view.load_survey(survey, measures, mosaic_bgr,
                              inspect=inspect, construct=construct)

    def set_spectra(self, inspect: dict[str, Any] | None) -> None:
        self.view.set_spectra(inspect)
