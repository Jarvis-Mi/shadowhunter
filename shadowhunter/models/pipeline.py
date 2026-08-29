"""The hybrid pipeline: RL zone selection -> CV height regression -> fusion.

    scene -> [RL hunter] -> clean zone -> [shadow mask] -> L_px
                                        -> [CNN regressor] -> h_cnn
                          h_geo = L*tan(theta)
                          h_fused = confidence-weighted(h_geo, h_cnn)

Nothing here imports a UI toolkit. Every view - Qt, Tk, Flet, NiceGUI, DPG -
consumes this same function through the API.
"""
from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.config import SETTINGS, EnvConfig, RewardConfig
from ..core.geo import SunGeometry, floors_from_height, geometric_uncertainty, height_from_shadow, quality_of_geometry
from ..core.logging import get_logger
from .rl.agent import HuntResult, greedy_hunt, hunt
from .vision.preprocess import Scene, safe_crop, synthesize_scene
from .vision.shadow_ops import analyse_crop, draw_zone, overlay_shadow, shadow_mask

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Model registry - loaded once, shared by every request
# --------------------------------------------------------------------------- #
class Registry:
    """Lazily holds the trained policy and the trained regressor.

    One hard constraint shapes this class: **Stable-Baselines3 policies must
    be deserialised on the main thread.** ``PPO.load`` rebuilds the policy
    network and its spaces through pickle, and on this Windows/torch build
    doing that from a worker thread corrupts the process heap - the failure
    surfaces later as an access violation during interpreter teardown, long
    after the call that caused it returned successfully.

    So ``load_policy`` refuses to run off the main thread. Instead it records
    the path and lets :meth:`warmup` pick it up from the main thread, where
    it is safe. Under uvicorn that is the lifespan hook, which already runs on
    the main thread; under ``TestClient`` (which drives the lifespan from an
    anyio portal thread) the load is deferred rather than fatal, and callers
    that genuinely need the learned policy call ``warmup()`` themselves.

    The regressor has no such problem - plain ``torch.load`` is thread-safe
    here - so it always loads inline.
    """

    def __init__(self) -> None:
        self.policy = None
        self.policy_path: Path | None = None
        self.cnn = None
        self.cnn_path: Path | None = None
        self.cnn_meta: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._deferred: tuple[Path, str] | None = None

    @staticmethod
    def _on_main_thread() -> bool:
        return threading.current_thread() is threading.main_thread()

    def load_policy(self, path: str | Path, algo: str = "PPO") -> bool:
        from .rl.agent import load_agent

        path = Path(path)
        with self._lock:
            if not self._on_main_thread():
                self._deferred = (path, algo)
                log.warning("policy load deferred to the main thread (SB3 cannot "
                            "deserialise off-thread safely): %s", path.name)
                return False
            try:
                self.policy = load_agent(path, algo)
                self.policy_path = path
                self._deferred = None
                log.info("policy loaded <- %s", path)
                return True
            except Exception as exc:
                log.error("policy load failed: %s", exc)
                return False

    def load_cnn(self, path: str | Path) -> bool:
        from .vision.height_cnn import load_model

        with self._lock:
            try:
                self.cnn, self.cnn_meta = load_model(path)
                self.cnn_path = Path(path)
                log.info("regressor loaded <- %s", path)
                return True
            except Exception as exc:
                log.error("regressor load failed: %s", exc)
                return False

    def latest(self, ckpt_dir: Path | None = None) -> tuple[Path | None, Path | None]:
        """Newest policy archive and regressor checkpoint on disk."""
        ckpt_dir = Path(ckpt_dir or SETTINGS.checkpoint_dir)
        zips = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return (zips[0] if zips else None), (pts[0] if pts else None)

    def autoload(self, ckpt_dir: Path | None = None) -> dict[str, bool]:
        policy_path, cnn_path = self.latest(ckpt_dir)
        out = {"policy": False, "cnn": False, "policy_deferred": False}
        if policy_path is not None:
            algo = "DQN" if "dqn" in policy_path.stem.lower() else "PPO"
            out["policy"] = self.load_policy(policy_path, algo)
            out["policy_deferred"] = not out["policy"] and self._deferred is not None
        if cnn_path is not None:
            out["cnn"] = self.load_cnn(cnn_path)
        return out

    def warmup(self, ckpt_dir: Path | None = None) -> dict[str, bool]:
        """Main-thread entry point: load everything, including a deferred policy.

        Every process that wants the learned policy calls this once, early,
        before it starts a server, a UI event loop or a worker pool.
        """
        if not self._on_main_thread():
            log.warning("warmup() called off the main thread - ignoring")
            return {"policy": self.policy is not None, "cnn": self.cnn is not None}

        with self._lock:
            pending = self._deferred
            self._deferred = None
        if pending is not None and self.policy is None:
            self.load_policy(*pending)
            return {"policy": self.policy is not None, "cnn": self.cnn is not None}
        return self.autoload(ckpt_dir)


REGISTRY = Registry()


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #
def build_scene(spec: dict[str, Any] | None = None) -> Scene:
    spec = spec or {}
    sun_spec = spec.get("sun") or {}
    sun = SunGeometry(
        azimuth_deg=float(sun_spec.get("azimuth_deg", 148.0)),
        elevation_deg=float(sun_spec.get("elevation_deg", 41.0)),
        gsd_m=float(sun_spec.get("gsd_m", 0.5)),
    ) if sun_spec else None

    name = spec.get("name")
    if name and not spec.get("synthesize", True):
        from .vision.preprocess import load_scene

        for base in (SETTINGS.data_dir / "synthetic", SETTINGS.data_dir / "raw"):
            for ext in (".png", ".jpg", ".tif", ".tiff"):
                p = base / f"{name}{ext}"
                if p.exists():
                    return load_scene(p, sun)
        raise FileNotFoundError(f"scene not found: {name}")

    return synthesize_scene(
        size=int(spec.get("size", 512)),
        n_buildings=int(spec.get("buildings", 14)),
        density=float(spec.get("density", 1.0)),
        sun=sun,
        seed=spec.get("seed"),
        name=name or "synthetic",
    )


# --------------------------------------------------------------------------- #
# Height estimation & fusion
# --------------------------------------------------------------------------- #
@dataclass
class Estimate:
    geometric_m: float
    cnn_m: float | None
    fused_m: float
    sigma_m: float
    floors: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometric_m": round(self.geometric_m, 2),
            "cnn_m": round(self.cnn_m, 2) if self.cnn_m is not None else None,
            "fused_m": round(self.fused_m, 2),
            "sigma_m": round(self.sigma_m, 2),
            "floors": self.floors,
            "confidence": round(self.confidence, 3),
        }


def estimate_height(crop_bgr: np.ndarray, sun: SunGeometry, metrics_dict: dict[str, Any]) -> Estimate:
    """Blend the analytic and learned estimates by their respective certainty."""
    shadow_len = float(metrics_dict.get("shadow_len_px", 0.0))
    h_geo = height_from_shadow(shadow_len, sun)
    sigma_geo = max(geometric_uncertainty(shadow_len, sun), 0.4)

    # Geometry is only trustworthy when the zone is clean and the sun cooperates.
    cleanliness = float(
        (1.0 - metrics_dict.get("occlusion", 0.0))
        * (1.0 - metrics_dict.get("truncation", 0.0))
        * (0.35 + 0.65 * metrics_dict.get("isolation", 0.0))
    )
    geometry_quality = quality_of_geometry(sun)
    confidence = float(np.clip(cleanliness * (0.55 + 0.45 * geometry_quality), 0.0, 1.0))

    h_cnn = sigma_cnn = None
    if REGISTRY.cnn is not None:
        from .vision.height_cnn import predict_height

        patch = cv2.resize(crop_bgr, (96, 96), interpolation=cv2.INTER_AREA)
        stacked = np.dstack([patch, shadow_mask(patch)])
        h_cnn, sigma_cnn = predict_height(REGISTRY.cnn, stacked, sun.elevation_deg,
                                          sun.gsd_m, shadow_len)

    if h_cnn is None:
        fused, sigma = h_geo, sigma_geo / max(confidence, 0.15)
    else:
        # Inverse-variance fusion, with the geometric arm widened when the
        # zone is dirty - exactly the situation the CNN is there to rescue.
        s_geo = sigma_geo / max(confidence, 0.15)
        s_cnn = max(float(sigma_cnn or 2.0), 0.5)
        w_geo, w_cnn = 1.0 / s_geo ** 2, 1.0 / s_cnn ** 2
        fused = (h_geo * w_geo + h_cnn * w_cnn) / (w_geo + w_cnn)
        sigma = float(np.sqrt(1.0 / (w_geo + w_cnn)))

    return Estimate(h_geo, h_cnn, float(fused), float(sigma),
                    floors_from_height(fused), confidence)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def render_overlay(scene: Scene, boxes: list[tuple[int, int, int, int]],
                   trajectory: list[dict[str, Any]] | None = None,
                   labels: list[str] | None = None) -> np.ndarray:
    """Annotated scene: cyan shadow tint, amber reticle, faint search trail."""
    mask = shadow_mask(scene.image)
    canvas = overlay_shadow(scene.image, mask)

    if trajectory:
        pts = [(b["box"][0] + b["box"][2] // 2, b["box"][1] + b["box"][3] // 2) for b in trajectory]
        for i in range(1, len(pts)):
            fade = 0.25 + 0.75 * (i / max(len(pts) - 1, 1))
            colour = (int(140 * fade), int(124 * fade), int(255 * fade))  # BGR violet->amber trail
            cv2.line(canvas, pts[i - 1], pts[i], colour, 1, cv2.LINE_AA)
        for p in pts[::4]:
            cv2.circle(canvas, p, 2, (140, 124, 255), -1, cv2.LINE_AA)

    for i, box in enumerate(boxes):
        canvas = draw_zone(canvas, box)
        if labels and i < len(labels):
            x, y, w, h = box
            cv2.putText(canvas, labels[i], (x, max(14, y - 7)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (32, 176, 255), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def analyze(scene_spec: dict[str, Any] | None = None, *, start: tuple[int, int] | None = None,
            policy: str = "auto", max_steps: int = 48,
            return_overlay: bool = True, return_trajectory: bool = True,
            scene: Scene | None = None) -> dict[str, Any]:
    """Full single-building analysis - the endpoint every UI calls."""
    t0 = time.perf_counter()
    scene = scene or build_scene(scene_spec)

    env_cfg = EnvConfig(tile_size=scene.size[0], max_steps=max_steps)
    reward_cfg = RewardConfig()

    if policy == "learned" and REGISTRY.policy is None:
        raise ValueError("no learned policy loaded")

    use_learned = policy == "learned" or (policy == "auto" and REGISTRY.policy is not None)
    if use_learned:
        result: HuntResult = hunt(scene, REGISTRY.policy, start=start,
                                  env_cfg=env_cfg, reward_cfg=reward_cfg, max_steps=max_steps)
    else:
        result = greedy_hunt(scene, start=start, env_cfg=env_cfg, reward_cfg=reward_cfg)

    crop = safe_crop(scene.image, result.box)
    metrics = analyse_crop(crop, scene.sun, shadow_mask(crop)).to_dict()
    est = estimate_height(crop, scene.sun, metrics)

    payload: dict[str, Any] = {
        "scene": scene.meta(),
        "box": list(result.box),
        "policy": result.policy,
        "policy_requested": policy,
        "committed": result.committed,
        "steps": result.steps,
        "score": round(result.score, 4),
        "metrics": metrics,
        "breakdown": result.breakdown,
        "height": est.to_dict(),
        "trajectory": result.trajectory if return_trajectory else [],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    if return_overlay:
        label = f"{est.fused_m:.1f}m  +/-{est.sigma_m:.1f}"
        payload["overlay_png"] = png_b64(render_overlay(scene, [result.box], result.trajectory, [label]))
        payload["crop_png"] = png_b64(overlay_shadow(crop, shadow_mask(crop)))
    return payload


def sweep(scene_spec: dict[str, Any] | None = None, *, policy: str = "auto",
          limit: int = 12, scene: Scene | None = None) -> dict[str, Any]:
    """Hunt every labelled building in a scene and score against ground truth."""
    t0 = time.perf_counter()
    scene = scene or build_scene(scene_spec)
    buildings = (scene.buildings or [])[:limit]

    items, boxes, labels, errors, errors_all = [], [], [], [], []
    for i, b in enumerate(buildings):
        res = analyze(policy=policy, start=b.center, scene=scene,
                      return_overlay=False, return_trajectory=False)
        est = res["height"]["fused_m"]
        err = abs(est - b.height_m)
        errors_all.append(err)
        bx, by, bw, bh = res["box"]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        bcx, bcy = b.center
        crop_size = max(bw, bh)
        wandered = float(np.hypot(cx - bcx, cy - bcy)) > 0.45 * crop_size
        matched = not wandered
        if matched:
            errors.append(err)
        items.append({
            "index": i, "box": res["box"], "truth_m": round(b.height_m, 2),
            "estimate_m": est, "error_m": round(err, 2),
            "score": res["score"], "occlusion": round(res["metrics"]["occlusion"], 3),
            "wandered": wandered, "matched": matched,
        })
        boxes.append(tuple(res["box"]))
        labels.append(f"{est:.0f}m")

    mae = float(np.mean(errors)) if errors else None
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    mae_all = float(np.mean(errors_all)) if errors_all else None
    return {
        "scene": scene.meta(),
        "items": items,
        "mae_m": round(mae, 2) if mae is not None else None,
        "mae_all_m": round(mae_all, 2) if mae_all is not None else None,
        "rmse_m": round(rmse, 2) if rmse is not None else None,
        "mean_score": round(float(np.mean([i["score"] for i in items])), 4) if items else 0.0,
        "overlay_png": png_b64(render_overlay(scene, boxes, None, labels)),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
