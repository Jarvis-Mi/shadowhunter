"""``ShadowHunterEnv`` - a Gymnasium environment where the agent hunts for the
one crop that contains a single, unoccluded, unclipped building shadow.

State  : 84x84x4 view (RGB + shadow mask) of the current window, plus a
         9-vector of proprioception & solar context.
Action : Discrete(10) - 8 translations, grow, shrink... plus COMMIT.
Reward : potential-based shaping over the composite proxy (see rewards.py).
Episode: ends on COMMIT, on the step budget, or on leaving the scene.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...core.config import EnvConfig, RewardConfig
from ...core.geo import SunGeometry, height_from_shadow
from ..vision.preprocess import Scene, safe_crop, synthesize_scene, to_observation
from ..vision.shadow_ops import CropMetrics, analyse_crop, shadow_mask
from .rewards import CompositeReward, RewardBreakdown

# Action ids, in the order the policy sees them.
MOVE_N, MOVE_S, MOVE_W, MOVE_E = 0, 1, 2, 3
MOVE_NW, MOVE_NE, MOVE_SW, MOVE_SE = 4, 5, 6, 7
GROW, SHRINK, COMMIT = 8, 9, 10

ACTION_NAMES = ["N", "S", "W", "E", "NW", "NE", "SW", "SE", "GROW", "SHRINK", "COMMIT"]

_DELTAS = {
    MOVE_N: (0, -1), MOVE_S: (0, 1), MOVE_W: (-1, 0), MOVE_E: (1, 0),
    MOVE_NW: (-1, -1), MOVE_NE: (1, -1), MOVE_SW: (-1, 1), MOVE_SE: (1, 1),
}


class ShadowHunterEnv(gym.Env):
    """Active zone selection over a satellite tile."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(self, scene: Scene | None = None, cfg: EnvConfig | None = None,
                 reward_cfg: RewardConfig | None = None, *,
                 randomize_scene: bool = True, seed: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.reward = CompositeReward(reward_cfg or RewardConfig())
        self.randomize_scene = randomize_scene and scene is None
        self._rng = np.random.default_rng(seed)
        self.scene: Scene = scene or synthesize_scene(size=self.cfg.tile_size, seed=seed)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict({
            "view": spaces.Box(0, 255, (4, self.cfg.obs_size, self.cfg.obs_size), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (9,), np.float32),
        })

        self.x = self.y = 0
        self.size = self.cfg.crop_init
        self.steps = 0
        self.last_metrics = CropMetrics()
        self.last_breakdown = RewardBreakdown()
        self.episode_return = 0.0
        self.committed = False

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.randomize_scene:
            self.scene = synthesize_scene(
                size=self.cfg.tile_size,
                n_buildings=int(self._rng.integers(8, 20)),
                seed=int(self._rng.integers(0, 2**31 - 1)),
            )

        W, H = self.scene.size
        self.size = self.cfg.crop_init
        self.x = int(self._rng.integers(0, max(1, W - self.size)))
        self.y = int(self._rng.integers(0, max(1, H - self.size)))
        self.steps = 0
        self.episode_return = 0.0
        self.committed = False
        self.reward.reset()

        self.last_metrics = self._analyse()
        self.last_breakdown = self.reward.step(self.last_metrics)
        return self._observation(), self._info()

    def step(self, action: int):
        action = int(action)
        self.steps += 1
        terminated = truncated = False
        reward = -self.cfg.step_cost

        if action in _DELTAS:
            dx, dy = _DELTAS[action]
            self.x += dx * self.cfg.step_px
            self.y += dy * self.cfg.step_px
        elif action == GROW:
            self.size = min(self.cfg.crop_max, self.size + self.cfg.scale_px)
        elif action == SHRINK:
            self.size = max(self.cfg.crop_min, self.size - self.cfg.scale_px)

        out_of_bounds = self._clamp()
        if out_of_bounds:
            reward -= 0.05  # nudge away from the frame edge, do not end the episode

        self.last_metrics = self._analyse()

        if action == COMMIT:
            reward += self.reward.terminal(self.last_metrics, self.cfg.commit_bonus)
            self.last_breakdown = self.reward.potential(self.last_metrics)
            self.last_breakdown.step_reward = reward
            self.committed = terminated = True
        else:
            self.last_breakdown = self.reward.step(self.last_metrics)
            reward += self.last_breakdown.step_reward

        if self.steps >= self.cfg.max_steps and not terminated:
            truncated = True

        self.episode_return += reward
        return self._observation(), float(reward), terminated, truncated, self._info(action, reward)

    def render(self):  # rgb_array
        from ..vision.shadow_ops import draw_zone, overlay_shadow

        img = self.scene.image
        mask = shadow_mask(img)
        canvas = overlay_shadow(img, mask)
        canvas = draw_zone(canvas, self.box)
        return canvas[:, :, ::-1]  # BGR -> RGB

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.size, self.size

    @property
    def sun(self) -> SunGeometry:
        return self.scene.sun

    def _clamp(self) -> bool:
        W, H = self.scene.size
        nx = int(np.clip(self.x, 0, max(0, W - self.size)))
        ny = int(np.clip(self.y, 0, max(0, H - self.size)))
        hit = (nx != self.x) or (ny != self.y)
        self.x, self.y = nx, ny
        return hit

    def _crop(self) -> np.ndarray:
        return safe_crop(self.scene.image, self.box)

    def _analyse(self) -> CropMetrics:
        crop = self._crop()
        return analyse_crop(crop, self.sun, shadow_mask(crop))

    def _observation(self) -> dict[str, np.ndarray]:
        crop = self._crop()
        mask = shadow_mask(crop)
        view = to_observation(crop, mask, self.cfg.obs_size)

        W, H = self.scene.size
        span = max(self.cfg.crop_max - self.cfg.crop_min, 1)
        az = math.radians(self.sun.shadow_bearing_deg)
        state = np.array([
            2.0 * self.x / max(W - self.size, 1) - 1.0,
            2.0 * self.y / max(H - self.size, 1) - 1.0,
            2.0 * (self.size - self.cfg.crop_min) / span - 1.0,
            2.0 * self.steps / self.cfg.max_steps - 1.0,
            math.sin(az), math.cos(az),
            self.sun.elevation_deg / 90.0,
            self.last_metrics.shadow_ratio * 2.0 - 1.0,
            self.last_metrics.occlusion * 2.0 - 1.0,
        ], np.float32)
        return {"view": view, "state": np.clip(state, -1.0, 1.0)}

    def _info(self, action: int | None = None, reward: float = 0.0) -> dict[str, Any]:
        m = self.last_metrics
        est = height_from_shadow(m.shadow_len_px, self.sun)
        return {
            "box": list(self.box),
            "step": self.steps,
            "action": ACTION_NAMES[action] if action is not None else None,
            "reward": round(float(reward), 4),
            "episode_return": round(float(self.episode_return), 4),
            "committed": self.committed,
            "metrics": m.to_dict(),
            "reward_breakdown": self.last_breakdown.to_dict(),
            "geometric_height_m": round(float(est), 2),
            "scene": self.scene.name,
        }


def make_env(scene: Scene | None = None, cfg: EnvConfig | None = None,
             reward_cfg: RewardConfig | None = None, seed: int | None = None):
    """Factory for SB3 vectorised environments."""
    def _init() -> gym.Env:
        return ShadowHunterEnv(scene=scene, cfg=cfg, reward_cfg=reward_cfg, seed=seed)
    return _init
