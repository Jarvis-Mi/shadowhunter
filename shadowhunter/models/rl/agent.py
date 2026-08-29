"""Stable-Baselines3 wrapper + a training-free fallback hunter.

The fallback matters more than it looks: it lets the whole application - all
five UIs, the API, the report - work on day one, before a single timestep of
training, and it doubles as the ablation baseline the paper needs ("does the
learned policy actually beat a greedy search?").
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...core.config import EnvConfig, RewardConfig, RLConfig
from ...core.logging import BUS, get_logger
from ..vision.preprocess import Scene, safe_crop
from ..vision.shadow_ops import analyse_crop, shadow_mask
from .env import ACTION_NAMES, COMMIT, ShadowHunterEnv, make_env
from .rewards import CompositeReward

log = get_logger(__name__)


@dataclass
class HuntResult:
    """One completed hunt: where the agent stopped and why."""
    box: tuple[int, int, int, int]
    score: float
    steps: int
    committed: bool
    metrics: dict[str, Any]
    breakdown: dict[str, float]
    trajectory: list[dict[str, Any]]
    policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box), "score": round(self.score, 4), "steps": self.steps,
            "committed": self.committed, "metrics": self.metrics,
            "breakdown": self.breakdown, "trajectory": self.trajectory, "policy": self.policy,
        }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_agent(cfg: RLConfig | None = None, env_cfg: EnvConfig | None = None,
                reward_cfg: RewardConfig | None = None, *,
                save_path: str | Path | None = None,
                run_id: str | None = None,
                stop_flag: dict[str, bool] | None = None,
                sink: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Train PPO or DQN on randomised synthetic cities."""
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .callbacks import StopFlagCallback, TelemetryCallback

    cfg = cfg or RLConfig()
    run_id = run_id or uuid.uuid4().hex[:8]
    BUS.publish("rl.start", run_id=run_id, algo=cfg.algo, total=cfg.total_timesteps)

    def factory():
        return Monitor(make_env(cfg=env_cfg, reward_cfg=reward_cfg, seed=cfg.seed)())

    venv = DummyVecEnv([factory])

    common = dict(policy="MultiInputPolicy", env=venv, verbose=0,
                  learning_rate=cfg.learning_rate, gamma=cfg.gamma,
                  seed=cfg.seed, device=cfg.device)
    if cfg.algo.upper() == "DQN":
        model = DQN(buffer_size=20_000, learning_starts=1_000, train_freq=4,
                    target_update_interval=500, exploration_fraction=0.35, **common)
    else:
        model = PPO(n_steps=cfg.n_steps, batch_size=64, n_epochs=6, ent_coef=0.012,
                    gae_lambda=0.95, clip_range=0.2, **common)

    callbacks = [TelemetryCallback(cfg.total_timesteps, run_id, sink=sink)]
    if stop_flag is not None:
        callbacks.append(StopFlagCallback(stop_flag))

    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, progress_bar=False)

    path = Path(save_path) if save_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        log.info("policy saved -> %s", path)

    venv.close()
    return {"run_id": run_id, "algo": cfg.algo, "timesteps": cfg.total_timesteps,
            "path": str(path) if path else None}


def load_agent(path: str | Path, algo: str = "PPO"):
    from stable_baselines3 import DQN, PPO

    cls = DQN if algo.upper() == "DQN" else PPO
    return cls.load(str(path), device="auto")


# --------------------------------------------------------------------------- #
# Inference - learned policy
# --------------------------------------------------------------------------- #
def hunt(scene: Scene, model=None, *, start: tuple[int, int] | None = None,
         env_cfg: EnvConfig | None = None, reward_cfg: RewardConfig | None = None,
         max_steps: int | None = None, deterministic: bool = True) -> HuntResult:
    """Run one episode over ``scene`` and return the zone the agent settled on."""
    env_cfg = env_cfg or EnvConfig()
    if max_steps:
        env_cfg.max_steps = max_steps

    env = ShadowHunterEnv(scene=scene, cfg=env_cfg, reward_cfg=reward_cfg, randomize_scene=False)
    obs, _ = env.reset()
    if start is not None:
        env.x, env.y = int(start[0] - env.size // 2), int(start[1] - env.size // 2)
        env._clamp()
        env.last_metrics = env._analyse()
        env.reward.reset()
        env.last_breakdown = env.reward.step(env.last_metrics)
        obs = env._observation()

    if model is None:
        env.close()
        return greedy_hunt(scene, start=start, env_cfg=env_cfg, reward_cfg=reward_cfg)

    trajectory: list[dict[str, Any]] = []
    best = (env.reward.potential(env.last_metrics).score, env.box, env.last_metrics)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        score = info["reward_breakdown"]["score"]
        trajectory.append({"step": info["step"], "box": info["box"],
                           "action": info["action"], "score": score,
                           "reward": info["reward"]})
        if score > best[0]:
            best = (score, tuple(info["box"]), env.last_metrics)
        done = terminated or truncated

    metrics_used = env.last_metrics if env.committed else best[2]
    breakdown = env.reward.potential(metrics_used)
    result = HuntResult(
        box=tuple(env.box) if env.committed else best[1],
        score=breakdown.score if env.committed else best[0],
        steps=env.steps, committed=env.committed,
        metrics=metrics_used.to_dict(),
        breakdown=breakdown.to_dict(), trajectory=trajectory, policy="learned",
    )
    env.close()
    return result


# --------------------------------------------------------------------------- #
# Inference - training-free baseline
# --------------------------------------------------------------------------- #
def greedy_hunt(scene: Scene, *, start: tuple[int, int] | None = None,
                env_cfg: EnvConfig | None = None, reward_cfg: RewardConfig | None = None,
                budget: int = 40) -> HuntResult:
    """Hill-climbing over the same proxy score. The honest baseline.

    Same action set as the RL agent, but purely myopic: it takes whichever
    single move most improves the proxy right now, and stops when nothing does.
    """
    cfg = env_cfg or EnvConfig()
    scorer = CompositeReward(reward_cfg or RewardConfig())
    W, H = scene.size

    size = cfg.crop_init
    if start is not None:
        x, y = int(start[0] - size // 2), int(start[1] - size // 2)
    else:
        x, y = (W - size) // 2, (H - size) // 2

    def evaluate(bx: int, by: int, bs: int):
        bx = int(np.clip(bx, 0, max(0, W - bs)))
        by = int(np.clip(by, 0, max(0, H - bs)))
        crop = safe_crop(scene.image, (bx, by, bs, bs))
        m = analyse_crop(crop, scene.sun, shadow_mask(crop))
        return scorer.potential(m).score, (bx, by, bs, bs), m

    score, box, metrics = evaluate(x, y, size)
    trajectory = [{"step": 0, "box": list(box), "action": "INIT", "score": round(score, 4), "reward": 0.0}]
    committed = False

    moves = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (-1, -1, 0), (1, -1, 0),
             (-1, 1, 0), (1, 1, 0), (0, 0, 1), (0, 0, -1)]
    for step in range(1, budget + 1):
        best = (score, box, metrics, "COMMIT")
        for i, (mx, my, ms) in enumerate(moves):
            s, b, m = evaluate(box[0] + mx * cfg.step_px,
                               box[1] + my * cfg.step_px,
                               int(np.clip(box[2] + ms * cfg.scale_px, cfg.crop_min, cfg.crop_max)))
            # ``evaluate`` re-clamps into the scene, so ``b`` is always valid
            if s > best[0] + 1e-4:
                best = (s, b, m, ACTION_NAMES[i])
        if best[3] == "COMMIT":
            trajectory.append({"step": step, "box": list(box), "action": "COMMIT",
                               "score": round(score, 4), "reward": 0.0})
            committed = True
            break
        gain = best[0] - score
        score, box, metrics = best[0], best[1], best[2]
        trajectory.append({"step": step, "box": list(box), "action": best[3],
                           "score": round(score, 4), "reward": round(gain, 4)})

    return HuntResult(box=box, score=score, steps=len(trajectory) - 1, committed=committed,
                      metrics=metrics.to_dict(), breakdown=scorer.potential(metrics).to_dict(),
                      trajectory=trajectory, policy="greedy")
