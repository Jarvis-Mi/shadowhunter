"""Stable-Baselines3 callbacks that stream training telemetry onto the bus."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ...core.logging import BUS, get_logger

log = get_logger(__name__)


class TelemetryCallback(BaseCallback):
    """Publish ``rl.progress`` events so every UI can watch the same run."""

    def __init__(self, total_timesteps: int, run_id: str, every: int = 256,
                 sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        super().__init__(verbose=0)
        self.total = max(1, total_timesteps)
        self.run_id = run_id
        self.every = every
        self.sink = sink
        self.started = time.time()
        self._returns: list[float] = []
        self._last_emit = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []) or []:
            ep = info.get("episode")
            if ep is not None:
                self._returns.append(float(ep["r"]))

        if self.num_timesteps - self._last_emit < self.every:
            return True
        self._last_emit = self.num_timesteps

        window = self._returns[-25:]
        elapsed = max(time.time() - self.started, 1e-6)
        payload = {
            "run_id": self.run_id,
            "timesteps": int(self.num_timesteps),
            "total": self.total,
            "progress": round(self.num_timesteps / self.total, 4),
            "ep_reward_mean": round(float(np.mean(window)), 4) if window else None,
            "ep_reward_max": round(float(np.max(window)), 4) if window else None,
            "episodes": len(self._returns),
            "fps": round(self.num_timesteps / elapsed, 1),
            "elapsed_s": round(elapsed, 1),
        }
        BUS.publish("rl.progress", **payload)
        if self.sink:
            self.sink(payload)
        return True

    def _on_training_end(self) -> None:
        BUS.publish("rl.done", run_id=self.run_id, episodes=len(self._returns),
                    ep_reward_mean=float(np.mean(self._returns[-25:])) if self._returns else None)


class StopFlagCallback(BaseCallback):
    """Cooperative cancellation - the UI's Abort button flips ``flag['stop']``."""

    def __init__(self, flag: dict[str, bool]) -> None:
        super().__init__(verbose=0)
        self.flag = flag

    def _on_step(self) -> bool:
        if self.flag.get("stop"):
            log.warning("training aborted by operator")
            return False
        return True
