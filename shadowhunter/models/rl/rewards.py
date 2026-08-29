"""Proxy reward functions - the scientific core of Shadow Hunter.

Constraint: during exploration the agent must be scored **without any
ground-truth height**. Otherwise the method collapses into supervised
learning with extra steps, and it cannot generalise to unlabelled cities.

Three independent, purely image-derived proxies are implemented, each with a
different failure mode, plus a composite that combines them and subtracts the
two contamination penalties (occlusion, truncation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...core.config import RewardConfig
from ..vision.shadow_ops import CropMetrics


def _band(value: float, low: float, high: float, soft: float = 0.06) -> float:
    """1.0 inside [low, high], smoothly decaying outside. Keeps the agent from
    degenerate crops that are all-shadow or all-roof."""
    if low <= value <= high:
        return 1.0
    d = (low - value) if value < low else (value - high)
    return float(np.exp(-(d / soft) ** 2))


# --------------------------------------------------------------------------- #
# R1 - Contrast / isolation
# --------------------------------------------------------------------------- #
def r1_contrast_isolation(m: CropMetrics) -> float:
    """Reward a crop where one dark region separates cleanly from sunlit ground.

    score = contrast * isolation * coverage_band

    Strength: extremely cheap, no assumptions about shape.
    Weakness:  a large water body or an asphalt lot scores well too - which is
               why R3 (solar geometry) has to vote as well.
    """
    coverage = _band(m.shadow_ratio, 0.10, 0.45, soft=0.10)
    return float(np.clip(m.contrast * m.isolation * coverage, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# R2 - Structural purity
# --------------------------------------------------------------------------- #
def r2_structural_purity(m: CropMetrics) -> float:
    """Reward crops whose edge energy is concentrated on the shadow boundary.

    score = edge_coherence * (1 - entropy)^0.5

    A crop holding one building and its shadow has *low* scene entropy and
    *high* boundary coherence. A crop straddling three rooftops, a road and
    two shadows has the opposite signature.
    """
    order = float(np.clip(1.0 - m.entropy, 0.0, 1.0)) ** 0.5
    return float(np.clip(m.edge_coherence * order, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# R3 - Solar-azimuth coherence
# --------------------------------------------------------------------------- #
def r3_azimuth_coherence(m: CropMetrics) -> float:
    """Reward a blob that is elongated *along the direction the sun casts*.

    score = axis_alignment * elongation * (1 - truncation)

    This is the only proxy that uses metadata the imagery ships with for free
    (sun azimuth), and it is what separates a genuine cast shadow from a dark
    parking lot: parking lots have no preferred orientation.
    """
    return float(np.clip(m.axis_alignment * (0.35 + 0.65 * m.elongation) * (1.0 - m.truncation), 0.0, 1.0))


REWARD_FUNCTIONS: dict[str, Callable[[CropMetrics], float]] = {
    "contrast_isolation": r1_contrast_isolation,
    "structural_purity": r2_structural_purity,
    "azimuth_coherence": r3_azimuth_coherence,
}


# --------------------------------------------------------------------------- #
# Composite + potential-based shaping
# --------------------------------------------------------------------------- #
@dataclass
class RewardBreakdown:
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    occlusion_penalty: float = 0.0
    truncation_penalty: float = 0.0
    score: float = 0.0          # potential of the current state, 0..1-ish
    step_reward: float = 0.0    # what the agent actually receives this step

    def to_dict(self) -> dict[str, float]:
        return {
            "r1_contrast": round(self.r1, 4),
            "r2_structure": round(self.r2, 4),
            "r3_azimuth": round(self.r3, 4),
            "occlusion_penalty": round(self.occlusion_penalty, 4),
            "truncation_penalty": round(self.truncation_penalty, 4),
            "score": round(self.score, 4),
            "step_reward": round(self.step_reward, 4),
        }


class CompositeReward:
    """Weighted blend of R1/R2/R3 with occlusion + truncation penalties.

    With ``shaping=True`` the agent receives the *improvement* in potential
    (Ng et al. potential-based shaping), which preserves the optimal policy
    while making the credit assignment dense enough for PPO to learn in a few
    thousand steps on a laptop.
    """

    def __init__(self, cfg: RewardConfig | None = None) -> None:
        self.cfg = cfg or RewardConfig()
        self._prev: float | None = None

    def reset(self) -> None:
        self._prev = None

    def potential(self, m: CropMetrics) -> RewardBreakdown:
        b = RewardBreakdown()
        b.r1 = r1_contrast_isolation(m)
        b.r2 = r2_structural_purity(m)
        b.r3 = r3_azimuth_coherence(m)
        b.occlusion_penalty = self.cfg.w_occlusion * m.occlusion
        b.truncation_penalty = self.cfg.w_truncation * m.truncation

        c = self.cfg
        pos = c.w_contrast * b.r1 + c.w_structure * b.r2 + c.w_azimuth * b.r3
        norm = max(c.w_contrast + c.w_structure + c.w_azimuth, 1e-6)
        b.score = float((pos / norm) - 0.35 * (b.occlusion_penalty + b.truncation_penalty))
        return b

    def step(self, m: CropMetrics) -> RewardBreakdown:
        b = self.potential(m)
        if self.cfg.shaping:
            b.step_reward = b.score if self._prev is None else (b.score - self._prev)
            self._prev = b.score
        else:
            b.step_reward = b.score
        return b

    def terminal(self, m: CropMetrics, commit_bonus: float) -> float:
        """Payoff for choosing to stop. Committing on a bad zone must hurt."""
        b = self.potential(m)
        return float(commit_bonus * (2.0 * b.score - 0.6))
