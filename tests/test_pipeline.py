"""End-to-end checks that need no downloads, no GPU and no trained weights."""
from __future__ import annotations

import math

import numpy as np
import pytest

from shadowhunter.core.config import EnvConfig, RewardConfig
from shadowhunter.core.geo import (SunGeometry, height_from_shadow, quality_of_geometry,
                                   shadow_from_height)
from shadowhunter.models import pipeline
from shadowhunter.models.rl.env import ACTION_NAMES, COMMIT, ShadowHunterEnv
from shadowhunter.models.rl.rewards import (CompositeReward, r1_contrast_isolation,
                                            r2_structural_purity, r3_azimuth_coherence)
from shadowhunter.models.vision.preprocess import crop_dataset, safe_crop, synthesize_scene
from shadowhunter.models.vision.shadow_ops import analyse_crop, largest_blob, shadow_mask
from shadowhunter.templates import theme


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_shadow_height_roundtrip():
    sun = SunGeometry(azimuth_deg=150, elevation_deg=35, gsd_m=0.5)
    for height in (6.0, 18.5, 44.0):
        px = shadow_from_height(height, sun)
        assert height_from_shadow(px, sun) == pytest.approx(height, rel=1e-6)


def test_geometry_quality_prefers_low_sun():
    low = quality_of_geometry(SunGeometry(elevation_deg=30))
    high = quality_of_geometry(SunGeometry(elevation_deg=80))
    assert low > high


def test_shadow_bearing_is_opposite_the_sun():
    sun = SunGeometry(azimuth_deg=90)
    assert sun.shadow_bearing_deg == pytest.approx(270.0)


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
def test_synthetic_scene_has_detectable_shadow():
    scene = synthesize_scene(size=320, n_buildings=8, seed=11)
    mask = shadow_mask(scene.image)
    coverage = (mask > 0).mean()
    assert 0.02 < coverage < 0.9, f"implausible shadow coverage {coverage:.3f}"
    blob, info = largest_blob(mask)
    assert info["area"] > 0


def test_metrics_are_bounded():
    scene = synthesize_scene(size=320, n_buildings=8, seed=12)
    crop = safe_crop(scene.image, (60, 60, 96, 96))
    m = analyse_crop(crop, scene.sun, shadow_mask(crop))
    for name, value in m.to_dict().items():
        if name in {"components", "shadow_len_px"}:
            continue
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_crop_dataset_shapes():
    scene = synthesize_scene(size=320, n_buildings=8, seed=13)
    X, y = crop_dataset(scene, crop=64, seed=0)
    assert X.ndim == 4 and X.shape[1:] == (64, 64, 4)
    assert len(X) == len(y) and len(X) > 0
    assert (y > 0).all()


# --------------------------------------------------------------------------- #
# Rewards
# --------------------------------------------------------------------------- #
def test_proxy_rewards_need_no_ground_truth():
    """A blank crop must score ~0; a real shadow crop must score higher."""
    scene = synthesize_scene(size=320, n_buildings=10, seed=14)
    blank = np.full((96, 96, 3), 140, np.uint8)
    m_blank = analyse_crop(blank, scene.sun, shadow_mask(blank))
    assert r1_contrast_isolation(m_blank) < 0.05

    best = 0.0
    for x in range(0, 220, 32):
        for y in range(0, 220, 32):
            crop = safe_crop(scene.image, (x, y, 96, 96))
            m = analyse_crop(crop, scene.sun, shadow_mask(crop))
            best = max(best, r1_contrast_isolation(m) + r2_structural_purity(m)
                       + r3_azimuth_coherence(m))
    assert best > 0.2, "no crop in a synthetic city scored above noise"


def test_shaping_returns_deltas():
    scene = synthesize_scene(size=256, n_buildings=6, seed=15)
    crop = safe_crop(scene.image, (40, 40, 96, 96))
    m = analyse_crop(crop, scene.sun, shadow_mask(crop))

    shaped = CompositeReward(RewardConfig(shaping=True))
    first = shaped.step(m).step_reward
    second = shaped.step(m).step_reward
    assert second == pytest.approx(0.0, abs=1e-9), "identical state must yield zero shaped reward"
    assert first != 0.0 or shaped.potential(m).score == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def test_env_contract():
    env = ShadowHunterEnv(cfg=EnvConfig(tile_size=256, max_steps=12), seed=1)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert obs["view"].shape == (4, env.cfg.obs_size, env.cfg.obs_size)

    total = 0.0
    for action in range(len(ACTION_NAMES) - 1):      # every move except COMMIT
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        assert env.observation_space.contains(obs)
        assert not terminated
        if truncated:
            break
    assert math.isfinite(total)


def test_commit_terminates():
    env = ShadowHunterEnv(cfg=EnvConfig(tile_size=256, max_steps=20), seed=2)
    env.reset(seed=2)
    _, _, terminated, _, info = env.step(COMMIT)
    assert terminated and info["committed"]


def test_window_stays_inside_the_scene():
    env = ShadowHunterEnv(cfg=EnvConfig(tile_size=256, max_steps=200), seed=3)
    env.reset(seed=3)
    for _ in range(60):
        env.step(2)          # push west forever
    x, y, w, h = env.box
    assert x >= 0 and y >= 0 and x + w <= 256 and y + h <= 256


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_analyze_without_any_trained_model():
    """The greedy fallback must carry the whole product on a cold install."""
    scene = synthesize_scene(size=320, n_buildings=10, seed=16)
    out = pipeline.analyze(scene=scene, policy="greedy", return_overlay=True)

    assert out["policy"] == "greedy"
    assert len(out["box"]) == 4
    assert out["height"]["fused_m"] > 0
    assert out["height"]["sigma_m"] > 0
    assert 0.0 <= out["height"]["confidence"] <= 1.0
    assert out["overlay_png"] and out["crop_png"]
    assert out["trajectory"], "greedy hunt must report its search path"


def test_learned_without_registry_raises():
    from shadowhunter.models.pipeline import REGISTRY

    scene = synthesize_scene(size=192, n_buildings=4, seed=18)
    previous = REGISTRY.policy
    REGISTRY.policy = None
    try:
        with pytest.raises(ValueError, match="no learned policy"):
            pipeline.analyze(scene=scene, policy="learned", return_overlay=False)
    finally:
        REGISTRY.policy = previous


def test_sweep_scores_against_ground_truth():
    scene = synthesize_scene(size=384, n_buildings=8, seed=17)
    out = pipeline.sweep(scene=scene, policy="greedy", limit=4)
    assert len(out["items"]) == 4
    assert out["mae_m"] is not None and out["mae_m"] >= 0
    for item in out["items"]:
        assert item["truth_m"] > 0 and item["estimate_m"] > 0


# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
def test_every_view_can_render_the_same_tokens():
    assert theme.qss().startswith("/*") and "#FFB020" in theme.qss()
    assert "--sh-solar: #FFB020;" in theme.css_variables()
    assert theme.flet_theme_dict()["primary"] == "#FFB020"
    assert theme.ctk_theme_dict()["CTkFrame"]["fg_color"][0] == theme.c("panel")
    assert theme.dpg_palette()["solar"] == (255, 176, 32, 255)


def test_qss_has_no_unresolved_placeholders():
    rendered = theme.qss()
    assert "{{" not in rendered and "}}" not in rendered
