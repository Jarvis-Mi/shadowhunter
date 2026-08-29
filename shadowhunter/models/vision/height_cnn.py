"""PyTorch height regressor - the second stage of the hybrid pipeline.

The RL agent hands over a clean, single-building shadow zone; this network
turns those pixels into metres. It is deliberately small: the research claim
is that *zone selection* buys the accuracy, not network capacity.

Physics is injected rather than learned: the solar elevation and the GSD enter
through a side channel, and the head predicts a residual around the analytic
estimate ``h = L*tan(theta)``. That keeps the model honest on unseen sun angles.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.logging import get_logger

log = get_logger(__name__)


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, cout), cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class HeightRegressor(nn.Module):
    """4-channel (RGB + shadow mask) -> building height in metres."""

    def __init__(self, in_channels: int = 4, width: int = 32, dropout: float = 0.1,
                 n_physics: int = 3) -> None:
        super().__init__()
        w = width
        self.stem = ConvBlock(in_channels, w, stride=2)
        self.stage1 = nn.Sequential(ConvBlock(w, w), ConvBlock(w, w * 2, stride=2))
        self.stage2 = nn.Sequential(ConvBlock(w * 2, w * 2), ConvBlock(w * 2, w * 4, stride=2))
        self.stage3 = nn.Sequential(ConvBlock(w * 4, w * 4), ConvBlock(w * 4, w * 8, stride=2))
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.physics = nn.Sequential(
            nn.Linear(n_physics, 32), nn.SiLU(inplace=True), nn.Linear(32, 32), nn.SiLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(w * 8 + 32, 128), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # (residual, log-sigma) - the model reports its own doubt
        )

    def forward(self, x: torch.Tensor, physics: torch.Tensor,
                analytic: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.stem(x)
        f = self.stage1(f)
        f = self.stage2(f)
        f = self.stage3(f)
        f = self.pool(f).flatten(1)
        p = self.physics(physics)
        out = self.head(torch.cat([f, p], dim=1))
        residual, log_sigma = out[:, 0], out[:, 1].clamp(-4.0, 3.0)
        base = analytic if analytic is not None else torch.zeros_like(residual)
        height = F.softplus(base + residual)
        return height, log_sigma.exp()


# --------------------------------------------------------------------------- #
# Tensor plumbing
# --------------------------------------------------------------------------- #
def prepare_batch(images_hwc: np.ndarray, sun_elev_deg: np.ndarray, gsd_m: np.ndarray,
                  shadow_len_px: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """Normalise a NHWC uint8 batch plus its physics side-channel."""
    x = torch.from_numpy(images_hwc).float().permute(0, 3, 1, 2) / 255.0
    x = (x - 0.45) / 0.25

    elev = np.asarray(sun_elev_deg, np.float32)
    gsd = np.asarray(gsd_m, np.float32)
    slen = np.asarray(shadow_len_px, np.float32)
    tan_e = np.tan(np.radians(np.clip(elev, 1.0, 85.0)))
    analytic = slen * gsd * tan_e

    physics = np.stack([np.radians(elev) / (math.pi / 2), gsd, np.log1p(slen) / 6.0], axis=1)
    return {
        "x": x.to(device),
        "physics": torch.from_numpy(physics.astype(np.float32)).to(device),
        "analytic": torch.from_numpy(analytic.astype(np.float32)).to(device),
    }


def gaussian_nll(pred: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic loss - the network is rewarded for calibrated doubt."""
    sigma = sigma.clamp(min=0.15)
    return (torch.log(sigma) + (pred - target) ** 2 / (2 * sigma ** 2)).mean()


def height_loss(pred: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor,
                warmup: bool) -> torch.Tensor:
    """Huber during warm-up, NLL afterwards.

    Starting straight on the NLL lets the model 'explain' its error by
    inflating sigma instead of moving the mean - the loss falls while the MAE
    sits still. Fitting the mean first, then handing it the variance term,
    removes that trap.
    """
    if warmup:
        return F.smooth_l1_loss(pred, target, beta=2.0)
    return gaussian_nll(pred, sigma, target)


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #
def train_regressor(images: np.ndarray, heights: np.ndarray, sun_elev: np.ndarray,
                    gsd: np.ndarray, shadow_len: np.ndarray, *,
                    epochs: int = 20, batch_size: int = 32, lr: float = 3e-4,
                    width: int = 32, device: str = "auto",
                    on_epoch=None) -> tuple[HeightRegressor, dict[str, Any]]:
    """Train from arrays. ``on_epoch(dict)`` streams progress to the UIs."""
    dev = resolve_device(device)
    model = HeightRegressor(in_channels=images.shape[-1], width=width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    n = len(images)
    idx = np.random.permutation(n)
    split = max(1, int(n * 0.85))
    tr, va = idx[:split], idx[split:]
    if len(va) == 0:
        va = tr[-max(1, n // 10):]

    warmup_epochs = max(1, epochs // 3)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(tr)
        losses = []
        warmup = epoch < warmup_epochs
        for i in range(0, len(tr), batch_size):
            sel = tr[i:i + batch_size]
            batch = prepare_batch(images[sel], sun_elev[sel], gsd[sel], shadow_len[sel], dev)
            target = torch.from_numpy(heights[sel].astype(np.float32)).to(dev)
            pred, sigma = model(batch["x"], batch["physics"], batch["analytic"])
            loss = height_loss(pred, sigma, target, warmup)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        model.eval()
        with torch.no_grad():
            batch = prepare_batch(images[va], sun_elev[va], gsd[va], shadow_len[va], dev)
            target = torch.from_numpy(heights[va].astype(np.float32)).to(dev)
            pred, _ = model(batch["x"], batch["physics"], batch["analytic"])
            mae = float(torch.abs(pred - target).mean().item())
            rmse = float(torch.sqrt(((pred - target) ** 2).mean()).item())

        rec = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_mae": mae,
               "val_rmse": rmse, "phase": "warmup" if warmup else "nll"}
        history.append(rec)
        log.info("cnn epoch %02d [%s]  loss=%.4f  MAE=%.2f m  RMSE=%.2f m",
                 epoch + 1, rec["phase"], rec["loss"], mae, rmse)
        if on_epoch:
            on_epoch(rec)

    return model, {"history": history, "device": str(dev), "n_samples": int(n)}


@torch.no_grad()
def predict_height(model: HeightRegressor, image_hwc: np.ndarray, sun_elev_deg: float,
                   gsd_m: float, shadow_len_px: float, device: str = "auto") -> tuple[float, float]:
    """Single-crop inference -> (height_m, sigma_m)."""
    dev = resolve_device(device)
    model = model.to(dev).eval()
    batch = prepare_batch(image_hwc[None, ...], np.array([sun_elev_deg]), np.array([gsd_m]),
                          np.array([shadow_len_px]), dev)
    pred, sigma = model(batch["x"], batch["physics"], batch["analytic"])
    return float(pred.item()), float(sigma.item())


def save_model(model: HeightRegressor, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {},
                "in_channels": model.stem.conv.in_channels}, path)
    return path


def load_model(path: str | Path, device: str = "auto") -> tuple[HeightRegressor, dict[str, Any]]:
    dev = resolve_device(device)
    try:
        blob = torch.load(path, map_location=dev, weights_only=True)
    except Exception:
        blob = torch.load(path, map_location=dev, weights_only=False)
    model = HeightRegressor(in_channels=blob.get("in_channels", 4))
    model.load_state_dict(blob["state_dict"])
    return model.to(dev).eval(), blob.get("meta", {})
