"""Inference endpoints: health, single-zone analysis, full-scene sweep."""
from __future__ import annotations

import platform
import time

from fastapi import APIRouter, HTTPException

from ....core.config import SETTINGS
from ....core.logging import BUS, get_logger
from ....models import pipeline
from ....models.pipeline import REGISTRY
from ....models.schemas import (AnalyzeRequest, AnalyzeResponse, HealthResponse,
                                SweepRequest, SweepResponse)
from ....models.store import STORE

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])
_START = time.time()


@router.get("/health", response_model=HealthResponse, summary="Deck status")
async def health() -> HealthResponse:
    torch_v = cuda = device = None
    try:
        import torch

        torch_v = torch.__version__
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu"
    except Exception:
        cuda, device = False, "cpu"

    sb3_v = None
    try:
        import stable_baselines3

        sb3_v = stable_baselines3.__version__
    except Exception:
        pass

    from ....models.vision.preprocess import HAS_RASTERIO

    return HealthResponse(
        version="1.0.0", torch=torch_v, cuda=bool(cuda), device=str(device),
        rasterio=HAS_RASTERIO, sb3=sb3_v,
        policy_loaded=REGISTRY.policy is not None,
        cnn_loaded=REGISTRY.cnn is not None,
        uptime_s=round(time.time() - _START, 1),
    )


@router.post("/analyze", response_model=AnalyzeResponse, summary="Hunt one zone and estimate its height")
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        payload = pipeline.analyze(
            req.scene.model_dump(),
            start=tuple(req.start) if req.start else None,
            policy=req.policy,
            max_steps=req.max_steps,
            return_overlay=req.return_overlay,
            return_trajectory=req.return_trajectory,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # surface the real cause to the operator
        log.exception("analyze failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    STORE.record_analysis(payload)
    BUS.publish("analysis.done", box=payload["box"], height=payload["height"]["fused_m"],
                score=payload["score"], policy=payload["policy"])
    return AnalyzeResponse(**payload)


@router.post("/sweep", response_model=SweepResponse, summary="Hunt every building in a scene")
async def sweep(req: SweepRequest) -> SweepResponse:
    try:
        payload = pipeline.sweep(req.scene.model_dump(), policy=req.policy, limit=req.limit)
    except Exception as exc:
        log.exception("sweep failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    BUS.publish("sweep.done", n=len(payload["items"]), mae=payload["mae_m"])
    return SweepResponse(**payload)


@router.get("/history", summary="Recent analyses")
async def history(limit: int = 25):
    return {"items": STORE.recent_analyses(limit)}


@router.get("/config", summary="Effective runtime configuration")
async def config():
    return SETTINGS.to_dict()
