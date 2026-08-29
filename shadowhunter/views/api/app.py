"""FastAPI application - the single backend all five UIs talk to.

MVT note: this is a *view* in the Django sense (a controller). It owns no
domain logic; it validates, delegates to ``models.pipeline`` and serialises.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ...core.config import SETTINGS
from ...core.jsonutil import FiniteJSONResponse
from ...core.logging import get_logger
from ...models.pipeline import REGISTRY
from .routers import analysis, atlas, copilot, scenes, telemetry, training

log = get_logger(__name__)
START_TIME = time.time()
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warmup() is main-thread-safe: under uvicorn it loads the SB3 policy here,
    # under TestClient (whose lifespan runs on an anyio portal thread) it defers
    # the policy rather than corrupting the heap. See Registry's docstring.
    loaded = REGISTRY.warmup()
    log.info("checkpoints -> policy=%s cnn=%s%s", loaded["policy"], loaded["cnn"],
             "  (policy deferred to main thread)"
             if loaded.get("policy_deferred") else "")
    log.info("deck online at %s", SETTINGS.api_base)
    yield
    log.info("deck offline")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Shadow Hunter",
        version=VERSION,
        summary="Hybrid DRL + CNN building-height estimation from free satellite imagery",
        description=(
            "Draw a rectangle anywhere on Earth. The deck stitches free satellite "
            "tiles for it, places the sun from latitude, longitude and time, counts "
            "the structures inside, then sends an RL agent hunting for each one's "
            "clean, unoccluded shadow before a CNN turns it into metres. Every "
            "reward the agent sees comes from pixels alone - no ground-truth "
            "heights are needed during exploration."
        ),
        lifespan=lifespan,
        default_response_class=FiniteJSONResponse,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    origins = SETTINGS.server.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(analysis.router)
    app.include_router(atlas.router)
    app.include_router(copilot.router)
    app.include_router(scenes.router)
    app.include_router(training.router)
    app.include_router(telemetry.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return FiniteJSONResponse({
            "name": "Shadow Hunter",
            "version": VERSION,
            "docs": "/docs",
            "health": "/api/health",
            "uptime_s": round(time.time() - START_TIME, 1),
        })

    return app


app = create_app()
