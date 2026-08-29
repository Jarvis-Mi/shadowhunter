"""Training endpoints - RL and CNN runs execute in background threads.

Progress is not polled through this router; it is pushed onto the event bus
and out over the websocket in ``telemetry.py``. Polling ``/api/train/jobs``
remains available for clients that prefer it (DearPyGui, curl).
"""
from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException

from ....core.config import SETTINGS, CNNConfig, RLConfig
from ....core.logging import BUS, get_logger
from ....models.pipeline import REGISTRY
from ....models.schemas import ArtifactInfo, JobStatus, TrainCNNRequest, TrainRLRequest
from ....models.store import STORE

log = get_logger(__name__)
router = APIRouter(prefix="/api/train", tags=["training"])


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
def _run_rl(job_id: str, req: TrainRLRequest) -> None:
    from ....models.rl.agent import train_agent

    STORE.update(job_id, state="running", message=f"{req.algo} warming up")
    BUS.publish("job.state", id=job_id, kind="rl", state="running")
    try:
        cfg = RLConfig(algo=req.algo, total_timesteps=req.total_timesteps,
                       learning_rate=req.learning_rate, gamma=req.gamma, seed=req.seed)
        out_path = SETTINGS.checkpoint_dir / f"{req.tag}_{req.algo.lower()}_{job_id}.zip"

        def sink(payload):
            STORE.update(job_id, progress=payload["progress"],
                         message=f"{payload['timesteps']}/{payload['total']} steps"
                                 f"  |  R={payload['ep_reward_mean']}",
                         metrics=payload)

        result = train_agent(cfg, save_path=out_path, run_id=job_id,
                             stop_flag=STORE.stop_flag(job_id), sink=sink)
        aborted = STORE.stop_flag(job_id).get("stop")
        STORE.finish(job_id, "aborted" if aborted else "done",
                     progress=1.0 if not aborted else STORE.get(job_id)["progress"],
                     artifact=str(out_path), message="policy saved")
        REGISTRY.load_policy(out_path, req.algo)
        BUS.publish("job.state", id=job_id, kind="rl",
                    state="aborted" if aborted else "done", artifact=str(out_path), **result)
    except Exception as exc:
        log.error("rl job %s failed: %s", job_id, exc)
        STORE.finish(job_id, "failed", message=f"{type(exc).__name__}: {exc}")
        BUS.publish("job.state", id=job_id, kind="rl", state="failed",
                    error=traceback.format_exc(limit=3))


def _run_cnn(job_id: str, req: TrainCNNRequest) -> None:
    from ....models.vision.height_cnn import save_model, train_regressor
    from ....models.vision.preprocess import crop_dataset, synthesize_scene
    from ....models.vision.shadow_ops import analyse_crop, shadow_mask

    STORE.update(job_id, state="running", message="rendering scenes")
    BUS.publish("job.state", id=job_id, kind="cnn", state="running")
    try:
        images, heights, elevs, gsds, lens = [], [], [], [], []
        for i in range(req.scenes):
            scene = synthesize_scene(size=512, n_buildings=req.buildings, seed=1000 + i)
            X, y = crop_dataset(scene, crop=req.crop, seed=i)
            for patch, h in zip(X, y):
                bgr = patch[:, :, :3]
                m = analyse_crop(bgr, scene.sun, shadow_mask(bgr))
                images.append(patch)
                heights.append(h)
                elevs.append(scene.sun.elevation_deg)
                gsds.append(scene.sun.gsd_m)
                lens.append(m.shadow_len_px)
            STORE.update(job_id, progress=0.25 * (i + 1) / req.scenes,
                         message=f"scene {i + 1}/{req.scenes}  |  {len(images)} crops")
            if STORE.stop_flag(job_id).get("stop"):
                STORE.finish(job_id, "aborted", message="aborted while building dataset")
                return

        if not images:
            raise RuntimeError("dataset is empty - increase scenes or buildings")

        images = np.stack(images)
        heights = np.asarray(heights, np.float32)
        elevs = np.asarray(elevs, np.float32)
        gsds = np.asarray(gsds, np.float32)
        lens = np.asarray(lens, np.float32)
        BUS.publish("cnn.dataset", n=int(len(images)), shape=list(images.shape[1:]))

        def on_epoch(rec):
            frac = 0.25 + 0.75 * rec["epoch"] / max(req.epochs, 1)
            STORE.update(job_id, progress=round(frac, 4),
                         message=f"epoch {rec['epoch']}/{req.epochs}  |  MAE {rec['val_mae']:.2f} m",
                         metrics=rec)
            BUS.publish("cnn.progress", id=job_id, **rec, progress=round(frac, 4))
            if STORE.stop_flag(job_id).get("stop"):
                raise KeyboardInterrupt("aborted by operator")

        model, meta = train_regressor(
            images, heights, elevs, gsds, lens,
            epochs=req.epochs, batch_size=req.batch_size, lr=req.lr,
            width=CNNConfig().width, on_epoch=on_epoch,
        )
        out_path = SETTINGS.checkpoint_dir / f"{req.tag}_{job_id}.pt"
        save_model(model, out_path, meta)
        STORE.finish(job_id, "done", progress=1.0, artifact=str(out_path),
                     message=f"MAE {meta['history'][-1]['val_mae']:.2f} m", metrics=meta["history"][-1])
        REGISTRY.load_cnn(out_path)
        BUS.publish("job.state", id=job_id, kind="cnn", state="done", artifact=str(out_path))
    except KeyboardInterrupt:
        STORE.finish(job_id, "aborted", message="aborted by operator")
        BUS.publish("job.state", id=job_id, kind="cnn", state="aborted")
    except Exception as exc:
        log.error("cnn job %s failed: %s", job_id, exc)
        STORE.finish(job_id, "failed", message=f"{type(exc).__name__}: {exc}")
        BUS.publish("job.state", id=job_id, kind="cnn", state="failed",
                    error=traceback.format_exc(limit=3))


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.post("/rl", response_model=JobStatus, summary="Train the zone-selection policy")
async def train_rl(req: TrainRLRequest, background: BackgroundTasks) -> JobStatus:
    job_id = STORE.create("rl", req.model_dump())
    background.add_task(lambda: threading.Thread(target=_run_rl, args=(job_id, req), daemon=True).start())
    return _status(job_id)


@router.post("/cnn", response_model=JobStatus, summary="Train the height regressor")
async def train_cnn(req: TrainCNNRequest, background: BackgroundTasks) -> JobStatus:
    job_id = STORE.create("cnn", req.model_dump())
    background.add_task(lambda: threading.Thread(target=_run_cnn, args=(job_id, req), daemon=True).start())
    return _status(job_id)


@router.get("/jobs", summary="All jobs, newest first")
async def jobs(limit: int = 40):
    return {"items": [_status(j["id"]).model_dump() for j in STORE.list(limit)]}


@router.get("/jobs/{job_id}", response_model=JobStatus, summary="One job")
async def job(job_id: str) -> JobStatus:
    if STORE.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return _status(job_id)


@router.post("/jobs/{job_id}/abort", summary="Cooperatively stop a running job")
async def abort(job_id: str):
    if not STORE.request_stop(job_id):
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return {"aborting": job_id}


@router.get("/artifacts", summary="Checkpoints on disk")
async def artifacts() -> dict[str, list[ArtifactInfo]]:
    items: list[ArtifactInfo] = []
    for p in sorted(Path(SETTINGS.checkpoint_dir).glob("*")):
        if p.suffix not in {".zip", ".pt"}:
            continue
        st = p.stat()
        items.append(ArtifactInfo(
            name=p.name, kind="rl" if p.suffix == ".zip" else "cnn", path=str(p),
            size_bytes=st.st_size,
            modified=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        ))
    return {"items": items}


@router.post("/artifacts/{name}/load", summary="Activate a checkpoint")
async def load_artifact(name: str):
    ckpt_dir = Path(SETTINGS.checkpoint_dir).resolve()
    path = (ckpt_dir / Path(name).name).resolve()
    if not path.is_relative_to(ckpt_dir):
        raise HTTPException(status_code=400, detail="invalid artifact name")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no such artifact: {name}")
    if path.suffix == ".zip":
        ok = REGISTRY.load_policy(path, "DQN" if "dqn" in path.stem.lower() else "PPO")
        kind = "rl"
    else:
        ok = REGISTRY.load_cnn(path)
        kind = "cnn"
    if not ok:
        raise HTTPException(status_code=500, detail="checkpoint could not be loaded")
    return {"loaded": name, "kind": kind}


def _status(job_id: str) -> JobStatus:
    row = STORE.get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return JobStatus(
        id=row["id"], kind=row["kind"], state=row["state"], progress=row["progress"],
        message=row["message"], started_at=row["started_at"], finished_at=row["finished_at"],
        metrics=row["metrics"], artifact=row["artifact"],
    )
