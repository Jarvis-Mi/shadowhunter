"""Scene endpoints: synthesise, list, preview, upload."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from ....core.config import SETTINGS
from ....models import pipeline
from ....models.schemas import SceneSpec
from ....models.vision.preprocess import save_scene, shadow_mask_preview

router = APIRouter(prefix="/api/scenes", tags=["scenes"])

SYNTH_DIR = SETTINGS.data_dir / "synthetic"
RAW_DIR = SETTINGS.data_dir / "raw"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@router.get("", summary="List available scenes on disk")
async def list_scenes():
    items = []
    for base, kind in ((SYNTH_DIR, "synthetic"), (RAW_DIR, "raw")):
        for p in sorted(base.glob("*")):
            if p.suffix.lower() in IMAGE_EXT:
                items.append({
                    "name": p.stem, "kind": kind, "path": str(p),
                    "size_bytes": p.stat().st_size, "format": p.suffix.lstrip("."),
                })
    return {"items": items, "count": len(items)}


@router.post("/synthesize", summary="Render and persist a synthetic tile")
async def synthesize(spec: SceneSpec):
    scene = pipeline.build_scene(spec.model_dump())
    path = save_scene(scene, SYNTH_DIR)
    meta = scene.meta()
    meta["path"] = str(path)
    meta["preview_png"] = pipeline.png_b64(scene.image)
    return meta


@router.get("/{name}/preview.png", summary="Rendered preview with shadow overlay")
async def preview(name: str, overlay: bool = True):
    path = _resolve(name)
    scene = pipeline.build_scene({"name": name, "synthesize": False}) if path else None
    if scene is None:
        raise HTTPException(status_code=404, detail=f"scene not found: {name}")
    img = shadow_mask_preview(scene.image) if overlay else scene.image
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.post("/upload", summary="Upload a GeoTIFF or raster tile")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "scene.png").suffix.lower()
    if suffix not in IMAGE_EXT:
        raise HTTPException(status_code=400, detail=f"unsupported format: {suffix}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / Path(file.filename).name
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"name": dest.stem, "path": str(dest), "size_bytes": dest.stat().st_size}


@router.delete("/{name}", summary="Delete a stored scene")
async def delete(name: str):
    path = _resolve(name)
    if not path:
        raise HTTPException(status_code=404, detail=f"scene not found: {name}")
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"deleted": name}


def _resolve(name: str) -> Path | None:
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="invalid scene name")
    for base in (SYNTH_DIR, RAW_DIR):
        root = base.resolve()
        for ext in IMAGE_EXT:
            p = (base / f"{name}{ext}").resolve()
            if not p.is_relative_to(root):
                raise HTTPException(status_code=400, detail="invalid scene name")
            if p.exists():
                return p
    return None
