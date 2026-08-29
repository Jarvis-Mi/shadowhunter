"""Pydantic contracts shared by the API and every client.

MVT note: these are the *M* of MVT in its narrow sense - the shapes that cross
the process boundary. The heavy model code (CNN, RL, CV) lives next door under
``models/vision`` and ``models/rl``; this file is what the views are allowed
to depend on.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SunSpec(BaseModel):
    azimuth_deg: float = Field(148.0, ge=0, lt=360, description="Bearing the sun comes from")
    elevation_deg: float = Field(41.0, gt=0, lt=90, description="Solar elevation above horizon")
    gsd_m: float = Field(0.5, gt=0, description="Ground sample distance, metres per pixel")


class SceneSpec(BaseModel):
    """Either name an existing scene or ask for a fresh synthetic one."""
    name: str | None = None
    synthesize: bool = True
    size: int = Field(512, ge=128, le=2048)
    buildings: int = Field(14, ge=1, le=80)
    density: float = Field(1.0, ge=0.2, le=3.0)
    seed: int | None = None
    sun: SunSpec | None = None


class AnalyzeRequest(BaseModel):
    scene: SceneSpec = Field(default_factory=SceneSpec)
    start: tuple[int, int] | None = Field(None, description="Optional seed point (px)")
    policy: Literal["auto", "learned", "greedy"] = "auto"
    max_steps: int = Field(48, ge=4, le=256)
    return_overlay: bool = True
    return_trajectory: bool = True


class MetricsOut(BaseModel):
    shadow_ratio: float = 0.0
    contrast: float = 0.0
    isolation: float = 0.0
    components: int = 0
    entropy: float = 0.0
    edge_coherence: float = 0.0
    axis_alignment: float = 0.0
    elongation: float = 0.0
    truncation: float = 0.0
    occlusion: float = 0.0
    shadow_len_px: float = 0.0


class HeightEstimate(BaseModel):
    geometric_m: float = Field(..., description="h = L*tan(theta), from the mask alone")
    cnn_m: float | None = Field(None, description="Learned regressor output, if a model is loaded")
    fused_m: float = Field(..., description="Confidence-weighted blend of the two")
    sigma_m: float = Field(..., description="1-sigma uncertainty in metres")
    floors: int = 0
    confidence: float = Field(..., ge=0, le=1)


class AnalyzeResponse(BaseModel):
    scene: dict[str, Any]
    box: list[int]
    policy: str
    policy_requested: str = "auto"
    committed: bool
    steps: int
    score: float
    metrics: MetricsOut
    breakdown: dict[str, float]
    height: HeightEstimate
    trajectory: list[dict[str, Any]] = []
    overlay_png: str | None = Field(None, description="base64 PNG of the annotated scene")
    crop_png: str | None = Field(None, description="base64 PNG of the selected zone")
    elapsed_ms: float = 0.0


class SweepRequest(BaseModel):
    """Run the hunter over every labelled building in a synthetic scene."""
    scene: SceneSpec = Field(default_factory=SceneSpec)
    policy: Literal["auto", "learned", "greedy"] = "auto"
    limit: int = Field(12, ge=1, le=80)


class SweepItem(BaseModel):
    index: int
    box: list[int]
    truth_m: float | None = None
    estimate_m: float = 0.0
    error_m: float | None = None
    score: float = 0.0
    occlusion: float = 0.0
    wandered: bool = False
    matched: bool = True


class SweepResponse(BaseModel):
    scene: dict[str, Any]
    items: list[SweepItem]
    mae_m: float | None = None
    mae_all_m: float | None = None
    rmse_m: float | None = None
    mean_score: float = 0.0
    overlay_png: str | None = None
    elapsed_ms: float = 0.0


class TrainRLRequest(BaseModel):
    algo: Literal["PPO", "DQN"] = "PPO"
    total_timesteps: int = Field(20_000, ge=512, le=2_000_000)
    learning_rate: float = Field(3e-4, gt=0, lt=1)
    gamma: float = Field(0.98, gt=0, le=1)
    seed: int = 42
    tag: str = "shadow_hunter"


class TrainCNNRequest(BaseModel):
    scenes: int = Field(24, ge=1, le=400, description="Synthetic scenes to render")
    buildings: int = Field(14, ge=1, le=80)
    crop: int = Field(96, ge=32, le=256)
    epochs: int = Field(20, ge=1, le=500)
    batch_size: int = Field(32, ge=2, le=512)
    lr: float = Field(3e-4, gt=0, lt=1)
    tag: str = "height_cnn"


class JobStatus(BaseModel):
    id: str
    kind: Literal["rl", "cnn"]
    state: Literal["queued", "running", "done", "failed", "aborted"]
    progress: float = 0.0
    message: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict[str, Any] = {}
    artifact: str | None = None


class ArtifactInfo(BaseModel):
    name: str
    kind: Literal["rl", "cnn"]
    path: str
    size_bytes: int
    modified: str


class HealthResponse(BaseModel):
    status: str = "nominal"
    version: str
    torch: str | None = None
    cuda: bool = False
    device: str = "cpu"
    rasterio: bool = False
    sb3: str | None = None
    policy_loaded: bool = False
    cnn_loaded: bool = False
    uptime_s: float = 0.0


# --------------------------------------------------------------------------- #
# Map / AOI - real imagery
# --------------------------------------------------------------------------- #
class BBox(BaseModel):
    """Geographic rectangle, WGS84 degrees."""
    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-85.05, le=85.05)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-85.05, le=85.05)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (min(self.west, self.east), min(self.south, self.north),
                max(self.west, self.east), max(self.south, self.north))


class SunQuery(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    when: str | None = Field(None, description="ISO 8601 UTC; defaults to now")
    best_hour: bool = Field(False, description="Also return the best time of that day")


class SunResponse(BaseModel):
    azimuth_deg: float
    elevation_deg: float
    declination_deg: float
    is_daylight: bool
    when: str
    quality: float = Field(..., description="0..1 fitness of this geometry for shadow work")
    shadow_bearing_deg: float
    best_when: str | None = None
    best_elevation_deg: float | None = None
    sunrise: str | None = None
    sunset: str | None = None
    season: str | None = None
    seasons: list[dict[str, Any]] = []


class SurveyRequest(BaseModel):
    """One call: fetch the AOI imagery, place the sun, count the structures."""
    bbox: BBox
    provider: str = "esri"
    zoom: int | None = Field(None, description="None picks the finest affordable zoom")
    max_tiles: int = Field(36, ge=1, le=144)
    when: str | None = Field(None, description="ISO 8601 UTC acquisition time for the sun")
    auto_sun: bool = Field(False, description="Let the image's own shadows set the azimuth")
    min_size_m: float = Field(10.0, gt=0, le=200)
    max_structures: int = Field(80, ge=1, le=400)
    detect: bool = True


class StructureOut(BaseModel):
    box: list[int]
    center: list[int]
    score: float
    roof_score: float
    shadow_support: float
    area_px: int
    width_m: float
    height_m: float
    shadow_len_px: float
    quick_height_m: float
    index: int = 0
    lat: float | None = None
    lon: float | None = None
    seeded: bool = False


class SurveyResponse(BaseModel):
    aoi_id: str
    scene: dict[str, Any]
    sun: dict[str, Any]
    sun_estimate: dict[str, Any] = {}
    structures: list[StructureOut] = []
    count: int = 0
    with_shadow: int = 0
    mean_score: float = 0.0
    shadow_coverage: float = 0.0
    water_coverage: float = 0.0
    honesty: dict[str, Any] = {}
    image_png: str | None = None
    overlay_png: str | None = None
    elapsed_ms: float = 0.0
    season: str | None = None
    year_curve: list[dict[str, Any]] = []


class SurveyAnalyzeRequest(BaseModel):
    """Run the hunt + height pipeline over structures found by a survey."""
    aoi_id: str
    indices: list[int] | None = Field(None, description="None analyses every structure")
    policy: Literal["auto", "learned", "greedy"] = "auto"
    max_steps: int = Field(40, ge=4, le=256)
    limit: int = Field(24, ge=1, le=200)


class SurveyAnalyzeItem(BaseModel):
    index: int
    box: list[int]
    lon: float
    lat: float
    height_m: float
    sigma_m: float
    floors: int
    confidence: float
    score: float
    occlusion: float
    shadow_len_px: float
    truncation: float = 0.0


class SurveyAnalyzeResponse(BaseModel):
    aoi_id: str
    items: list[SurveyAnalyzeItem]
    tallest: SurveyAnalyzeItem | None = None
    mean_height_m: float = 0.0
    total_floors: int = 0
    overlay_png: str | None = None
    elapsed_ms: float = 0.0


class GeocodeResult(BaseModel):
    name: str
    lat: float
    lon: float
    bbox: list[float] | None = None
    kind: str | None = None


# --------------------------------------------------------------------------- #
# Copilot / intel / 3D
# --------------------------------------------------------------------------- #
class BriefRequest(BaseModel):
    aoi_id: str | None = None
    locale: Literal["fa", "en"] = "fa"


class Scene3DRequest(BaseModel):
    aoi_id: str
    open: bool = False
    measures: list[dict[str, Any]] | None = None


class FavoriteIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    bbox: list[float] | None = None
    note: str = ""


class PlanRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    span_m: float = Field(400.0, gt=0, le=50_000)
    width_m: float | None = None
    height_m: float | None = None


class FieldRunRequest(BaseModel):
    bbox: BBox
    locale: Literal["fa", "en"] = "fa"
    query: str | None = None
    when: str | None = None
