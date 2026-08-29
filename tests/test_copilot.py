"""Copilot stack, raster inspect, timeline, honesty, cache — no network, no GPU."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math

import numpy as np

from shadowhunter.core.cache import Cache, hash_key
from shadowhunter.core.geo import SunGeometry
from shadowhunter.core.jsonutil import finite
from shadowhunter.core.sun import resolve_illumination
from shadowhunter.models.honesty import imagery_honesty
from shadowhunter.models.inspect import inspect_raster, inspect_to_jsonable
from shadowhunter.models.reconstruct import compose_field, export_scene
from shadowhunter.models.timeline import build_timeline, hour_samples, render_timeline_strip
from shadowhunter.models.vision.detect import seed_frame_structure
from shadowhunter.models.vision.preprocess import synthesize_scene


def test_inspect_reports_bgr_layout_and_histogram():
    scene = synthesize_scene(size=192, n_buildings=6, seed=31)
    raw = inspect_raster(scene.image)
    assert raw["layout"] == "BGR"
    assert raw["channels"] == 3
    assert raw["bit_depth"] == 8
    assert raw["width"] == 192 and raw["height"] == 192
    assert len(raw["histogram"]["r"]) == 32
    assert 0.0 <= raw["shadow_fraction"] <= 1.0
    assert raw["_depth_u8"].shape == (192, 192)
    clean = inspect_to_jsonable(raw)
    assert "_depth_u8" not in clean
    assert "rgb_mean" in clean


def test_timeline_has_daylight_and_strip():
    when = datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc)
    samples = hour_samples(25.08, 55.14, when, step_minutes=60)
    assert len(samples) == 24
    assert any(s["is_daylight"] for s in samples)
    strip = render_timeline_strip(samples)
    assert strip.shape[2] == 3 and strip.shape[0] >= 80
    payload = build_timeline(25.08, 55.14, when)
    assert payload["best_when"]
    assert payload["best_elevation"] > 0


def test_finite_json_drops_inf_and_nan():
    encoded = json.dumps(finite({"a": math.inf, "b": float("nan"), "c": [1.0, -math.inf]}),
                         allow_nan=False)
    assert json.loads(encoded) == {"a": None, "b": None, "c": [1.0, None]}


def test_timeline_payload_is_json_safe_at_night():
    when = datetime(2026, 8, 28, 21, 0, tzinfo=timezone.utc)
    payload = build_timeline(35.70, 51.34, when, step_minutes=60)
    json.dumps(finite(payload), allow_nan=False)


def test_honesty_marks_basemap_indicative():
    flags = imagery_honesty(provider="esri", source="basemap", is_daylight=True, quality=0.7)
    assert flags["height_indicative"] is True
    assert flags["elevation_trusted"] is False
    assert flags["verdict"] == "undated_basemap"
    night = imagery_honesty(provider="esri", is_daylight=False, quality=0.1)
    assert night["verdict"] == "night"


def test_cache_roundtrip(tmp_path: Path):
    c = Cache(root=tmp_path / "cache", capacity=8)
    key = hash_key("demo", 1)
    assert c.get("ns", key) is None
    c.set("ns", key, {"ok": True}, ttl_s=60)
    assert c.get("ns", key) == {"ok": True}
    stats = c.stats()
    assert stats["entries_mem"] >= 1


def test_reconstruct_writes_workspace(tmp_path: Path):
    scene = synthesize_scene(size=160, n_buildings=4, seed=32)
    structures = [{
        "box": [20, 20, 30, 40], "index": 0, "quick_height_m": 18.0,
    }]
    out = export_scene(
        tmp_path / "aoi", mosaic_bgr=scene.image, structures=structures,
        measures=[{"index": 0, "height_m": 22.0}], gsd_m=0.5,
        sun={"azimuth_deg": 150, "elevation_deg": 35}, meta={"name": "test"},
    )
    assert Path(out["html"]).exists()
    html = Path(out["html"]).read_text(encoding="utf-8")
    assert "three" in html.lower()
    assert out["buildings"] == 1
    assert Path(out["json"]).exists()


def test_compose_field_samples_rgb_and_rejects_squat_hunt():
    scene = synthesize_scene(size=160, n_buildings=3, seed=11)
    field = compose_field(
        scene.image,
        [{"box": [10, 10, 80, 90], "index": 0, "width_m": 40.0, "height_m": 45.0,
          "quick_height_m": 4.0, "shadow_len_px": 8.0, "score": 0.4}],
        [{"index": 0, "height_m": 13.1}],
        {"azimuth_deg": 150, "elevation_deg": 35, "gsd_m": 0.5},
        0.5,
    )
    b = field["buildings"][0]
    assert len(b["rgb"]) == 3
    assert b["height_m"] > 20
    assert b["height_source"] in {"footprint", "shadow"}
    assert "shadow_dx_m" in b
    assert field["viable"] is True


def test_covering_seed_falls_back_from_volume():
    img = np.zeros((80, 120, 3), np.uint8)
    img[:] = (40, 90, 130)
    field = compose_field(
        img,
        [{"box": [0, 0, 120, 80], "index": 0, "seeded": True,
          "width_m": 60.0, "height_m": 40.0, "score": 0.3,
          "shadow_len_px": 0.0, "shadow_support": 0.0}],
        None,
        {"azimuth_deg": 150, "elevation_deg": 30, "gsd_m": 0.5},
        0.5,
    )
    assert field["viable"] is False
    assert field["buildings"][0]["cover"] is True


def test_night_tehran_snaps_to_daylight():
    night = datetime(2026, 8, 28, 21, 20, tzinfo=timezone.utc)
    moment, pos, sun, snapped = resolve_illumination(35.6997, 51.3380, night, 0.5)
    assert snapped is True
    assert pos.is_daylight
    assert sun.elevation_deg > 8.0


def test_seed_frame_and_empty_3d(tmp_path: Path):
    scene = synthesize_scene(size=160, n_buildings=3, seed=9)
    item = seed_frame_structure(scene.image, scene.sun)
    assert item["seeded"] is True
    assert item["width_m"] > 0
    empty = export_scene(
        tmp_path / "empty3d", mosaic_bgr=scene.image, structures=[],
        measures=None, gsd_m=0.5,
        sun={"azimuth_deg": 150, "elevation_deg": 35}, meta={"name": "empty"},
    )
    assert empty["buildings"] >= 1
    assert Path(empty["html"]).exists()


def test_scene3d_accepts_positional_measures():
    import inspect

    from shadowhunter.services.client import DeckClient

    params = inspect.signature(DeckClient.scene3d).parameters
    assert params["measures"].kind != inspect.Parameter.KEYWORD_ONLY


def test_learned_policy_without_weights_is_409():
    from fastapi.testclient import TestClient
    from shadowhunter.models.pipeline import REGISTRY
    from shadowhunter.views.api.app import app

    previous_policy, previous_cnn = REGISTRY.policy, REGISTRY.cnn
    try:
        with TestClient(app) as client:
            # Lifespan autoloads checkpoints; unload after startup.
            REGISTRY.policy = None
            r = client.post("/api/analyze", json={
                "scene": {"synthesize": True, "size": 192, "buildings": 4, "seed": 40},
                "policy": "learned", "max_steps": 8, "return_overlay": False,
            })
            assert r.status_code == 409
    finally:
        REGISTRY.policy = previous_policy
        REGISTRY.cnn = previous_cnn


def test_survey_rejects_carto_basemap():
    from fastapi.testclient import TestClient
    from shadowhunter.views.api.app import app

    with TestClient(app) as client:
        r = client.post("/api/aoi/survey", json={
            "bbox": {"west": 55.13, "south": 25.07, "east": 55.14, "north": 25.08},
            "provider": "carto_dark", "detect": False, "max_tiles": 4,
        })
        assert r.status_code == 400


def test_brain_health_and_timeline_routes():
    from fastapi.testclient import TestClient
    from shadowhunter.views.api.app import app

    with TestClient(app) as client:
        health = client.get("/api/brain/health")
        assert health.status_code == 200
        body_h = health.json()
        assert "models" in body_h
        cfg = body_h["configured"]
        assert cfg["llm"] == "qwen3.5:4b"
        assert cfg["embed"] == "mxbai-embed-large:latest"
        assert cfg["vlm"] == "glm-ocr:latest"
        tl = client.get("/api/timeline?lat=25.08&lon=55.14&when=2026-06-21T08:00:00Z")
        assert tl.status_code == 200
        body = tl.json()
        assert body["samples"] and body["strip_png"]
        assert [c["local_hour"] for c in body["captures"]] == [10, 12, 13, 14, 15, 16]
        assert len(body["calendar"]["seasons"]) == 4
        brief = client.post("/api/brain/brief", json={"locale": "fa"})
        assert brief.status_code == 200
        report = brief.json()
        assert report["verdict"]
        assert report["report_fa"]


def test_offline_brain_always_returns_a_verdict(monkeypatch):
    from shadowhunter.core.brain import decide

    monkeypatch.setattr("shadowhunter.core.brain.llm_up", lambda: False)

    out = decide({
        "scene": {"provider": "esri", "source": "basemap"},
        "sun": {"elevation_deg": 32, "is_daylight": True, "quality": 0.7},
        "count": 4,
    }, locale="fa")
    assert out["provider"] == "offline" or out["verdict"]
    assert out["honesty"]["height_indicative"] is True
    assert "INDICATIVE" in (out["report_en"] or "").upper() or out["honesty"]["height_indicative"]


def test_openapi_includes_copilot_routes():
    from fastapi.testclient import TestClient
    from shadowhunter.views.api.app import app

    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
        for route in ("/api/brain/health", "/api/timeline", "/api/intel", "/api/scene3d",
                      "/api/field/run", "/api/places/favorites", "/api/brain/plan",
                      "/api/brain/construct"):
            assert route in paths


def test_year_seasons_cover_four_markers():
    from shadowhunter.core.sun import year_seasons

    curve = year_seasons(35.6997, 51.3380, 2026)
    names = [row["season"] for row in curve]
    assert names == ["spring", "summer", "autumn", "winter"]
    assert all(0.0 <= float(row["quality"]) <= 1.0 for row in curve)


def test_inspect_exposes_spectra_and_depth_arrays():
    scene = synthesize_scene(size=128, n_buildings=4, seed=7)
    raw = inspect_raster(scene.image)
    assert set(raw["spectra"]) >= {"rgb", "hsv", "lab", "false_color", "shadow_boost"}
    assert raw["_false_u8"].shape[:2] == (128, 128)
    assert raw["_lab_u8"].shape[:2] == (128, 128)
    clean = inspect_to_jsonable(raw)
    assert "spectra" in clean and "_false_u8" not in clean


def test_places_favorites_and_history(tmp_path: Path, monkeypatch):
    from shadowhunter.models import places

    monkeypatch.setattr(places.SETTINGS, "data_dir", tmp_path)
    item = places.add_favorite("Azadi", 35.6997, 51.3380, note="tower")
    listed = places.list_favorites()
    assert listed and listed[0]["id"] == item["id"]
    places.record_event("search", query="Azadi", lat=35.6997, lon=51.3380, name="Azadi")
    hist = places.list_history()
    assert hist and hist[0]["kind"] == "search"
    assert places.remove_favorite(item["id"]) is True
    assert places.list_favorites() == []


def test_plan_run_and_dossier_offline(monkeypatch):
    from shadowhunter.core.brain import decide, plan_run

    monkeypatch.setattr("shadowhunter.core.brain.llm_up", lambda: False)

    plan = plan_run({"lat": 35.7, "lon": 51.34, "span_m": 220})
    assert plan["auto_sun"] is True
    assert plan["prefer_year"] is True
    assert 6 <= plan["min_size_m"] <= 40
    assert "greedy_hunt" in plan["tools"]
    out = decide({
        "aoi_id": "test-azadi",
        "scene": {"provider": "esri", "source": "basemap", "center": [51.338, 35.6997],
                  "name": "Azadi"},
        "sun": {"elevation_deg": 32, "is_daylight": True, "quality": 0.7,
                "season": "winter", "when": "2026-12-21T08:00+00:00"},
        "count": 3,
    }, locale="fa")
    assert out["dossier"]["season"] == "winter"
    assert abs(float(out["dossier"]["lat"]) - 35.6997) < 1e-4
    assert any(t["id"] == "height_cnn" for t in out["dossier"]["tools"])
def test_azadi_landmark_and_stated_extrusion():
    from shadowhunter.core.geo import SunGeometry
    from shadowhunter.models.landmarks import lookup
    from shadowhunter.models.reconstruct import compose_field
    from shadowhunter.models.vision.detect import compact_footprint

    hit = lookup(35.70013, 51.3380)
    assert hit is not None
    assert abs(float(hit["height_m"]) - 43.0) < 0.01
    img = np.full((200, 240, 3), 36, np.uint8)
    img[70:130, 90:150] = (210, 214, 220)
    sun = SunGeometry(150, 35, 0.5)
    foot = compact_footprint(img, sun, min_size_m=8.0)
    assert foot is not None
    x, y, w, h = foot["box"]
    assert w * h < 0.5 * 200 * 240
    assert 80 < x + w / 2 < 160
    field = compose_field(
        img,
        [{**foot, "stated_height_m": 43.0, "index": 0}],
        [{"index": 0, "height_m": 4.3, "cnn_m": None, "geometric_m": 4.3}],
        {"azimuth_deg": 150, "elevation_deg": 35, "gsd_m": 0.5},
        0.5,
    )
    b = field["buildings"][0]
    assert abs(b["height_m"] - 43.0) < 0.05
    assert b["height_source"] == "stated"
    assert field["viable"] is True
    assert b["cover"] is False


def test_compose_field_extrudes_silhouette_not_aabb():
    """Azadi-like inverted-Y must become a prism outline, not a gray rectangle."""
    import cv2

    img = np.full((180, 220, 3), 28, np.uint8)
    y_shape = np.array([
        [30, 50], [95, 50], [105, 88], [115, 50], [190, 50],
        [200, 95], [130, 130], [90, 130], [20, 95],
    ], np.int32)
    cv2.fillPoly(img, [y_shape], (214, 218, 224))
    cv2.circle(img, (110, 78), 16, (36, 38, 42), -1)
    field = compose_field(
        img,
        [{"box": [18, 40, 190, 100], "index": 0, "compact": True, "score": 0.6}],
        [{"index": 0, "height_m": 7.7}],
        {"azimuth_deg": 198, "elevation_deg": 30, "gsd_m": 0.25},
        0.25,
        {"landmark": {"name": "Azadi Tower", "height_m": 43.0}},
    )
    b = field["buildings"][0]
    assert abs(b["height_m"] - 43.0) < 0.05
    assert b["height_source"] == "stated"
    assert b.get("outline_m") and len(b["outline_m"]) >= 5
    xs = [p[0] for p in b["outline_m"]]
    ys = [p[1] for p in b["outline_m"]]
    # Silhouette bbox should be tighter than the padded detect rectangle in at least one axis.
    assert (max(xs) - min(xs)) < 190 * 0.25 * 0.98


def test_tight_seeded_aoi_with_stated_height_is_volume():
    """Operator crop around Azadi used to fall back to luminance relief (score 0.35 seed)."""
    import cv2

    img = np.full((160, 140, 3), 32, np.uint8)
    y_shape = np.array([
        [18, 28], [60, 28], [68, 62], [76, 28], [122, 28],
        [128, 70], [82, 110], [56, 110], [12, 70],
    ], np.int32)
    cv2.fillPoly(img, [y_shape], (216, 220, 226))
    cv2.circle(img, (70, 52), 12, (40, 42, 46), -1)
    field = compose_field(
        img,
        [{"box": [6, 6, 128, 148], "index": 0, "seeded": True, "score": 0.35,
          "shadow_support": 0.0, "shadow_len_px": 0.0, "stated_height_m": 43.0}],
        [{"index": 0, "height_m": 5.5}],
        {"azimuth_deg": 81, "elevation_deg": 30, "gsd_m": 0.24},
        0.24,
        {"landmark": {"name": "Azadi Tower", "height_m": 43.0}},
    )
    b = field["buildings"][0]
    assert b["cover"] is False
    assert field["viable"] is True
    assert abs(b["height_m"] - 43.0) < 0.05
    assert b["height_source"] == "stated"
    assert b.get("outline_m") and len(b["outline_m"]) >= 5


def test_azadi_agent_massing_at_geo():
    """Azadi coords → parametric arch parts at stated 43 m (not luminance relief)."""
    from shadowhunter.models.landmarks import lookup
    from shadowhunter.models.massing import assemble, infer_kind
    from shadowhunter.models.reconstruct import compose_field

    hit = lookup(35.69974, 51.33810)
    assert hit is not None
    assert abs(float(hit["height_m"]) - 43.0) < 0.01
    kind, src = infer_kind(hit, {"text": "Azadi Tower inverted Y arch"}, None)
    assert kind == "azadi_arch"
    assert src == "landmark"

    # Detect often crops only the vault (~37×33); plan must expand to operator footprint.
    packed = assemble(
        height_m=43.0, width_m=37.0, depth_m=33.0, x_m=30.0, y_m=-35.0,
        landmark=hit, measured={"height_m": 5.5, "policy": "greedy", "cnn_m": 5.5},
        vlm={"text": "white monument with arch"}, cover=False,
        lat=35.69974, lon=51.33810,
    )
    assert packed is not None
    assert packed["kind"] == "azadi_arch"
    assert float(packed["plan_width_m"]) >= 60.0
    assert float(packed["plan_depth_m"]) >= 48.0
    roles = {p.get("role") for p in packed["parts"]}
    assert "pier_west" in roles and "pier_east" in roles and "vault" in roles
    xs = [float(p["x_m"]) for p in packed["parts"]]
    assert max(xs) - min(xs) >= 20.0  # piers visibly separated
    tops = [float(p["z_m"]) + float(p["height_m"]) * 0.5 for p in packed["parts"]]
    assert max(tops) >= 42.0

    img = np.full((200, 180, 3), 30, np.uint8)
    import cv2
    cv2.rectangle(img, (40, 30), (140, 160), (220, 222, 228), -1)
    field = compose_field(
        img,
        [{"box": [30, 20, 120, 150], "index": 0, "compact": True, "score": 0.6,
          "stated_height_m": 43.0, "lat": 35.69974, "lon": 51.33810,
          "width_m": 37.0, "height_m": 33.0}],
        [{"index": 0, "height_m": 5.5, "policy": "greedy", "cnn_m": 5.5}],
        {"azimuth_deg": 81, "elevation_deg": 30, "gsd_m": 0.24},
        0.24,
        {"landmark": hit, "vlm": {"text": "Azadi Tower"},
         "construct": {"massing": {"kind": "azadi_arch"}, "used_llm": False}},
    )
    b = field["buildings"][0]
    assert field["viable"] is True
    assert b["cover"] is False
    assert abs(b["height_m"] - 43.0) < 0.05
    assert b.get("massing_kind") == "azadi_arch"
    assert len(b.get("parts") or []) >= 4
    assert float(b["width_m"]) >= 60.0


def test_expand_azadi_detect_box_in_survey():
    from shadowhunter.core.geo import SunGeometry
    from shadowhunter.models.survey import _expand_landmark_footprint
    from shadowhunter.models.landmarks import lookup

    hit = lookup(35.69974, 51.33810)
    tiny = {"box": [100, 80, 150, 135], "width_m": 37.0, "height_m": 33.0, "compact": True}
    # 0.24 m/px → 63 m ≈ 262 px
    out = _expand_landmark_footprint(tiny, hit, 0.24, 400, 350)
    assert out["box"][2] * 0.24 >= 55.0
    assert out["box"][3] * 0.24 >= 45.0
    assert out.get("landmark_footprint") is True


def test_suggest_azadi_before_draw(monkeypatch):
    """Without an operator AOI, Azadi coords still propose a landmark box."""
    from shadowhunter.models import survey as survey_mod

    def fake_survey(*_a, **_k):
        raise RuntimeError("network skipped in unit test")

    monkeypatch.setattr(survey_mod, "survey", fake_survey)
    out = survey_mod.suggest_structures(35.69974, 51.33810, span_m=200.0)
    assert out["count"] >= 1
    best = out["best"]
    assert best["kind"] == "landmark"
    assert "Azadi" in best["name"] or best.get("name_fa")
    assert abs(float(best["stated_height_m"]) - 43.0) < 0.01
    west, south, east, north = best["bbox"]
    assert west < 51.33810 < east
    assert south < 35.69974 < north
    assert float(best["width_m"]) >= 60.0


def test_azadi_solar_clock_and_construct(monkeypatch):
    from shadowhunter.core.brain import construct
    from shadowhunter.models.timeline import (
        CAPTURE_HOURS_LOCAL, build_timeline, capture_calendar, capture_slots,
        render_timeline_strip,
    )

    monkeypatch.setattr("shadowhunter.core.brain.llm_up", lambda: False)

    lat, lon = 35.70013, 51.3380
    summer = datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc)
    winter = datetime(2026, 12, 21, 8, 0, tzinfo=timezone.utc)
    slots = capture_slots(lat, lon, summer)
    assert [s["local_hour"] for s in slots] == list(CAPTURE_HOURS_LOCAL)
    assert slots[0]["when"].startswith("2026-06-21T06:30")
    assert all(s["is_daylight"] for s in slots)
    wslots = capture_slots(lat, lon, winter)
    assert any(s["is_daylight"] for s in wslots)
    assert wslots[1]["is_daylight"]  # 12:00 local midwinter Tehran
    cal = capture_calendar(lat, lon, 2026)
    assert [row["season"] for row in cal["seasons"]] == ["spring", "summer", "autumn", "winter"]
    payload = build_timeline(lat, lon, summer)
    assert payload["captures"] and payload["calendar"]
    strip = render_timeline_strip(payload["samples"])
    assert strip.ndim == 3 and float(strip.mean()) > 4.0
    out = construct({
        "scene": {"provider": "esri", "source": "basemap",
                  "center": [51.3380, 35.70013], "name": "Azadi Tower"},
        "sun": {"elevation_deg": 30, "is_daylight": True, "quality": 0.61},
        "count": 1,
    }, timeline=payload, intel={
        "wikipedia": [{"title": "Azadi Tower"}],
        "overpass": {"count": 2},
        "reverse": {"name": "Azadi Tower"},
    }, measures={"items": [{"index": 0, "height_m": 4.3}]}, locale="fa")
    assert abs(float(out["lat"]) - 35.70013) < 1e-4
    assert abs(float(out["lon"]) - 51.3380) < 1e-4
    assert "35.70013" in out["coords"]
    assert out["height_indicative"] is True
    assert "INDICATIVE" in out["build_en"].upper()
    assert "تقریبی" in out["build_fa"] or "indicative" in out["build_fa"].lower()
    assert 10 in out["shadow_hours"] and 16 in out["shadow_hours"]


def test_configured_local_models_and_think_strip():
    from shadowhunter.core import brain
    from shadowhunter.core.config import SETTINGS

    assert SETTINGS.llm_model == "qwen3.5:4b"
    assert SETTINGS.embed_model == "mxbai-embed-large:latest"
    assert SETTINGS.vlm_model == "glm-ocr:latest"
    parsed = brain._extract_json('<think>noise</think>{"verdict":"indicative"}')
    assert parsed is not None and parsed["verdict"] == "indicative"
    ids = {t["id"] for t in brain.list_tools()}
    assert {"vlm_ocr", "embed_intel", "solar_clock"} <= ids
    ranked = brain.rerank_intel(
        {"wikipedia": [{"title": "A"}, {"title": "B"}]}, "",
    )
    assert ranked["wikipedia"][0]["title"] == "A"
