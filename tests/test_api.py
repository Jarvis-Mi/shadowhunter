"""API contract tests - run against the real app via TestClient, no server."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shadowhunter.views.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "nominal"
    assert "torch" in body and "device" in body


def test_analyze_roundtrip(client):
    r = client.post("/api/analyze", json={
        "scene": {"synthesize": True, "size": 320, "buildings": 8, "seed": 21},
        "policy": "greedy", "max_steps": 20,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["box"]) == 4
    assert body["height"]["fused_m"] > 0
    assert body["overlay_png"]
    assert set(body["breakdown"]) >= {"r1_contrast", "r2_structure", "r3_azimuth", "score"}


def test_analyze_respects_supplied_sun(client):
    r = client.post("/api/analyze", json={
        "scene": {"synthesize": True, "size": 320, "buildings": 6, "seed": 22,
                  "sun": {"azimuth_deg": 200.0, "elevation_deg": 25.0, "gsd_m": 0.5}},
        "policy": "greedy", "max_steps": 16, "return_overlay": False,
    })
    sun = r.json()["scene"]["sun"]
    assert sun["azimuth_deg"] == pytest.approx(200.0)
    assert sun["elevation_deg"] == pytest.approx(25.0)


def test_sweep(client):
    r = client.post("/api/sweep", json={
        "scene": {"synthesize": True, "size": 320, "buildings": 6, "seed": 23},
        "policy": "greedy", "limit": 3,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 3
    assert body["mae_m"] is not None


def test_validation_rejects_impossible_sun(client):
    r = client.post("/api/analyze", json={
        "scene": {"synthesize": True, "sun": {"azimuth_deg": 10, "elevation_deg": 95, "gsd_m": 0.5}},
    })
    assert r.status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/api/train/jobs/does-not-exist").status_code == 404


def test_scene_synthesis_and_listing(client):
    created = client.post("/api/scenes/synthesize", json={
        "synthesize": True, "size": 256, "buildings": 5, "seed": 24, "name": "pytest_tile",
    }).json()
    assert created["buildings"] == 5
    assert created["preview_png"]

    names = [s["name"] for s in client.get("/api/scenes").json()["items"]]
    assert "pytest_tile" in names

    assert client.get("/api/scenes/pytest_tile/preview.png").status_code == 200
    assert client.delete("/api/scenes/pytest_tile").status_code == 200


def test_history_records_the_analysis(client):
    client.post("/api/analyze", json={
        "scene": {"synthesize": True, "size": 256, "buildings": 5, "seed": 25},
        "policy": "greedy", "max_steps": 12, "return_overlay": False,
    })
    items = client.get("/api/history?limit=5").json()["items"]
    assert items and items[0]["height_m"] > 0


def test_openapi_documents_every_router(client):
    paths = client.get("/openapi.json").json()["paths"]
    for route in ("/api/health", "/api/analyze", "/api/sweep", "/api/train/rl",
                  "/api/train/cnn", "/api/train/artifacts", "/api/scenes"):
        assert route in paths
