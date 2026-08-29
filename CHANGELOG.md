# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [1.0.0] — 2026-08-29

First public release.

### Added

**Stage 1 — RL zone selection**
- `ShadowHunterEnv`, a Gymnasium environment with an 84×84×4 view plus 9 scalars, and an
  11-action space: eight pans, `GROW`, `SHRINK`, `COMMIT`.
- Three label-free proxy rewards — contrast · isolation (R1), structural purity (R2), and
  azimuth coherence (R3) — with occlusion and truncation penalties.
- Potential-based reward shaping, so credit assignment is dense enough for PPO to converge
  on CPU.
- PPO and DQN training through Stable-Baselines3, with telemetry callbacks and cooperative
  abort.
- `greedy_hunt`, a training-free hill-climbing baseline over the same score and action set —
  the application works on a cold install, and the learned policy has something to beat.

**Stage 2 — height regression**
- A PyTorch CNN over RGB plus the shadow mask, with a physics side-channel carrying θ_sun,
  GSD and shadow length.
- Residual prediction around `h = L·tan θ`, so a partly-trained model degrades to physics
  rather than to noise.
- A heteroscedastic head emitting a calibrated σ alongside each height.

**Fusion**
- Inverse-variance blending of the analytic and learned estimates, weighted by measured zone
  cleanliness. Reports `h ± σ`, storey count and confidence.

**Data**
- `synthesize_scene()`: a physically consistent overhead city where shadows obey
  `h = L·tan θ`, buildings are painted back-to-front along the shadow direction so taller
  neighbours genuinely occlude shorter roofs, and shadows are rendered cooler as well as
  darker.
- GeoTIFF ingestion through rasterio, with sun geometry read from image metadata.
- `scripts/download_data.py` for the free SpaceNet, Sentinel-2 and Inria datasets.

**Service**
- A FastAPI backend: analysis, tile sweep, scene management and upload, background training
  jobs, checkpoint listing and activation, and a `/ws/telemetry` event bus.
- `DeckClient`, the single HTTP client shared by every front-end.
- `supervisor.py`, embedding the API in-process for single-binary launch.

**Interfaces**
- PySide6 instrument deck (flagship) — scene canvas with reticle and search trail, solar
  compass, live gauges, Hunt / Train / Archive stations.
- CustomTkinter field console, Flet portable console, NiceGUI browser observatory, and a
  DearPyGui telemetry scope.
- `Orbital Dusk`: one token file (`templates/tokens.json`) rendered to QSS, CSS variables,
  Flet dicts, CustomTkinter JSON and DearPyGui tuples.

**Project**
- MVT layout enforcing two rules: no view imports `models`, and no view hard-codes a colour.
- SQLite job and analysis history.
- Test suite requiring no dataset download and no GPU.

[Unreleased]: https://github.com/Jarvis-Mi/shadowhunter/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Jarvis-Mi/shadowhunter/releases/tag/v1.0.0
