"""Runtime configuration: YAML file <- environment <- defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
WORKSPACE_DIR = DATA_DIR / "workspace"


@dataclass
class EnvConfig:
    """Geometry of the Shadow Hunter gym environment."""
    tile_size: int = 512          # side of the working scene in pixels
    crop_min: int = 32            # smallest window the agent may shrink to
    crop_max: int = 224           # largest window the agent may grow to
    crop_init: int = 96
    obs_size: int = 84            # everything is resampled to this for the policy
    step_px: int = 8              # translation quantum
    scale_px: int = 8             # grow/shrink quantum
    max_steps: int = 48           # episode budget
    commit_bonus: float = 1.5     # payoff for stopping on a good zone
    step_cost: float = 0.01       # pressure against wandering forever


@dataclass
class RewardConfig:
    """Weights of the composite proxy reward (no ground-truth height needed)."""
    w_contrast: float = 1.0       # R1 - shadow/sunlit separation + isolation
    w_structure: float = 0.8      # R2 - edge coherence over entropy
    w_azimuth: float = 0.9        # R3 - alignment with solar azimuth
    w_occlusion: float = 1.2      # penalty for neighbouring-shadow contamination
    w_truncation: float = 0.8     # penalty for a shadow clipped by the crop border
    shaping: bool = True          # reward = delta of score (potential-based)


@dataclass
class CNNConfig:
    in_channels: int = 4          # R, G, B, shadow-mask
    width: int = 32
    dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 20


@dataclass
class RLConfig:
    algo: str = "PPO"             # PPO | DQN
    total_timesteps: int = 20_000
    n_steps: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.98
    seed: int = 42
    device: str = "auto"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8077
    reload: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class Settings:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    data_dir: Path = DATA_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    cache_dir: Path = CACHE_DIR
    workspace_dir: Path = WORKSPACE_DIR
    llm_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen3.5:4b"
    llm_key: str = ""
    embed_model: str = "mxbai-embed-large:latest"
    vlm_model: str = "glm-ocr:latest"

    @property
    def api_base(self) -> str:
        return f"http://{self.server.host}:{self.server.port}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["checkpoint_dir"] = str(self.checkpoint_dir)
        d["cache_dir"] = str(self.cache_dir)
        d["workspace_dir"] = str(self.workspace_dir)
        return d


def _apply(section: Any, values: dict[str, Any]) -> None:
    for k, v in (values or {}).items():
        if hasattr(section, k):
            setattr(section, k, v)


def load_settings(path: str | Path | None = None) -> Settings:
    s = Settings()
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    if cfg_path.exists():
        try:
            import yaml  # optional dependency

            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            for name in ("env", "reward", "cnn", "rl", "server"):
                _apply(getattr(s, name), raw.get(name, {}))
            llm = raw.get("llm") or {}
            if isinstance(llm, dict):
                s.llm_url = str(llm.get("url") or s.llm_url)
                s.llm_model = str(llm.get("model") or s.llm_model)
                s.embed_model = str(llm.get("embed_model") or s.embed_model)
                s.vlm_model = str(llm.get("vlm_model") or s.vlm_model)
                if llm.get("key") is not None:
                    s.llm_key = str(llm.get("key") or "")
        except ImportError:
            pass

    # Environment overrides win last - handy for containers and CI.
    s.server.host = os.getenv("SH_HOST", s.server.host)
    s.server.port = int(os.getenv("SH_PORT", s.server.port))
    s.rl.device = os.getenv("SH_DEVICE", s.rl.device)
    s.llm_url = os.getenv("SH_LLM_URL", s.llm_url)
    s.llm_model = os.getenv("SH_LLM_MODEL", s.llm_model)
    s.llm_key = os.getenv("SH_LLM_KEY", s.llm_key)
    s.embed_model = os.getenv("SH_EMBED_MODEL", s.embed_model)
    s.vlm_model = os.getenv("SH_VLM_MODEL", s.vlm_model)

    for d in (s.data_dir, s.checkpoint_dir, SYNTHETIC_DIR, RAW_DIR,
              s.cache_dir, s.workspace_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    return s


SETTINGS = load_settings()
