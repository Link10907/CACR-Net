from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - optional at parse time
    yaml = None


@dataclass
class DatasetConfig:
    root: str = "data/Teeth3DS+"
    points_per_tooth: int = 2048       # paper: 2048 uniformly sampled points per tooth
    teeth_per_jaw: int = 16
    include_normals: bool = True
    normal_radius: float = 0.1
    normal_max_nn: int = 30


@dataclass
class Stage1Config:
    batch_size: int = 16               # paper: batch size of 16
    epochs: int = 300
    learning_rate: float = 1e-4        # paper: initial learning rate 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    point_dim: int = 6                 # (x, y, z, nx, ny, nz)
    hidden_dim: int = 128
    global_dim: int = 512
    voxel_resolution: int = 64         # paper: 64^3 voxel grid
    k_neighbors: int = 32             # paper: k = 32
    low_points: int = 512              # n/4 = 2048/4
    mid_points: int = 1024             # n/2 = 2048/2
    high_points: int = 2048            # n = 2048
    alpha_low: float = 0.2            # paper: alpha_L = 0.2
    alpha_mid: float = 0.4            # paper: alpha_M = 0.4
    lambda_cd: float = 0.5            # paper: lambda_cd = 0.5
    lambda_cgoc: float = 0.3          # paper: lambda_cgoc = 0.3
    lambda_sdf: float = 0.2           # paper: lambda_sdf = 0.2
    cgoc_top_ratio: float = 0.3       # paper: top-T = 30%
    cgoc_temperature: float = 0.1     # paper: tau = 0.1
    schemes: tuple[str, ...] = ("hilbert", "z_order", "raster", "zigzag")


@dataclass
class Stage2Config:
    batch_size: int = 16               # paper: batch size of 16
    epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    latent_dim: int = 512              # paper: VAE latent dimension 512
    latent_tokens: int = 16
    token_dim: int = 32                # 16 * 32 = 512
    condition_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 6
    vae_kl_weight: float = 1e-5       # paper: beta = 1e-5
    num_query_points: int = 4096
    grid_resolution: int = 128         # paper: Marching Cubes on 128^3
    diffusion_steps: int = 800         # paper: T = 800
    beta_start: float = 1e-4
    beta_end: float = 0.02


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
    log_interval: int = 20
    checkpoint_interval: int = 10
    mixed_precision: bool = False


@dataclass
class ProjectConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_config(path: Optional[str] = None) -> ProjectConfig:
    cfg = ProjectConfig()
    if not path:
        return cfg
    if yaml is None:
        raise ImportError("PyYAML is required to load config files.")

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    merged = _deep_update(cfg.to_dict(), data)
    return ProjectConfig(
        dataset=DatasetConfig(**merged["dataset"]),
        stage1=Stage1Config(**merged["stage1"]),
        stage2=Stage2Config(**merged["stage2"]),
        runtime=RuntimeConfig(**merged["runtime"]),
    )
