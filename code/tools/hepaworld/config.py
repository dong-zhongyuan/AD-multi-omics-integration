from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import json

try:
    import yaml  # type: ignore
except Exception:  # 可选依赖
    yaml = None


@dataclass
class Step4Config:
    """Configuration for Step4 dynamics training.

    这里的默认值基本照抄你 notebook 里 Cell 0，
    这样下游 Step5–Step10 不会受影响。
    """

    # Paths
    root: str = "D:/dzy-hepaworld-251214-v3"
    in_h5ad: str = "D:/dzy-hepaworld-251214-v3/output_1/HepaWorld_Master.core_noPerturb.qc_filtered.h5ad"
    out_dir: str = "D:/dzy-hepaworld-251214-v3/output_1/step4_dynamics_ot_ode_run3"

    # Preprocess (HVG -> PCA latent)
    n_top_hvg: int = 3000          # 2000–5000 typical
    pca_dim: int = 50              # ODE state dim
    normalize_target_sum: float = 1e4
    hvg_batch_key: str = "dataset"
    time_key: str = "time_score"

    # Train
    seed: int = 42
    deterministic: bool = True     # 新增：控制 torch 是否 deterministic
    device: str = "cuda"           # "cuda" or "cpu"
    epochs: int = 100
    steps_per_pair_per_epoch: int = 200   # minibatch updates per adjacent time pair
    batch_size: int = 256

    # ODE integrator
    integrator: str = "rk4"        # "euler" or "rk4"
    n_steps_per_interval: int = 4  # RK4/Euler sub-steps per (t1 - t0)

    # OT / Loss
    sinkhorn_iters: int = 50
    ot_epsilon: float = 0.15       # entropic regularization; tune 0.05–0.5
    loss_reg_drift: float = 1e-4   # regularize drift magnitude

    # Optimizer
    lr: float = 2e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0

    # Splits (per-time random split)
    val_frac: float = 0.1

    # Vector field sampling for export
    vf_samples_per_time: int = 800  # export N = vf_samples_per_time * n_times

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_step4_config(path: Optional[str] = None) -> Step4Config:
    """Load Step4Config from a JSON/YAML file.

    If *path* is None, returns the default Step4Config().
    Unknown keys in the config file are silently ignored.
    """
    cfg = Step4Config()
    if path is None:
        return cfg

    path_p = Path(path)
    if not path_p.exists():
        raise FileNotFoundError(f"Config file not found: {path_p}")

    ext = path_p.suffix.lower()
    if ext in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files.")
        with path_p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        with path_p.open("r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping, got: {type(data)}")

    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg
