from __future__ import annotations

import random

import numpy as np

try:
    import torch
except Exception:  # 兼容无 torch 的环境
    torch = None  # type: ignore


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for Python, NumPy and torch.

    If *deterministic* is True and torch is available, enables
    deterministic cuDNN / torch behavior as far as reasonably possible.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        # PyTorch 新版：尽可能使用确定性的实现
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        try:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
