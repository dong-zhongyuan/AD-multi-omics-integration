from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, List

try:
    import torch
except Exception:  # optional
    torch = None  # type: ignore


def log(msg: str) -> None:
    """Simple stdout logger with flush (keeps behavior of the notebook)."""
    print(msg, flush=True)


def sha1_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Compute SHA1 hash of a file in a streaming fashion."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest_step4(cfg: Any, out_dir: Path) -> None:
    """Write a simple manifest for Step4 (env + input hash).

    *cfg* is expected to have at least ``in_h5ad`` and ``device`` attributes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("=== Step4 Manifest ===")
    lines.append(time.strftime("time_local=%Y-%m-%d %H:%M:%S"))
    lines.append(f"python={sys.version.replace(os.linesep, ' ')}")

    if torch is not None:
        lines.append(
            f"torch={torch.__version__} "
            f"cuda_available={torch.cuda.is_available()} "
            f"device={getattr(cfg, 'device', 'unknown')}"
        )
    lines.append("")

    in_path = Path(getattr(cfg, "in_h5ad", ""))
    if in_path:
        if in_path.exists():
            lines.append(f"in_h5ad={in_path}")
            try:
                lines.append(f"in_h5ad_sha1={sha1_file(in_path)}")
            except Exception as e:  # 极小概率异常
                lines.append(f"in_h5ad_sha1=ERROR({e})")
        else:
            lines.append(f"in_h5ad={in_path} (MISSING)")
    lines.append("")

    txt_path = out_dir / "manifest_step4.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log(f"[MANIFEST] wrote manifest -> {txt_path}")
