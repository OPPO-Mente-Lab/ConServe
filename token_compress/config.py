# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and str(v).strip() != "" else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and str(v).strip() != "" else default


@dataclass(frozen=True)
class CompressConfig:
    enabled: bool = False
    # ``scatter`` preserves the original sequence length for backward
    # compatibility. ``prune`` removes unselected image placeholders before
    # the LLM forward and is intentionally restricted to the verified FSDP
    # dense-input path.
    mode: str = "scatter"
    # student-only: teacher forwards should leave context empty / enabled=False
    method: str = "spatial_quota_fps"
    keep_ratio: float = 0.20
    spatial_weight: float = 0.25
    quota_grid_rows: int = 4
    quota_grid_cols: int = 4
    quota_min_per_cell: int = 1
    quota_power: float = 0.5
    quota_diversity_weight: float = 1.0
    prototype: str = "centroid"
    seed: int = 0
    # Directory containing ogstb_cluster_merge.py and ogstb_b1_merge.py.
    cluster_code_dir: str = str(Path(__file__).resolve().parents[1] / "compression")


def load_config_from_env() -> CompressConfig:
    code = os.environ.get("TOKEN_COMPRESS_CLUSTER_CODE_DIR") or CompressConfig.cluster_code_dir
    return CompressConfig(
        enabled=_env_bool("TOKEN_COMPRESS", False),
        mode=os.environ.get("TOKEN_COMPRESS_MODE", "scatter").strip().lower(),
        method=os.environ.get("TOKEN_COMPRESS_METHOD", "spatial_quota_fps"),
        keep_ratio=_env_float("TOKEN_COMPRESS_KEEP_RATIO", 0.20),
        spatial_weight=_env_float("TOKEN_COMPRESS_SPATIAL_WEIGHT", 0.25),
        quota_grid_rows=_env_int("TOKEN_COMPRESS_GRID_ROWS", 4),
        quota_grid_cols=_env_int("TOKEN_COMPRESS_GRID_COLS", 4),
        quota_min_per_cell=_env_int("TOKEN_COMPRESS_MIN_PER_CELL", 1),
        quota_power=_env_float("TOKEN_COMPRESS_QUOTA_POWER", 0.5),
        quota_diversity_weight=_env_float("TOKEN_COMPRESS_DIVERSITY", 1.0),
        prototype=os.environ.get("TOKEN_COMPRESS_PROTOTYPE", "centroid"),
        seed=_env_int("TOKEN_COMPRESS_SEED", 0),
        cluster_code_dir=code,
    )
