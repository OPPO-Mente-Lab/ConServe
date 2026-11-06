# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .config import CompressConfig, load_config_from_env
from .context import compress_context


@dataclass(frozen=True)
class PrunedImageEmbeds:
    """Selected merger outputs for one image, ordered in original raster order."""

    embeddings: torch.Tensor
    selected_indices: torch.Tensor
    foreground_count: int
    background_count: int


@lru_cache(maxsize=1)
def _load_cluster_mod(code_dir: str):
    code_path = Path(code_dir)
    if not code_path.is_dir():
        raise FileNotFoundError(
            f"Cluster code directory not found: {code_dir}. "
            "Set TOKEN_COMPRESS_CLUSTER_CODE_DIR to the local compression directory."
        )
    parent = str(code_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import ogstb_b1_merge as b1  # noqa: WPS433
    import ogstb_cluster_merge as cluster  # noqa: WPS433

    return b1, cluster


def _merge_size_patch(visual) -> tuple[int, int]:
    merge_size = int(getattr(visual, "spatial_merge_size", 2) or 2)
    patch_size = int(getattr(visual, "patch_size", 14) or 14)
    return merge_size, patch_size


def _scatter_centroids(
    embeddings: torch.Tensor,
    fg_mask: np.ndarray,
    assignments: torch.Tensor,
    representatives: torch.Tensor,
    background_indices: torch.Tensor,
) -> torch.Tensor:
    """Length-preserving: replace each bg token with its cluster prototype."""
    out = embeddings.clone()
    bg_idx = background_indices.to(device=embeddings.device, dtype=torch.long)
    assign = assignments.to(device=embeddings.device, dtype=torch.long)
    reps = representatives.to(device=embeddings.device, dtype=embeddings.dtype)
    if bg_idx.numel() == 0:
        return out
    out[bg_idx] = reps[assign]
    # keep fg untouched (already in out)
    fg = torch.as_tensor(fg_mask, device=embeddings.device, dtype=torch.bool).reshape(-1)
    if fg.numel() == embeddings.shape[0]:
        out[fg] = embeddings[fg]
    return out


def compress_one_image_embeds(
    embeddings: torch.Tensor,
    *,
    grid_thw: Sequence[int],
    boxes_xyxy: list[tuple[int, int, int, int]],
    orig_hw: tuple[int, int],
    visual: Any,
    cfg: Optional[CompressConfig] = None,
) -> torch.Tensor:
    """Apply spatial_quota_fps then scatter prototypes back to length N."""
    cfg = cfg or load_config_from_env()
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be (N,D), got {tuple(embeddings.shape)}")
    if not boxes_xyxy:
        return embeddings

    b1, cluster = _load_cluster_mod(cfg.cluster_code_dir)
    merge_size, patch_size = _merge_size_patch(visual)
    grid = [int(x) for x in grid_thw]
    n = int(embeddings.shape[0])
    gh, gw = max(1, grid[1] // merge_size), max(1, grid[2] // merge_size)
    resized_h, resized_w = max(1, grid[1] * patch_size), max(1, grid[2] * patch_size)
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])

    fg = b1.bbox_mask_on_grid(
        boxes_xyxy,
        resized_h=resized_h,
        resized_w=resized_w,
        grid_h=gh,
        grid_w=gw,
        orig_h=orig_h,
        orig_w=orig_w,
        merge_size=merge_size,
        patch_size=patch_size,
    )
    if fg.shape[0] != n:
        fg = fg[:n] if fg.shape[0] > n else np.pad(fg, (0, n - fg.shape[0]))
    coords = cluster.grid_coordinates(gh, gw).to(embeddings.device)
    if coords.shape[0] != n:
        coords = torch.stack(
            (torch.zeros(n, device=embeddings.device), torch.linspace(0, 1, n, device=embeddings.device)),
            dim=1,
        )

    compressed = cluster.compress_background_tokens(
        embeddings,
        fg,
        coords=coords,
        method=cfg.method,
        keep_ratio=cfg.keep_ratio,
        spatial_weight=cfg.spatial_weight,
        prototype=cfg.prototype,
        quota_grid_rows=cfg.quota_grid_rows,
        quota_grid_cols=cfg.quota_grid_cols,
        quota_min_per_cell=cfg.quota_min_per_cell,
        quota_power=cfg.quota_power,
        quota_diversity_weight=cfg.quota_diversity_weight,
        seed=cfg.seed,
    )
    return _scatter_centroids(
        embeddings,
        fg,
        compressed["assignments"],
        compressed["bg_representatives"],
        compressed["background_indices"],
    )


def select_one_image_embeds(
    embeddings: torch.Tensor,
    *,
    grid_thw: Sequence[int],
    boxes_xyxy: list[tuple[int, int, int, int]],
    orig_hw: tuple[int, int],
    visual: Any,
    cfg: Optional[CompressConfig] = None,
) -> PrunedImageEmbeds:
    """Return real merger tokens selected for a physical LLM-sequence prune.

    OCR foreground tokens are kept exactly. Background representatives are the
    original tokens selected by spatial-quota FPS, not centroid vectors. The
    final sort restores raster order so their inherited M-RoPE coordinates and
    placeholder order remain aligned.
    """
    cfg = cfg or load_config_from_env()
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be (N,D), got {tuple(embeddings.shape)}")

    n = int(embeddings.shape[0])
    all_indices = torch.arange(n, device=embeddings.device, dtype=torch.long)
    if not boxes_xyxy:
        return PrunedImageEmbeds(embeddings, all_indices, foreground_count=0, background_count=n)

    b1, cluster = _load_cluster_mod(cfg.cluster_code_dir)
    merge_size, patch_size = _merge_size_patch(visual)
    grid = [int(x) for x in grid_thw]
    gh, gw = max(1, grid[1] // merge_size), max(1, grid[2] // merge_size)
    resized_h, resized_w = max(1, grid[1] * patch_size), max(1, grid[2] * patch_size)
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])

    fg = b1.bbox_mask_on_grid(
        boxes_xyxy,
        resized_h=resized_h,
        resized_w=resized_w,
        grid_h=gh,
        grid_w=gw,
        orig_h=orig_h,
        orig_w=orig_w,
        merge_size=merge_size,
        patch_size=patch_size,
    )
    if fg.shape[0] != n:
        fg = fg[:n] if fg.shape[0] > n else np.pad(fg, (0, n - fg.shape[0]))
    fg_tensor = torch.as_tensor(fg, device=embeddings.device, dtype=torch.bool).reshape(-1)
    coords = cluster.grid_coordinates(gh, gw).to(embeddings.device)
    if coords.shape[0] != n:
        raise ValueError(
            f"merger grid {gh}x{gw} yields {coords.shape[0]} positions, but image has {n} embeddings"
        )

    compressed = cluster.compress_background_tokens(
        embeddings,
        fg,
        coords=coords,
        method=cfg.method,
        keep_ratio=cfg.keep_ratio,
        spatial_weight=cfg.spatial_weight,
        # Pruning must retain true encoder outputs. ``representative_source_idx``
        # is also returned for FPS, so this stays valid if a different supported
        # clustering method is selected later.
        prototype="medoid",
        quota_grid_rows=cfg.quota_grid_rows,
        quota_grid_cols=cfg.quota_grid_cols,
        quota_min_per_cell=cfg.quota_min_per_cell,
        quota_power=cfg.quota_power,
        quota_diversity_weight=cfg.quota_diversity_weight,
        seed=cfg.seed,
    )
    bg_selected = compressed["representative_source_idx"].to(embeddings.device, dtype=torch.long)
    selected = torch.cat((torch.nonzero(fg_tensor, as_tuple=False).flatten(), bg_selected)).unique(sorted=True)
    return PrunedImageEmbeds(
        embeddings=embeddings[selected],
        selected_indices=selected,
        foreground_count=int(fg_tensor.sum().item()),
        background_count=int(bg_selected.numel()),
    )


def compress_image_embeds_scatter(
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    visual: Any,
    cfg: Optional[CompressConfig] = None,
) -> torch.Tensor:
    """Compress a packed batch of image embeds using CompressContext boxes.

    If context is missing/disabled or boxes absent, returns embeds unchanged.
    """
    ctx = compress_context()
    cfg = cfg or (ctx.config if ctx is not None else load_config_from_env())
    if not cfg.enabled or ctx is None:
        return image_embeds
    if image_grid_thw is None or image_embeds is None:
        return image_embeds

    merge_size, _ = _merge_size_patch(visual)
    parts: list[torch.Tensor] = []
    offset = 0
    n_img = int(image_grid_thw.shape[0])
    for i in range(n_img):
        thw = image_grid_thw[i].tolist()
        n_i = int(thw[0] * thw[1] * thw[2] // (merge_size * merge_size))
        # Qwen3.5 pooler already merges spatially; n_i should match chunk size.
        # Fall back to equal split if prod formula mismatches.
        remaining = int(image_embeds.shape[0] - offset)
        if n_i <= 0 or n_i > remaining:
            n_i = remaining // max(1, n_img - i)
        emb_i = image_embeds[offset : offset + n_i]
        offset += n_i

        boxes = ctx.boxes_xyxy[i] if i < len(ctx.boxes_xyxy) else []
        hw = ctx.orig_hw[i] if i < len(ctx.orig_hw) else (1, 1)
        if boxes:
            emb_i = compress_one_image_embeds(
                emb_i,
                grid_thw=thw,
                boxes_xyxy=boxes,
                orig_hw=hw,
                visual=visual,
                cfg=cfg,
            )
        parts.append(emb_i)

    if offset != int(image_embeds.shape[0]):
        # safety: if split drifted, do not corrupt training
        return image_embeds
    return torch.cat(parts, dim=0)
