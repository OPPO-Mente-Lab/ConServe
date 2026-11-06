#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Deployable background-token clustering for M3 multi-box translation.

The target-box tokens are always preserved.  Only background tokens are
compressed.  The module deliberately has no dependency on scikit-learn so it
can run in the existing PyTorch inference environment.

The low-level :func:`compress_background_tokens` function is model agnostic.
The high-level :func:`encode_merge_batch_cluster` adapter follows the return
contract used by ``ogstb_b1_merge.encode_merge_batch``:

``{merged_embeds, merged_grid_thw, meta}``.

Arbitrary clusters are not a regular 2-D grid.  For current vLLM integration
we use the same approximate ``grid_thw`` packing as B-1/B-5/B-8, while keeping
the true centroid coordinates in the debug output for a future B-4-style
explicit M-RoPE path.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ogstb_b1_merge import (
    _resolve_inner_vlm,
    align_merged_pack_for_vllm,
    bbox_mask_on_grid,
    compute_merged_grid_thw,
    extract_visual_embeddings_hf,
)
from ogstb_b3_grid_merge import _dilate_fg_2d


SUPPORTED_METHODS = (
    "kmeans",
    "spatial_kmeans",
    "minibatch_kmeans",
    "local_kmeans",
    "fps",
    "spatial_quota_fps",
    "tome",
    "mean",
    "weighted",
    "random",
)


def resize_image_for_vllm_budget(
    pil_image: Any,
    *,
    max_pixels: int | None,
    min_pixels: int | None,
    patch_size: int,
    merge_size: int,
) -> tuple[Any, tuple[int, int]]:
    """Apply Qwen smart resize before a processor that ignores its limits."""
    width, height = (int(value) for value in pil_image.size)
    if max_pixels is None and min_pixels is None:
        return pil_image, (width, height)
    factor = max(1, int(patch_size) * int(merge_size))
    min_px = max(1, int(min_pixels) if min_pixels is not None else 0)
    max_px = max(1, int(max_pixels) if max_pixels is not None else 2**62)
    if min_px > max_px:
        raise ValueError(f"min_pixels={min_px} exceeds max_pixels={max_px}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_px:
        beta = math.sqrt((height * width) / max_px)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif min_px > 0 and h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    if (w_bar, h_bar) == (width, height):
        return pil_image, (width, height)
    from PIL import Image
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return pil_image.resize((w_bar, h_bar), resample=resample), (w_bar, h_bar)


def resize_image_to_visual_token_target(
    pil_image: Any,
    *,
    target_tokens: int,
    patch_size: int,
    merge_size: int,
) -> tuple[Any, tuple[int, int]]:
    """Resize to an exact merger-token count with minimal aspect distortion."""
    target = int(target_tokens)
    if target < 1:
        raise ValueError(f"target_tokens must be >= 1, got {target_tokens}")
    width, height = (int(value) for value in pil_image.size)
    original_ratio = height / max(width, 1)
    candidates: list[tuple[float, int, int]] = []
    for grid_h in range(1, int(math.isqrt(target)) + 1):
        if target % grid_h:
            continue
        grid_w = target // grid_h
        # Merger-grid dimensions must multiply to target exactly.
        candidates.append((abs(math.log((grid_h / grid_w) / original_ratio)), grid_h, grid_w))
        if grid_h != grid_w:
            candidates.append((abs(math.log((grid_w / grid_h) / original_ratio)), grid_w, grid_h))
    if not candidates:
        raise RuntimeError(f"no factorization found for target_tokens={target}")
    _, grid_h, grid_w = min(candidates, key=lambda row: row[0])
    factor = max(1, int(patch_size) * int(merge_size))
    out_w, out_h = grid_w * factor, grid_h * factor
    if (out_w, out_h) == (width, height):
        return pil_image, (out_w, out_h)
    from PIL import Image
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return pil_image.resize((out_w, out_h), resample=resample), (out_w, out_h)


def grid_coordinates(grid_h: int, grid_w: int) -> torch.Tensor:
    """Return normalized merger-grid coordinates in raster order, ``(N,2)``."""
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h/grid_w must be positive")
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, grid_h),
        torch.linspace(0.0, 1.0, grid_w),
        indexing="ij",
    )
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


def _resolve_budget(n_bg: int, keep_ratio: float, n_clusters: int | None) -> int:
    if n_bg <= 0:
        return 0
    if n_clusters is not None:
        return max(1, min(int(n_clusters), n_bg))
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError(f"keep_ratio must be in (0,1], got {keep_ratio}")
    return max(1, min(n_bg, int(round(n_bg * float(keep_ratio)))))


def _semantic_features(x: torch.Tensor, metric: str) -> torch.Tensor:
    x = x.float()
    if metric == "cosine":
        return F.normalize(x, dim=-1, eps=1e-6)
    if metric != "euclidean":
        raise ValueError(f"unknown metric={metric!r}")
    # Per-image standardization keeps large embedding dimensions from
    # dominating the optional two spatial dimensions by raw scale alone.
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-5)
    return (x - x.mean(dim=0, keepdim=True)) / scale


def _clustering_features(
    embeddings: torch.Tensor,
    coords: torch.Tensor,
    *,
    metric: str,
    spatial_weight: float,
) -> torch.Tensor:
    semantic = _semantic_features(embeddings, metric)
    if spatial_weight <= 0:
        return semantic
    centered_xy = (coords.float() - 0.5) * 2.0
    return torch.cat((semantic, centered_xy * float(spatial_weight)), dim=-1)


def _kmeans_plus_plus(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Deterministic k-means++ initialization on the input device."""
    n = int(x.shape[0])
    if k >= n:
        return x.clone()
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=gen, device=x.device).item())
    chosen = [first]
    min_dist = ((x - x[first]) ** 2).sum(dim=1)
    for _ in range(1, k):
        total = min_dist.sum()
        if not torch.isfinite(total) or float(total.item()) <= 1e-12:
            # All remaining points are identical. Pick the first unused index.
            idx = next(i for i in range(n) if i not in set(chosen))
        else:
            idx = int(torch.multinomial(min_dist / total, 1, generator=gen).item())
            if idx in chosen:
                idx = next(i for i in range(n) if i not in set(chosen))
        chosen.append(idx)
        dist = ((x - x[idx]) ** 2).sum(dim=1)
        min_dist = torch.minimum(min_dist, dist)
    return x[torch.as_tensor(chosen, device=x.device, dtype=torch.long)].clone()


def _repair_empty_clusters(
    x: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guarantee exactly k non-empty clusters without duplicating medoids."""
    assignments = assignments.clone()
    counts = torch.bincount(assignments, minlength=k)
    empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if empty.numel():
        current_dist = ((x.float() - centers[assignments].float()) ** 2).sum(dim=1)
        candidates = torch.argsort(current_dist, descending=True).tolist()
        used: set[int] = set()
        for cid in empty.tolist():
            chosen = next(
                idx for idx in candidates
                if idx not in used and int(counts[int(assignments[idx])].item()) > 1
            )
            donor = int(assignments[chosen].item())
            counts[donor] -= 1
            assignments[chosen] = cid
            counts[cid] += 1
            used.add(chosen)
    sums = torch.zeros(k, x.shape[1], device=x.device, dtype=torch.float32)
    sums.index_add_(0, assignments, x.float())
    counts_f = torch.bincount(assignments, minlength=k).float().clamp_min(1)
    return sums / counts_f[:, None], assignments


def torch_kmeans(
    x: torch.Tensor,
    k: int,
    *,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Lloyd k-means with k-means++ initialization.

    Returns ``centers, assignments, stats``.  Empty clusters are repaired with
    currently worst represented points, which makes high compression runs
    stable without silently reducing the requested budget.
    """
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"x must be a non-empty (N,D) tensor, got {tuple(x.shape)}")
    k = max(1, min(int(k), int(x.shape[0])))
    centers = _kmeans_plus_plus(x.float(), k, seed)
    assignments = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    inertia = float("inf")
    iterations = 0
    for iterations in range(1, max(1, int(max_iter)) + 1):
        dist = torch.cdist(x.float(), centers, p=2) ** 2
        min_dist, assignments = dist.min(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, x.float())
        counts = torch.bincount(assignments, minlength=k).to(x.dtype)
        new_centers = sums / counts.clamp_min(1).unsqueeze(1)
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
        if empty.numel():
            worst = torch.argsort(min_dist, descending=True)
            used: set[int] = set()
            for cid, point_idx in zip(empty.tolist(), worst.tolist()):
                if point_idx in used:
                    continue
                new_centers[cid] = x[point_idx]
                used.add(point_idx)
        shift = torch.linalg.vector_norm(new_centers - centers, dim=1).max()
        centers = new_centers
        inertia = float(min_dist.mean().item())
        if float(shift.item()) <= float(tol):
            break
    # Final assignment must correspond to the returned centers.
    final_dist = torch.cdist(x.float(), centers, p=2) ** 2
    _, assignments = final_dist.min(dim=1)
    centers, assignments = _repair_empty_clusters(x, centers, assignments, k)
    min_dist = ((x.float() - centers[assignments]) ** 2).sum(dim=1)
    inertia = float(min_dist.mean().item())
    return centers, assignments, {"iterations": iterations, "inertia": inertia}


def torch_minibatch_kmeans(
    x: torch.Tensor,
    k: int,
    *,
    max_iter: int = 40,
    batch_size: int = 256,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Small, deterministic MiniBatch K-Means implementation.

    It is mainly useful when resolution sweeps create several thousand
    background tokens.  A final full assignment/recompute removes most of the
    stochastic centroid bias while retaining cheap iterative updates.
    """
    x = x.float()
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    centers = _kmeans_plus_plus(x, k, seed)
    counts = torch.zeros(k, device=x.device, dtype=torch.float32)
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed) + 1)
    bs = max(k, min(int(batch_size), n))
    iterations = max(1, int(max_iter))
    for _ in range(iterations):
        idx = torch.randperm(n, generator=gen, device=x.device)[:bs]
        batch = x[idx]
        assign = torch.cdist(batch, centers).argmin(dim=1)
        batch_sums = torch.zeros_like(centers)
        batch_sums.index_add_(0, assign, batch)
        batch_counts = torch.bincount(assign, minlength=k).float()
        total = counts + batch_counts
        active = batch_counts > 0
        centers[active] = (
            centers[active] * counts[active, None] + batch_sums[active]
        ) / total[active, None].clamp_min(1.0)
        counts = total
    dist = torch.cdist(x, centers) ** 2
    _, assignments = dist.min(dim=1)
    centers, assignments = _repair_empty_clusters(x, centers, assignments, k)
    min_dist = ((x - centers[assignments]) ** 2).sum(dim=1)
    return centers, assignments, {
        "iterations": iterations,
        "inertia": float(min_dist.mean().item()),
        "batch_size": bs,
    }


def _farthest_point_indices(x: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Greedy k-center/FPS indices with deterministic seeded initialization."""
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must be a non-empty (N,D) tensor")
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    if k == n:
        return torch.arange(n, device=x.device)
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=gen, device=x.device).item())
    selected = torch.empty(k, dtype=torch.long, device=x.device)
    selected[0] = first
    min_dist = ((x.float() - x[first].float()) ** 2).sum(dim=1)
    min_dist[first] = -1.0
    for pos in range(1, k):
        idx = int(min_dist.argmax().item())
        selected[pos] = idx
        dist = ((x.float() - x[idx].float()) ** 2).sum(dim=1)
        min_dist = torch.minimum(min_dist, dist)
        min_dist[selected[:pos + 1]] = -1.0
    return selected


def torch_fps(
    x: torch.Tensor,
    k: int,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Max-min diversity selection followed by nearest-center assignment."""
    selected = _farthest_point_indices(x, k, seed)
    centers = x[selected].float()
    dist = torch.cdist(x.float(), centers) ** 2
    min_dist, assignments = dist.min(dim=1)
    return centers, assignments, selected, {
        "iterations": int(selected.numel()),
        "inertia": float(min_dist.mean().item()),
    }


def _spatial_cell_ids(coords: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Map normalized ``(y,x)`` coordinates to regular spatial cells."""
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    y = torch.floor(coords[:, 0].float().clamp(0, 1 - 1e-7) * rows).long()
    x = torch.floor(coords[:, 1].float().clamp(0, 1 - 1e-7) * cols).long()
    return y * cols + x


def _largest_remainder_quotas(
    capacities: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    min_per_cell: int,
) -> torch.Tensor:
    """Allocate exactly ``k`` slots with per-cell floors and capacities."""
    capacities = capacities.long().clamp_min(0)
    active_idx = torch.nonzero(capacities > 0, as_tuple=False).flatten()
    quotas = torch.zeros_like(capacities)
    if active_idx.numel() == 0:
        return quotas
    k = max(1, min(int(k), int(capacities.sum().item())))
    floor = max(0, int(min_per_cell))
    base = torch.minimum(capacities[active_idx], torch.full_like(active_idx, floor))
    if int(base.sum().item()) > k:
        order = active_idx[torch.argsort(scores[active_idx], descending=True)]
        quotas[order[:k]] = 1
        return quotas
    quotas[active_idx] = base
    remaining = k - int(quotas.sum().item())
    while remaining > 0:
        spare = capacities - quotas
        eligible = spare > 0
        if not bool(eligible.any()):
            break
        weight = scores.float().clamp_min(0) * eligible.float()
        if float(weight.sum().item()) <= 1e-12:
            weight = spare.float()
        ideal = weight / weight.sum().clamp_min(1e-12) * remaining
        add = torch.minimum(torch.floor(ideal).long(), spare)
        if int(add.sum().item()) == 0:
            fractional = ideal - torch.floor(ideal)
            order = torch.argsort(fractional, descending=True)
            for cell in order.tolist():
                if remaining <= 0:
                    break
                if int(spare[cell].item()) > 0:
                    quotas[cell] += 1
                    remaining -= 1
            continue
        quotas += add
        remaining -= int(add.sum().item())
    return quotas


def torch_spatial_quota_fps(
    embeddings: torch.Tensor,
    coords: torch.Tensor,
    k: int,
    *,
    metric: str = "cosine",
    spatial_weight: float = 0.25,
    grid_rows: int = 4,
    grid_cols: int = 4,
    min_per_cell: int = 1,
    quota_power: float = 0.5,
    diversity_weight: float = 1.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Spatially constrained FPS with exact adaptive crop quotas.

    Every affordable non-empty cell gets a minimum quota. Remaining slots are
    distributed by cell mass and semantic diversity. FPS selection and token
    assignment stay inside each cell, preventing cross-layout aggregation.
    """
    if embeddings.ndim != 2 or coords.shape != (embeddings.shape[0], 2):
        raise ValueError("embeddings must be (N,D) and coords must be (N,2)")
    n = int(embeddings.shape[0])
    k = max(1, min(int(k), n))
    grid_rows, grid_cols = max(1, int(grid_rows)), max(1, int(grid_cols))
    n_cells = grid_rows * grid_cols
    cell_ids = _spatial_cell_ids(coords, grid_rows, grid_cols)
    capacities = torch.bincount(cell_ids, minlength=n_cells)
    semantic = _semantic_features(embeddings, metric)
    diversity = torch.zeros(n_cells, device=embeddings.device, dtype=torch.float32)
    for cell in torch.nonzero(capacities > 0, as_tuple=False).flatten().tolist():
        idx = torch.nonzero(cell_ids == cell, as_tuple=False).flatten()
        local = semantic[idx]
        center = local.mean(dim=0, keepdim=True)
        diversity[cell] = ((local - center) ** 2).sum(dim=1).mean()
    active = capacities > 0
    diversity_scaled = torch.zeros_like(diversity)
    if bool(active.any()):
        d = diversity[active]
        diversity_scaled[active] = (d - d.min()) / (d.max() - d.min()).clamp_min(1e-6)
    scores = capacities.float().pow(max(0.0, float(quota_power)))
    scores *= 1.0 + max(0.0, float(diversity_weight)) * diversity_scaled
    quotas = _largest_remainder_quotas(capacities, scores, k, min_per_cell)

    # Cells with zero quota still need a valid cluster assignment below.  They
    # occur whenever the requested budget is smaller than the number of active
    # spatial cells; leaving them uninitialized later corrupts bincount.
    assignments = torch.full((n,), -1, dtype=torch.long, device=embeddings.device)
    selected_parts: list[torch.Tensor] = []
    center_parts: list[torch.Tensor] = []
    cluster_offset = 0
    inertia_sum = 0.0
    for cell in torch.nonzero(quotas > 0, as_tuple=False).flatten().tolist():
        idx = torch.nonzero(cell_ids == cell, as_tuple=False).flatten()
        local_k = int(quotas[cell].item())
        local_feat = _clustering_features(
            embeddings[idx], coords[idx], metric=metric, spatial_weight=spatial_weight,
        )
        local_selected = _farthest_point_indices(
            local_feat, local_k, seed=int(seed) + 1_000_003 * int(cell),
        )
        local_centers = local_feat[local_selected]
        local_dist = torch.cdist(local_feat.float(), local_centers.float()) ** 2
        min_dist, local_assign = local_dist.min(dim=1)
        assignments[idx] = local_assign + cluster_offset
        selected_parts.append(idx[local_selected])
        center_parts.append(local_centers)
        inertia_sum += float(min_dist.sum().item())
        cluster_offset += local_k
    selected = torch.cat(selected_parts)
    centers = torch.cat(center_parts)
    if cluster_offset != k:
        raise RuntimeError(f"spatial quota produced {cluster_offset} centers, expected {k}")

    # Assign zero-quota cells to the nearest selected center globally.  This
    # preserves the exact budget while ensuring every background token belongs
    # to a cluster for centroid/coordinate aggregation.
    unassigned = assignments < 0
    if bool(unassigned.any()):
        missing_idx = torch.nonzero(unassigned, as_tuple=False).flatten()
        missing_feat = _clustering_features(
            embeddings[missing_idx], coords[missing_idx],
            metric=metric, spatial_weight=spatial_weight,
        )
        missing_dist = torch.cdist(missing_feat.float(), centers.float()) ** 2
        min_dist, missing_assign = missing_dist.min(dim=1)
        assignments[missing_idx] = missing_assign
        inertia_sum += float(min_dist.sum().item())
    if bool((assignments < 0).any()) or bool((assignments >= k).any()):
        raise RuntimeError("spatial quota produced invalid cluster assignments")
    active_quotas = quotas[quotas > 0]
    return centers, assignments, selected, {
        "iterations": int(k),
        "inertia": float(inertia_sum / max(1, n)),
        "quota_grid_rows": grid_rows,
        "quota_grid_cols": grid_cols,
        "quota_active_cells": int((capacities > 0).sum().item()),
        "quota_covered_cells": int((quotas > 0).sum().item()),
        "quota_min_per_cell": int(min_per_cell),
        "quota_power": float(quota_power),
        "quota_diversity_weight": float(diversity_weight),
        "quota_min_allocated": int(active_quotas.min().item()),
        "quota_max_allocated": int(active_quotas.max().item()),
        "quota_allocations": quotas.detach().cpu().tolist(),
        "cell_capacities": capacities.detach().cpu().tolist(),
    }


def torch_local_kmeans(
    x: torch.Tensor,
    coords: torch.Tensor,
    k: int,
    *,
    max_iter: int = 20,
    tol: float = 1e-4,
    local_radius: float = 2.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """SLIC-style K-Means with a hard spatial candidate neighborhood.

    ``local_radius`` is measured in units of the expected cluster spacing
    ``1/sqrt(k)``. Every token is additionally allowed to use its nearest
    spatial center, so the constraint cannot leave a token unassigned.
    """
    if x.ndim != 2 or coords.shape != (x.shape[0], 2):
        raise ValueError("x must be (N,D) and coords must be (N,2)")
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    seed_idx = _farthest_point_indices(coords.float(), k, seed)
    centers = x[seed_idx].float().clone()
    center_xy = coords[seed_idx].float().clone()
    radius = max(1e-6, float(local_radius) / math.sqrt(float(k)))
    assignments = torch.zeros(n, dtype=torch.long, device=x.device)
    inertia = float("inf")
    iterations = 0
    for iterations in range(1, max(1, int(max_iter)) + 1):
        feature_dist = torch.cdist(x.float(), centers) ** 2
        spatial_dist = torch.cdist(coords.float(), center_xy)
        allowed = spatial_dist <= radius
        nearest_spatial = spatial_dist.argmin(dim=1)
        allowed[torch.arange(n, device=x.device), nearest_spatial] = True
        constrained = feature_dist.masked_fill(~allowed, torch.inf)
        min_dist, assignments = constrained.min(dim=1)

        sums = torch.zeros_like(centers)
        xy_sums = torch.zeros_like(center_xy)
        sums.index_add_(0, assignments, x.float())
        xy_sums.index_add_(0, assignments, coords.float())
        counts = torch.bincount(assignments, minlength=k).float()
        new_centers = sums / counts.clamp_min(1).unsqueeze(1)
        new_xy = xy_sums / counts.clamp_min(1).unsqueeze(1)
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
        if empty.numel():
            worst = torch.argsort(min_dist, descending=True)
            for cid, point_idx in zip(empty.tolist(), worst.tolist()):
                new_centers[cid] = x[point_idx]
                new_xy[cid] = coords[point_idx]
        shift = torch.linalg.vector_norm(new_centers - centers, dim=1).max()
        centers, center_xy = new_centers, new_xy
        inertia = float(min_dist.mean().item())
        if float(shift.item()) <= float(tol):
            break

    feature_dist = torch.cdist(x.float(), centers) ** 2
    spatial_dist = torch.cdist(coords.float(), center_xy)
    allowed = spatial_dist <= radius
    nearest_spatial = spatial_dist.argmin(dim=1)
    allowed[torch.arange(n, device=x.device), nearest_spatial] = True
    _, assignments = feature_dist.masked_fill(~allowed, torch.inf).min(dim=1)
    centers, assignments = _repair_empty_clusters(x, centers, assignments, k)
    min_dist = ((x.float() - centers[assignments]) ** 2).sum(dim=1)
    return centers, assignments, {
        "iterations": iterations,
        "inertia": float(min_dist.mean().item()),
        "local_radius": float(local_radius),
        "effective_radius": radius,
    }


def torch_tome(
    embeddings: torch.Tensor,
    coords: torch.Tensor,
    k: int,
    *,
    metric: str = "cosine",
    spatial_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Iterative ToMe-style bipartite matching on encoder output tokens.

    Standard ToMe normally matches transformer keys inside a ViT. At this
    pre-LLM boundary we instead match normalized visual embeddings, preserve
    token mass with size-weighted averaging, and keep original memberships so
    medoids and reconstruction metrics remain well defined.
    """
    n = int(embeddings.shape[0])
    k = max(1, min(int(k), n))
    cur_emb = embeddings.float().clone()
    cur_xy = coords.float().clone()
    cur_size = torch.ones(n, device=embeddings.device, dtype=torch.float32)
    members: list[list[int]] = [[i] for i in range(n)]
    rounds = 0

    while int(cur_emb.shape[0]) > k:
        rounds += 1
        count = int(cur_emb.shape[0])
        a_idx = torch.arange(0, count, 2, device=embeddings.device)
        b_idx = torch.arange(1, count, 2, device=embeddings.device)
        if b_idx.numel() == 0:
            break
        feat = _clustering_features(
            cur_emb, cur_xy, metric=metric, spatial_weight=spatial_weight,
        )
        pair_dist = torch.cdist(feat[a_idx], feat[b_idx]) ** 2
        best_dist, best_b_local = pair_dist.min(dim=1)
        n_merge = min(count - k, int(a_idx.numel()))
        merge_a_local = torch.argsort(best_dist)[:n_merge]
        merge_a_set = set(merge_a_local.tolist())

        new_emb: list[torch.Tensor] = []
        new_xy: list[torch.Tensor] = []
        new_size: list[torch.Tensor] = []
        new_members: list[list[int]] = []
        for b_local, b_global in enumerate(b_idx.tolist()):
            sources = [b_global]
            sources.extend(
                int(a_idx[a_local].item())
                for a_local in merge_a_local.tolist()
                if int(best_b_local[a_local].item()) == b_local
            )
            src = torch.as_tensor(sources, device=embeddings.device, dtype=torch.long)
            weight = cur_size[src]
            total = weight.sum().clamp_min(1e-6)
            new_emb.append((cur_emb[src] * weight[:, None]).sum(dim=0) / total)
            new_xy.append((cur_xy[src] * weight[:, None]).sum(dim=0) / total)
            new_size.append(total)
            merged_members: list[int] = []
            for source in sources:
                merged_members.extend(members[source])
            new_members.append(merged_members)

        for a_local, a_global in enumerate(a_idx.tolist()):
            if a_local in merge_a_set:
                continue
            new_emb.append(cur_emb[a_global])
            new_xy.append(cur_xy[a_global])
            new_size.append(cur_size[a_global])
            new_members.append(members[a_global])

        cur_emb = torch.stack(new_emb)
        cur_xy = torch.stack(new_xy)
        cur_size = torch.stack(new_size)
        order_np = np.lexsort((
            cur_xy[:, 1].detach().cpu().numpy(),
            cur_xy[:, 0].detach().cpu().numpy(),
        ))
        order = torch.as_tensor(order_np, device=embeddings.device, dtype=torch.long)
        cur_emb, cur_xy, cur_size = cur_emb[order], cur_xy[order], cur_size[order]
        members = [new_members[i] for i in order.tolist()]

    assignments = torch.empty(n, dtype=torch.long, device=embeddings.device)
    for cid, idx in enumerate(members):
        assignments[torch.as_tensor(idx, device=embeddings.device, dtype=torch.long)] = cid
    orig_feat = _clustering_features(
        embeddings, coords, metric=metric, spatial_weight=spatial_weight,
    )
    centers_feat = _clustering_features(
        cur_emb, cur_xy, metric=metric, spatial_weight=spatial_weight,
    )
    inertia = ((orig_feat - centers_feat[assignments]) ** 2).sum(dim=1).mean()
    return centers_feat, assignments, {
        "iterations": rounds,
        "inertia": float(inertia.item()),
    }


def _chunk_assignments(n: int, k: int, device: torch.device) -> torch.Tensor:
    """Exactly ``k`` non-empty contiguous raster-order groups."""
    boundaries = torch.linspace(0, n, k + 1, device=device).round().long()
    out = torch.empty(n, dtype=torch.long, device=device)
    for cid in range(k):
        out[boundaries[cid]:boundaries[cid + 1]] = cid
    return out


def _centroids_from_assignments(
    values: torch.Tensor,
    assignments: torch.Tensor,
    k: int,
) -> torch.Tensor:
    sums = torch.zeros(k, values.shape[1], device=values.device, dtype=torch.float32)
    sums.index_add_(0, assignments, values.float())
    counts = torch.bincount(assignments, minlength=k).float().clamp_min(1)
    return (sums / counts.unsqueeze(1)).to(values.dtype)


def _weighted_chunk_centroids(
    bg: torch.Tensor,
    assignments: torch.Tensor,
    k: int,
    fg_summary: torch.Tensor | None,
    *,
    alpha: float,
    beta: float,
    temperature: float,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for cid in range(k):
        chunk = bg[assignments == cid]
        x = chunk.float()
        dev = torch.linalg.vector_norm(x - x.mean(dim=0, keepdim=True), dim=1)
        if fg_summary is None:
            rel = torch.zeros_like(dev)
        else:
            rel = F.cosine_similarity(x, fg_summary.float().unsqueeze(0), dim=1)

        def z(v: torch.Tensor) -> torch.Tensor:
            std = v.std(unbiased=False)
            return torch.zeros_like(v) if float(std.item()) < 1e-6 else (v - v.mean()) / (std + 1e-6)

        logits = (float(alpha) * z(rel) + float(beta) * z(dev)) / max(float(temperature), 1e-4)
        weight = torch.softmax(logits, dim=0)
        rows.append((weight[:, None] * x).sum(dim=0).to(bg.dtype))
    return torch.stack(rows)


def _nearest_medoids(
    features: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    k: int,
) -> torch.Tensor:
    selected: list[int] = []
    for cid in range(k):
        idx = torch.nonzero(assignments == cid, as_tuple=False).flatten()
        if idx.numel() == 0:
            idx = torch.arange(features.shape[0], device=features.device)
        local = torch.linalg.vector_norm(features[idx] - centers[cid], dim=1)
        selected.append(int(idx[local.argmin()].item()))
    return torch.as_tensor(selected, device=features.device, dtype=torch.long)


def compress_background_tokens(
    embeddings: torch.Tensor,
    fg_mask: np.ndarray | torch.Tensor,
    *,
    coords: torch.Tensor | np.ndarray | None = None,
    method: str = "spatial_kmeans",
    keep_ratio: float = 0.10,
    target_total_keep_ratio: float | None = None,
    n_clusters: int | None = None,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    spatial_weight: float = 0.25,
    metric: str = "cosine",
    prototype: str = "centroid",
    max_iter: int = 20,
    minibatch_size: int = 256,
    local_radius: float = 2.0,
    quota_grid_rows: int = 4,
    quota_grid_cols: int = 4,
    quota_min_per_cell: int = 1,
    quota_power: float = 0.5,
    quota_diversity_weight: float = 1.0,
    seed: int = 0,
    weighted_alpha: float = 0.0,
    weighted_beta: float = 1.0,
    weighted_temperature: float = 1.0,
) -> dict[str, Any]:
    """Compress background tokens to a fixed budget and preserve all foreground.

    ``prototype='centroid'`` gives the lowest reconstruction error;
    ``prototype='medoid'`` keeps real encoder tokens and is useful for a clean
    pruning-vs-merging ablation.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be (N,D), got {tuple(embeddings.shape)}")
    method = method.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}, got {method!r}")
    if prototype not in ("centroid", "medoid"):
        raise ValueError("prototype must be centroid or medoid")
    n = int(embeddings.shape[0])
    fg = torch.as_tensor(fg_mask, dtype=torch.bool, device=embeddings.device).reshape(-1)
    if fg.numel() != n:
        raise ValueError(f"fg_mask length {fg.numel()} != N {n}")
    if coords is None:
        xy = torch.stack(
            (torch.zeros(n, device=embeddings.device), torch.linspace(0, 1, n, device=embeddings.device)),
            dim=1,
        )
    else:
        xy = torch.as_tensor(coords, dtype=torch.float32, device=embeddings.device)
        if xy.shape != (n, 2):
            raise ValueError(f"coords must be (N,2), got {tuple(xy.shape)}")

    fg_idx = torch.nonzero(fg, as_tuple=False).flatten()
    bg_idx = torch.nonzero(~fg, as_tuple=False).flatten()
    n_bg = int(bg_idx.numel())
    k = _resolve_budget(n_bg, keep_ratio, n_clusters)
    if n_bg == 0:
        # All visual tokens overlap OCR foreground.  This is a valid
        # passthrough case: keep the original visual sequence unchanged.
        return {
            "embeddings": embeddings,
            "fg_mask": fg.detach().cpu().numpy(),
            "coords": xy,
            "assignments": torch.empty(0, dtype=torch.long),
            "representative_source_idx": torch.empty(0, dtype=torch.long),
            "bg_representatives": embeddings.new_empty((0, embeddings.shape[1])),
            "background_indices": torch.empty(0, dtype=torch.long),
            "foreground_indices": fg_idx.detach().cpu(),
            "sampled_background_indices": torch.empty(0, dtype=torch.long),
            "source_indices": fg_idx.detach().cpu(),
            "region_ids": _spatial_cell_ids(xy, quota_grid_rows, quota_grid_cols).detach().cpu(),
            "sampling_quota": torch.empty(0, dtype=torch.long),
            "meta": {
                "method": method,
                "prototype": prototype,
                "requested_prototype": prototype,
                "metric": metric,
                "spatial_weight": float(spatial_weight),
                "n_tokens_orig": n,
                "n_tokens_out": n,
                "n_fg": int(fg.sum()),
                "n_bg_orig": 0,
                "n_bg_out": 0,
                "requested_keep_ratio": float(keep_ratio),
                "actual_bg_keep_ratio": 1.0,
                "compression_ratio": 1.0,
                "cluster_wall_sec": 0.0,
                "local_radius": float(local_radius) if method == "local_kmeans" else None,
                "seed": int(seed),
            },
        }

    bg = embeddings[bg_idx]
    bg_xy = xy[bg_idx]
    if embeddings.is_cuda:
        torch.cuda.synchronize(embeddings.device)
    t0 = time.perf_counter()
    cluster_stats: dict[str, Any] = {}
    representative_local_idx = torch.full((k,), -1, device=embeddings.device, dtype=torch.long)
    effective_prototype = prototype

    if method in ("mean", "weighted"):
        effective_prototype = "centroid"
        assignments = _chunk_assignments(n_bg, k, embeddings.device)
        centers_feat = None
        if method == "mean":
            representatives = _centroids_from_assignments(bg, assignments, k)
        else:
            fg_summary = embeddings[fg_idx].float().mean(dim=0) if fg_idx.numel() else None
            representatives = _weighted_chunk_centroids(
                bg, assignments, k, fg_summary,
                alpha=weighted_alpha,
                beta=weighted_beta,
                temperature=weighted_temperature,
            )
    elif method == "random":
        effective_prototype = "medoid"
        gen = torch.Generator(device=embeddings.device)
        gen.manual_seed(int(seed))
        representative_local_idx = torch.randperm(n_bg, generator=gen, device=embeddings.device)[:k]
        representatives = bg[representative_local_idx]
        centers_feat = _clustering_features(
            representatives, bg_xy[representative_local_idx], metric=metric, spatial_weight=0.0,
        )
        all_feat = _clustering_features(bg, bg_xy, metric=metric, spatial_weight=0.0)
        assignments = torch.cdist(all_feat, centers_feat).argmin(dim=1)
    else:
        use_spatial = method in (
            "spatial_kmeans", "local_kmeans", "fps", "spatial_quota_fps", "tome",
        )
        feat = _clustering_features(
            bg,
            bg_xy,
            metric=metric,
            spatial_weight=spatial_weight if use_spatial else 0.0,
        )
        if method == "minibatch_kmeans":
            centers_feat, assignments, cluster_stats = torch_minibatch_kmeans(
                feat, k, max_iter=max_iter, batch_size=minibatch_size, seed=seed,
            )
        elif method == "local_kmeans":
            centers_feat, assignments, cluster_stats = torch_local_kmeans(
                feat, bg_xy, k, max_iter=max_iter, local_radius=local_radius, seed=seed,
            )
        elif method == "fps":
            centers_feat, assignments, selected, cluster_stats = torch_fps(
                feat, k, seed=seed,
            )
            if prototype == "medoid":
                representative_local_idx = selected
        elif method == "spatial_quota_fps":
            centers_feat, assignments, selected, cluster_stats = torch_spatial_quota_fps(
                bg,
                bg_xy,
                k,
                metric=metric,
                spatial_weight=spatial_weight,
                grid_rows=quota_grid_rows,
                grid_cols=quota_grid_cols,
                min_per_cell=quota_min_per_cell,
                quota_power=quota_power,
                diversity_weight=quota_diversity_weight,
                seed=seed,
            )
            if prototype == "medoid":
                representative_local_idx = selected
        elif method == "tome":
            centers_feat, assignments, cluster_stats = torch_tome(
                bg, bg_xy, k, metric=metric,
                spatial_weight=spatial_weight if use_spatial else 0.0,
            )
        else:
            centers_feat, assignments, cluster_stats = torch_kmeans(
                feat, k, max_iter=max_iter, seed=seed,
            )
        if prototype == "medoid":
            representative_local_idx = _nearest_medoids(feat, centers_feat, assignments, k)
            representatives = bg[representative_local_idx]
        else:
            representatives = _centroids_from_assignments(bg, assignments, k)

    rep_xy = _centroids_from_assignments(bg_xy, assignments, k).float()
    if effective_prototype == "medoid" and (representative_local_idx >= 0).all():
        rep_xy = bg_xy[representative_local_idx]
    representative_source_idx = torch.where(
        representative_local_idx >= 0,
        bg_idx[representative_local_idx.clamp_min(0)],
        torch.full_like(representative_local_idx, -1),
    )

    # Merge foreground and representative background tokens in spatial reading
    # order.  This is more faithful than appending all centroids at the end.
    all_emb = torch.cat((embeddings[fg_idx], representatives), dim=0)
    all_xy = torch.cat((xy[fg_idx], rep_xy), dim=0)
    all_fg = torch.cat((torch.ones(fg_idx.numel(), dtype=torch.bool, device=fg.device),
                        torch.zeros(k, dtype=torch.bool, device=fg.device)))
    order_np = np.lexsort((
        all_xy[:, 1].detach().cpu().numpy(),
        all_xy[:, 0].detach().cpu().numpy(),
    ))
    order = torch.as_tensor(order_np, device=embeddings.device, dtype=torch.long)
    out = all_emb[order].contiguous()
    out_xy = all_xy[order]
    out_fg = all_fg[order]
    representative_source_idx = representative_source_idx.detach().cpu()
    source_indices = torch.cat((fg_idx.detach().cpu(), representative_source_idx), dim=0)[order.detach().cpu()]
    sampled_background_indices = representative_source_idx[representative_source_idx >= 0]
    region_ids = _spatial_cell_ids(xy, quota_grid_rows, quota_grid_cols).detach().cpu()

    if embeddings.is_cuda:
        torch.cuda.synchronize(embeddings.device)
    elapsed = time.perf_counter() - t0
    meta = {
        "method": method,
        "prototype": effective_prototype,
        "requested_prototype": prototype,
        "metric": metric,
        "spatial_weight": float(
            spatial_weight
            if method in ("spatial_kmeans", "local_kmeans", "fps", "spatial_quota_fps", "tome")
            else 0.0
        ),
        "n_tokens_orig": n,
        "n_fg": int(fg.sum().item()),
        "n_bg_orig": n_bg,
        "n_bg_out": k,
        "n_tokens_out": int(out.shape[0]),
        "requested_keep_ratio": float(keep_ratio),
        "actual_bg_keep_ratio": float(k / n_bg),
        "compression_ratio": float(n / max(1, int(out.shape[0]))),
        "cluster_wall_sec": elapsed,
        "local_radius": float(local_radius) if method == "local_kmeans" else None,
        "seed": int(seed),
        **cluster_stats,
    }
    return {
        "embeddings": out,
        "fg_mask": out_fg.detach().cpu().numpy(),
        "coords": out_xy,
        "assignments": assignments.detach().cpu(),
        "representative_source_idx": representative_source_idx.detach().cpu(),
        "bg_representatives": representatives,
        "background_indices": bg_idx.detach().cpu(),
        "foreground_indices": fg_idx.detach().cpu(),
        "sampled_background_indices": sampled_background_indices,
        "source_indices": source_indices,
        "region_ids": region_ids,
        "sampling_quota": torch.as_tensor(
            cluster_stats.get("quota_allocations", []), dtype=torch.long,
        ),
        "meta": meta,
    }


def encode_merge_batch_cluster(
    model: Any,
    processor: Any,
    pil_image: Any,
    bboxes_xyxy: list[tuple[int, int, int, int]],
    *,
    orig_hw: tuple[int, int],
    bg_merge_factor: int = 4,  # accepted for B-1 runner compatibility
    device: str = "cuda",
    method: str = "spatial_kmeans",
    keep_ratio: float = 0.10,
    target_total_keep_ratio: float | None = None,
    n_clusters: int | None = None,
    spatial_weight: float = 0.25,
    metric: str = "cosine",
    prototype: str = "centroid",
    max_iter: int = 20,
    minibatch_size: int = 256,
    local_radius: float = 2.0,
    quota_grid_rows: int = 4,
    quota_grid_cols: int = 4,
    quota_min_per_cell: int = 1,
    quota_power: float = 0.5,
    quota_diversity_weight: float = 1.0,
    seed: int = 0,
    weighted_alpha: float = 0.0,
    weighted_beta: float = 1.0,
    weighted_temperature: float = 1.0,
    fg_dilate: int = 0,
    debug_dir: str | None = None,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    target_visual_tokens: int | None = None,
    return_visualization: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Framed M3 image -> cluster-compressed B-1-compatible visual pack."""
    inner = _resolve_inner_vlm(model)
    visual = getattr(inner, "visual", None)
    merge_size = int(getattr(visual, "spatial_merge_size", 2)) if visual is not None else 2
    patch_size = int(getattr(visual, "patch_size", 16)) if visual is not None else 16
    if target_visual_tokens is None:
        resized_pil, resized_input_hw = resize_image_for_vllm_budget(
            pil_image,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            patch_size=patch_size,
            merge_size=merge_size,
        )
    else:
        resized_pil, resized_input_hw = resize_image_to_visual_token_target(
            pil_image,
            target_tokens=target_visual_tokens,
            patch_size=patch_size,
            merge_size=merge_size,
        )
    # External embedding injection needs only visual tensors.  Calling the
    # image processor directly avoids ProcessorMixin silently dropping image
    # resize kwargs on older Qwen3.5 installations.
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise AttributeError("processor lacks image_processor")
    image_kwargs: dict[str, int] = {}
    if max_pixels is not None:
        image_kwargs["max_pixels"] = int(max_pixels)
    if min_pixels is not None:
        image_kwargs["min_pixels"] = int(min_pixels)
    inputs = image_processor(
        images=[resized_pil], return_tensors="pt", **image_kwargs,
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise KeyError("processor output lacks pixel_values/image_grid_thw")

    embeds = extract_visual_embeddings_hf(model, pixel_values, image_grid_thw)
    n_orig = int(embeds.shape[0])
    grid = [int(value) for value in image_grid_thw[0].tolist()]
    expected_tokens = (
        (resized_input_hw[1] // patch_size)
        * (resized_input_hw[0] // patch_size)
        // (merge_size * merge_size)
    )
    if n_orig != expected_tokens:
        raise RuntimeError(
            "HF image processor changed the explicitly resized input: "
            f"expected {expected_tokens} visual tokens from {resized_input_hw}, got {n_orig}"
        )
    gh, gw = max(1, grid[1] // merge_size), max(1, grid[2] // merge_size)
    resized_h, resized_w = max(1, grid[1] * patch_size), max(1, grid[2] * patch_size)
    fg = bbox_mask_on_grid(
        bboxes_xyxy,
        resized_h=resized_h,
        resized_w=resized_w,
        grid_h=gh,
        grid_w=gw,
        orig_h=orig_hw[1],
        orig_w=orig_hw[0],
        merge_size=merge_size,
        patch_size=patch_size,
    )
    if fg.shape[0] != n_orig:
        fg = fg[:n_orig] if fg.shape[0] > n_orig else np.pad(fg, (0, n_orig - fg.shape[0]))
    if fg_dilate > 0 and gh * gw == n_orig:
        fg = _dilate_fg_2d(fg.reshape(gh, gw), int(fg_dilate)).reshape(-1)
    coords = grid_coordinates(gh, gw).to(embeds.device)
    if coords.shape[0] != n_orig:
        coords = torch.stack((torch.zeros(n_orig), torch.linspace(0, 1, n_orig)), dim=1).to(embeds.device)

    effective_keep_ratio = float(keep_ratio)
    if target_total_keep_ratio is not None:
        if n_clusters is not None:
            raise ValueError("target_total_keep_ratio cannot be combined with n_clusters")
        if not 0.0 < float(target_total_keep_ratio) <= 1.0:
            raise ValueError("target_total_keep_ratio must be in (0, 1]")
        n_fg = int(fg.sum())
        n_bg = n_orig - n_fg
        target_total = int(round(n_orig * float(target_total_keep_ratio)))
        if n_bg > 0:
            # Foreground tokens are never dropped.  Translate a total-token
            # budget into the background representative ratio for this batch.
            effective_keep_ratio = max(1, target_total - n_fg) / n_bg
            effective_keep_ratio = min(1.0, effective_keep_ratio)

    compressed = compress_background_tokens(
        embeds,
        fg,
        coords=coords,
        method=method,
        keep_ratio=effective_keep_ratio,
        n_clusters=n_clusters,
        spatial_weight=spatial_weight,
        metric=metric,
        prototype=prototype,
        max_iter=max_iter,
        minibatch_size=minibatch_size,
        local_radius=local_radius,
        quota_grid_rows=quota_grid_rows,
        quota_grid_cols=quota_grid_cols,
        quota_min_per_cell=quota_min_per_cell,
        quota_power=quota_power,
        quota_diversity_weight=quota_diversity_weight,
        seed=seed,
        weighted_alpha=weighted_alpha,
        weighted_beta=weighted_beta,
        weighted_temperature=weighted_temperature,
    )
    merged = compressed["embeddings"]
    merged_grid = compute_merged_grid_thw(
        image_grid_thw[0], n_orig, int(merged.shape[0]), merge_size=merge_size,
    )
    merged_aligned, grid_aligned, align_info = align_merged_pack_for_vllm(
        merged, merged_grid, merge_size=merge_size,
    )

    debug_path = None
    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"cluster_{time.time_ns()}"
        debug_path = out_dir / f"{stamp}.npz"
        np.savez_compressed(
            debug_path,
            coords_orig=coords.detach().cpu().numpy(),
            fg_mask=fg.astype(bool),
            coords_out=compressed["coords"].detach().cpu().numpy(),
            assignments=compressed["assignments"].numpy(),
            representative_source_idx=compressed["representative_source_idx"].numpy(),
        )

    meta = {
        **compressed["meta"],
        "requested_bg_keep_ratio": float(keep_ratio),
        "effective_bg_keep_ratio": effective_keep_ratio,
        "requested_total_keep_ratio": target_total_keep_ratio,
        "n_tokens_merged": int(merged_aligned.shape[0]),
        "actual_total_keep_ratio": float(merged_aligned.shape[0] / max(1, n_orig)),
        "n_fg_orig": int(fg.sum()),
        "fg_dilate": int(fg_dilate),
        "grid_orig": grid,
        "resized_input_hw": list(resized_input_hw),
        "target_visual_tokens": target_visual_tokens,
        "grid_merged": [int(x) for x in grid_aligned.tolist()],
        "compress_ratio": round(n_orig / max(1, int(merged_aligned.shape[0])), 4),
        "vllm_align": align_info,
        "merge_mode": f"cluster_{method}",
        "position_mode": "b1_approx_grid",
        "debug_npz": str(debug_path) if debug_path else None,
    }
    pack = {
        "merged_embeds": merged_aligned.cpu(),
        "merged_grid_thw": grid_aligned.cpu(),
        "meta": meta,
    }
    if return_visualization:
        # Keep the exact pre-merger representation and selection provenance so
        # a figure can be regenerated without rerunning the language model.
        patch_xyxy_orig = []
        orig_w, orig_h = int(orig_hw[0]), int(orig_hw[1])
        for token_id in range(n_orig):
            row, col = divmod(token_id, max(1, gw))
            patch_xyxy_orig.append([
                col * orig_w / max(1, gw), row * orig_h / max(1, gh),
                (col + 1) * orig_w / max(1, gw), (row + 1) * orig_h / max(1, gh),
            ])
        pack["visualization"] = {
            "visual_tokens": embeds.detach().cpu(),
            "compressed_visual_tokens": compressed["embeddings"].detach().cpu(),
            "merged_visual_tokens": merged_aligned.detach().cpu(),
            "token_coords": coords.detach().float().cpu(),
            "token_patch_xyxy_orig": torch.tensor(patch_xyxy_orig, dtype=torch.float32),
            "foreground_indices": compressed["foreground_indices"],
            "background_indices": compressed["background_indices"],
            "sampled_background_indices": compressed["sampled_background_indices"],
            "source_indices": compressed["source_indices"],
            "assignments": compressed["assignments"],
            "representative_source_idx": compressed["representative_source_idx"],
            "compressed_coords": compressed["coords"].detach().float().cpu(),
            "compressed_fg_mask": compressed["fg_mask"],
            "region_ids": compressed["region_ids"],
            "sampling_quota": compressed["sampling_quota"],
            "quota_grid_rows": int(quota_grid_rows),
            "quota_grid_cols": int(quota_grid_cols),
            "grid_hw": [int(gh), int(gw)],
            # Preserve the processor's exact (t,h,w) grid for downstream
            # visualization/alignment audits; reconstructing it from the
            # merged grid loses information when a model uses a non-default
            # patch or merge factor.
            "processor_grid_thw": image_grid_thw.detach().cpu(),
            "patch_size": int(patch_size),
            "merge_size": int(merge_size),
            "resized_input_hw": [int(resized_input_hw[1]), int(resized_input_hw[0])],
        }
    return pack


__all__ = [
    "SUPPORTED_METHODS",
    "compress_background_tokens",
    "encode_merge_batch_cluster",
    "grid_coordinates",
    "resize_image_for_vllm_budget",
    "resize_image_to_visual_token_target",
    "torch_kmeans",
    "torch_fps",
    "torch_spatial_quota_fps",
    "torch_local_kmeans",
    "torch_minibatch_kmeans",
    "torch_tome",
]
