#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""B-3: foreground-protected "separable 2D grid compression" (grid-aware separable merge).

Improvements over B-1
---------------------
After the merger, B-1 **flattens the 2D visual grid into 1D**, then applies ``bg_merge_factor``
mean pooling to consecutive background segments, and finally back-computes a single
``grid_thw`` from the token count alone. Problems:

1. 1D adjacency != 2D adjacency (the end of one row and the start of the next row become
   adjacent after flattening), so pooled background tokens have no stable 2D spatial meaning;
2. The back-computed ``grid_thw`` only guarantees that the placeholder count matches; it does
   not record the true ``(h,w)`` origin of each token. M-RoPE on the LLM side then reassigns
   positions according to the new ``grid_thw``, disturbing the within-row order / cross-row
   relationships of foreground text tokens.

B-3 instead constructs a **new regular grid** ``(H', W')`` directly on the **2D grid**:

- reshape input embeddings to ``(gh, gw, D)`` and the foreground mask to ``(gh, gw)``;
- optionally dilate the foreground (``fg_dilate``) to protect text boundaries;
- decide protection and merging **separately along the row and column axes**: rows/columns
  containing foreground keep the original granularity (1 cell), while consecutive purely
  background rows/columns are block-wise mean-pooled by ``bg_merge_factor``;
- the Cartesian product of row groups x column groups **naturally forms a regular grid**
  ``(H', W')`` -> flattened to ``(H'*W', D)``;
- ``merged_grid_thw = [t, H'*spatial_merge_size, W'*spatial_merge_size]`` is derived from the
  true ``(H', W')`` rather than back-computed.

Why "separable" row/column blocking instead of arbitrary 2x2/3x3/4x4 tiles?
---------------------------------------------------------------------------
For the output to **remain a valid regular ``grid_thw``** (so that default vLLM/HF M-RoPE is
self-consistent), the grid must be a rectangular lattice defined by a "row partition x column
partition". Arbitrary-sized 2D blocks (quadtree) cannot tile into a regular rectangular grid
and therefore cannot be represented as a single ``grid_thw``. B-3 thus uses **separable
row/column blocking**: this is the only self-consistent form of "foreground-protected
compression strictly compatible with grid_thw". The trade-off is: as long as a row (column)
contains any foreground, the entire row (column) is retained at the original granularity, so
the compression ratio naturally drops when foreground is scattered — this is exactly the
intended and acceptable behavior (such images genuinely need more visual tokens).

A more aggressive variant with arbitrary 2D blocks + center coordinates is B-4
(``ogstb_b4_sparse_coord.py``); it requires rewriting the M-RoPE position-id injection logic
and is more invasive engineering-wise.

Token-order assumptions (verified against transformers 5.8.1 Qwen3-VL)
----------------------------------------------------------------------
- merger output tokens traverse ``(gh, gw)`` in row-major order (Qwen3-VL has no window reordering);
- ``get_vision_position_ids`` gives a row-major ``(h=i, w=j)`` meshgrid for images, with both
  axes sharing the same ``start_position`` offset. Hence B-3's regular grid + ``[t, H'*m, W'*m]``
  is fully self-consistent with default M-RoPE.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ogstb_b1_merge import (
    align_merged_pack_for_vllm,
    bbox_mask_on_grid,
    extract_visual_embeddings_hf,
    _resolve_inner_vlm,
)


def _dilate_fg_2d(fg2d: np.ndarray, radius: int) -> np.ndarray:
    """Dilate the 2D foreground mask with a square (Chebyshev) footprint of ``radius`` rings to protect text boundaries."""
    if radius <= 0 or not fg2d.any():
        return fg2d
    gh, gw = fg2d.shape
    out = fg2d.copy()
    ys, xs = np.nonzero(fg2d)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny = np.clip(ys + dy, 0, gh - 1)
            nx = np.clip(xs + dx, 0, gw - 1)
            out[ny, nx] = True
    return out


def _axis_groups(protected: np.ndarray, bg_merge_factor: int) -> list[tuple[int, int]]:
    """Partition one axis (length L) into ordered (start, end) groups.

    Foreground (protected) positions -> each becomes its own group of 1 (original granularity);
    consecutive pure-background positions -> every ``bg_merge_factor`` positions form 1 group.
    """
    k = max(1, int(bg_merge_factor))
    groups: list[tuple[int, int]] = []
    i = 0
    n = int(protected.shape[0])
    while i < n:
        if protected[i]:
            groups.append((i, i + 1))
            i += 1
            continue
        j = i
        while j < n and not protected[j]:
            j += 1
        for s in range(i, j, k):
            groups.append((s, min(j, s + k)))
        i = j
    return groups


def grid_aware_separable_merge(
    embeddings: torch.Tensor,
    fg_mask: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    bg_merge_factor: int = 3,
    fg_dilate: int = 1,
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int], dict[str, Any]]:
    """Foreground-protected separable 2D grid compression.

    Args:
        embeddings: ``(N, D)``, ``N == grid_h * grid_w`` (after the merger, row-major).
        fg_mask: ``(N,)`` bool, foreground (text to translate) token mask, row-major.
        grid_h / grid_w: grid height/width after the merger (i.e. ``grid_thw[1]//m`` / ``grid_thw[2]//m``).
        bg_merge_factor: consecutive background cells are merged into 1 per k along each axis (k x k or smaller at borders).
        fg_dilate: number of 2D foreground dilation rings to protect text boundaries (0=no dilation).

    Returns:
        merged: ``(H'*W', D)`` new regular grid, row-major.
        merged_fg: ``(H'*W',)`` bool, marks whether the cell comes from both a foreground row and a foreground column (approximate foreground).
        (Hp, Wp): new grid size.
        meta: row/column grouping and coordinate information (for debugging/paper).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings 应为 (N,D), 得到 {tuple(embeddings.shape)}")
    n, d = embeddings.shape
    if n != grid_h * grid_w:
        raise ValueError(
            f"N={n} != grid_h*grid_w={grid_h}*{grid_w}={grid_h * grid_w}"
        )
    if fg_mask.shape[0] != n:
        raise ValueError(f"fg_mask 长度 {fg_mask.shape[0]} != N {n}")

    emb2d = embeddings.view(grid_h, grid_w, d)
    fg2d = fg_mask.reshape(grid_h, grid_w).astype(bool)
    fg2d_dil = _dilate_fg_2d(fg2d, fg_dilate)

    row_protected = fg2d_dil.any(axis=1)
    col_protected = fg2d_dil.any(axis=0)

    row_groups = _axis_groups(row_protected, bg_merge_factor)
    col_groups = _axis_groups(col_protected, bg_merge_factor)
    hp, wp = len(row_groups), len(col_groups)

    out = embeddings.new_zeros((hp * wp, d))
    merged_fg = np.zeros(hp * wp, dtype=bool)
    cell_coords: list[tuple[float, float, int, int, int, int]] = []

    idx = 0
    for (r0, r1) in row_groups:
        row_is_fg = bool(row_protected[r0]) and (r1 - r0 == 1)
        for (c0, c1) in col_groups:
            col_is_fg = bool(col_protected[c0]) and (c1 - c0 == 1)
            block = emb2d[r0:r1, c0:c1, :]
            out[idx] = block.reshape(-1, d).mean(dim=0)
            # Treat as (approximate) foreground only when this cell lies in both a foreground row & column and is 1x1
            merged_fg[idx] = row_is_fg and col_is_fg and bool(fg2d[r0, c0])
            cell_coords.append((
                (r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0, r0, r1, c0, c1,
            ))
            idx += 1

    meta = {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "Hp": hp,
        "Wp": wp,
        "n_rows_protected": int(row_protected.sum()),
        "n_cols_protected": int(col_protected.sum()),
        "bg_merge_factor": int(bg_merge_factor),
        "fg_dilate": int(fg_dilate),
        "row_groups": row_groups,
        "col_groups": col_groups,
    }
    return out, merged_fg, (hp, wp), meta


def encode_merge_batch_grid(
    model: Any,
    processor: Any,
    pil_image: Any,
    bboxes_xyxy: list[tuple[int, int, int, int]],
    *,
    orig_hw: tuple[int, int],
    bg_merge_factor: int = 3,
    fg_dilate: int = 1,
    device: str = "cuda",
) -> dict[str, Any]:
    """Single framed image: processor -> visual encode -> B-3 2D grid compression.

    Interface-aligned with ``ogstb_b1_merge.encode_merge_batch``, but:
      - uses ``grid_aware_separable_merge`` for 2D-aware merging;
      - ``merged_grid_thw`` is derived from the true ``(H', W')`` (natively aligned, no pad/trim needed).
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "placeholder"},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise KeyError("processor 输出缺少 pixel_values / image_grid_thw")

    embeds = extract_visual_embeddings_hf(model, pixel_values, image_grid_thw)
    n_orig = int(embeds.shape[0])
    grid = [int(x) for x in image_grid_thw[0].tolist()]
    t = max(1, grid[0])

    inner = _resolve_inner_vlm(model)
    visual = getattr(inner, "visual", None)
    merge_size = int(getattr(visual, "spatial_merge_size", 2)) if visual is not None else 2
    patch_size = int(getattr(visual, "patch_size", 14)) if visual is not None else 14

    gh = max(1, grid[1] // merge_size)
    gw = max(1, grid[2] // merge_size)
    resized_h = max(1, grid[1] * patch_size)
    resized_w = max(1, grid[2] * patch_size)

    fg_mask = bbox_mask_on_grid(
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
    # Align mask length with the token count (defensive trimming using the same convention as B-1)
    if fg_mask.shape[0] != n_orig:
        if fg_mask.shape[0] > n_orig:
            fg_mask = fg_mask[:n_orig]
        else:
            fg_mask = np.concatenate(
                [fg_mask, np.zeros(n_orig - fg_mask.shape[0], dtype=bool)],
            )

    # If gh*gw disagrees with n_orig (rare deepstack/alignment cases), fall back to a row-vector approximate grid
    if gh * gw != n_orig:
        gw = max(1, n_orig // max(1, gh))
        gh = max(1, n_orig // gw)
        if gh * gw != n_orig:
            gh, gw = 1, n_orig
        fg_mask = fg_mask[:gh * gw]
        embeds = embeds[:gh * gw]
        n_orig = gh * gw

    merged, merged_fg, (hp, wp), grid_meta = grid_aware_separable_merge(
        embeds, fg_mask,
        grid_h=gh, grid_w=gw,
        bg_merge_factor=bg_merge_factor,
        fg_dilate=fg_dilate,
    )

    merged_grid = torch.tensor(
        [t, hp * merge_size, wp * merge_size],
        dtype=image_grid_thw.dtype, device=image_grid_thw.device,
    )
    # The true grid should be natively aligned (H'*W' == placeholder count); still run one defensive align
    merged_aligned, grid_aligned, align_info = align_merged_pack_for_vllm(
        merged, merged_grid, merge_size=merge_size,
    )

    return {
        "merged_embeds": merged_aligned.cpu(),
        "merged_grid_thw": grid_aligned.cpu(),
        "meta": {
            "n_tokens_orig": n_orig,
            "n_tokens_merged": int(merged_aligned.shape[0]),
            "n_fg_orig": int(fg_mask.sum()),
            "n_fg_merged": int(merged_fg.sum()),
            "bg_merge_factor": bg_merge_factor,
            "fg_dilate": fg_dilate,
            "grid_orig": grid,
            "grid_merged": [int(x) for x in grid_aligned.tolist()],
            "Hp": hp,
            "Wp": wp,
            "n_rows_protected": grid_meta["n_rows_protected"],
            "n_cols_protected": grid_meta["n_cols_protected"],
            "compress_ratio": round(n_orig / max(1, int(merged_aligned.shape[0])), 4),
            "vllm_align": align_info,
            "merge_mode": "b3_grid_separable",
        },
    }
