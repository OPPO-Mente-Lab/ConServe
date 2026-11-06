#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""B-1: OCR-guided visual token merge (after the vision encoder, before the LLM).

Applies a bbox mask to Qwen3-VL / Qwen3.5-VL visual tokens on the HF side: tokens outside
boxes are merged while in-box tokens are preserved; fed to vLLM via ``enable_mm_embeds``,
or used for HF direct generate.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def bbox_mask_on_grid(
    bboxes_xyxy: list[tuple[int, int, int, int]],
    *,
    resized_h: int,
    resized_w: int,
    grid_h: int,
    grid_w: int,
    orig_h: int,
    orig_w: int,
    merge_size: int = 2,
    patch_size: int = 14,
) -> np.ndarray:
    """Map original-image bboxes onto the post-merge visual token grid; return a bool mask of length grid_h*grid_w."""
    token_h = grid_h
    token_w = grid_w
    mask = np.zeros(token_h * token_w, dtype=bool)

    scale_x = resized_w / max(orig_w, 1)
    scale_y = resized_h / max(orig_h, 1)
    cell_w = resized_w / max(token_w, 1)
    cell_h = resized_h / max(token_h, 1)

    for x1, y1, x2, y2 in bboxes_xyxy:
        rx1 = x1 * scale_x
        ry1 = y1 * scale_y
        rx2 = x2 * scale_x
        ry2 = y2 * scale_y
        tx1 = int(np.floor(rx1 / cell_w))
        ty1 = int(np.floor(ry1 / cell_h))
        tx2 = int(np.ceil(rx2 / cell_w))
        ty2 = int(np.ceil(ry2 / cell_h))
        tx1 = max(0, min(token_w - 1, tx1))
        ty1 = max(0, min(token_h - 1, ty1))
        tx2 = max(tx1 + 1, min(token_w, tx2))
        ty2 = max(ty1 + 1, min(token_h, ty2))
        for ty in range(ty1, ty2):
            for tx in range(tx1, tx2):
                mask[ty * token_w + tx] = True

    return mask


def merge_visual_embeddings(
    embeddings: torch.Tensor,
    fg_mask: np.ndarray,
    *,
    bg_merge_factor: int = 4,
) -> tuple[torch.Tensor, np.ndarray]:
    """Merge background tokens along the token sequence (mean pooling); keep foreground."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings 应为 (N,D), 得到 {embeddings.shape}")
    n, _ = embeddings.shape
    if fg_mask.shape[0] != n:
        raise ValueError(f"fg_mask 长度 {fg_mask.shape[0]} != N {n}")

    merged_rows: list[torch.Tensor] = []
    merged_fg: list[bool] = []
    i = 0
    k = max(1, int(bg_merge_factor))
    while i < n:
        if fg_mask[i]:
            merged_rows.append(embeddings[i])
            merged_fg.append(True)
            i += 1
            continue
        j = i
        while j < n and not fg_mask[j]:
            j += 1
        bg_block = embeddings[i:j]
        for s in range(0, bg_block.shape[0], k):
            chunk = bg_block[s:s + k]
            merged_rows.append(chunk.mean(dim=0))
            merged_fg.append(False)
        i = j

    out = torch.stack(merged_rows, dim=0)
    return out, np.array(merged_fg, dtype=bool)


def vllm_image_token_count(grid_thw: torch.Tensor, merge_size: int = 2) -> int:
    """vLLM/Qwen: visual token count = prod(grid_thw) // merge_size^2."""
    g = [int(x) for x in grid_thw.detach().cpu().flatten().tolist()[:3]]
    t, h, w = max(1, g[0]), max(1, g[1]), max(1, g[2])
    m = max(1, int(merge_size))
    return t * h * w // (m * m)


def compute_merged_grid_thw(
    orig_grid_thw: torch.Tensor,
    orig_num_tokens: int,
    merged_num_tokens: int,
    *,
    merge_size: int = 2,
) -> torch.Tensor:
    """Find grid_thw (t,h,w) such that the vLLM placeholder count equals merged_num_tokens."""
    grid = orig_grid_thw.detach().cpu().flatten().tolist()
    if len(grid) != 3:
        t, h, w = 1, 1, 1
    else:
        t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    t = max(1, t)
    h = max(merge_size, h)
    w = max(merge_size, w)
    m = max(1, int(merge_size))
    target_prod = merged_num_tokens * m * m

    best: tuple[int, int, int] | None = None
    best_score = 10**18
    h_hi = max(h, w) * 3 + 1
    for nh in range(m, h_hi):
        denom = t * nh
        if denom <= 0 or target_prod % denom != 0:
            continue
        nw = target_prod // denom
        if nw < m:
            continue
        cand = torch.tensor([t, nh, nw])
        if vllm_image_token_count(cand, m) != merged_num_tokens:
            continue
        score = abs(nh - h) + abs(nw - w)
        if score < best_score:
            best_score = score
            best = (t, nh, nw)

    if best is None:
        nh = max(m, int(round(h * merged_num_tokens / max(orig_num_tokens, 1))))
        best = (t, nh, w)

    return torch.tensor(
        list(best), dtype=orig_grid_thw.dtype, device=orig_grid_thw.device,
    )


def align_merged_pack_for_vllm(
    merged_embeds: torch.Tensor,
    merged_grid_thw: torch.Tensor,
    *,
    merge_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Align the merged embed row count with the vLLM token count implied by grid_thw (pad/trim)."""
    expected = vllm_image_token_count(merged_grid_thw, merge_size)
    n = int(merged_embeds.shape[0])
    info: dict[str, Any] = {
        "n_tokens_vllm_expected": expected,
        "n_tokens_embed_before_align": n,
        "n_padded": 0,
        "n_trimmed": 0,
    }
    if n == expected:
        return merged_embeds, merged_grid_thw, info
    if n < expected:
        pad = merged_embeds[-1:].expand(expected - n, -1).clone()
        info["n_padded"] = expected - n
        return torch.cat([merged_embeds, pad], dim=0), merged_grid_thw, info
    info["n_trimmed"] = n - expected
    return merged_embeds[:expected].contiguous(), merged_grid_thw, info


def _resolve_inner_vlm(model: Any) -> Any:
    """AutoModelForImageTextToText or a visual-only shim."""
    if hasattr(model, "get_image_features") or hasattr(model, "visual"):
        return model
    if hasattr(model, "model"):
        return model.model
    return model


def _tensor_from_vision_output(out: Any) -> torch.Tensor:
    """Normalize the return value of visual / get_image_features to (N, D)."""
    if isinstance(out, torch.Tensor):
        t = out
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        po = out.pooler_output
        if isinstance(po, (tuple, list)):
            parts = [p for p in po if p is not None]
            if not parts:
                raise ValueError("pooler_output 为空")
            t = torch.cat(parts, dim=0)
        else:
            t = po
    elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        t = out.last_hidden_state
    elif isinstance(out, tuple) and out:
        return _tensor_from_vision_output(out[0])
    else:
        raise TypeError(f"无法从 vision 输出解析 tensor: {type(out)!r}")

    if t.ndim == 3:
        if t.shape[0] == 1:
            t = t.squeeze(0)
        else:
            t = t.reshape(-1, t.shape[-1])
    if t.ndim != 2:
        raise ValueError(f"期望 visual embeddings 形状 (N,D), 得到 {tuple(t.shape)}")
    return t


def extract_visual_embeddings_hf(
    model: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Extract visual token embeddings (N, D) from a Qwen3-VL / Qwen3.5-VL HF model."""
    inner = _resolve_inner_vlm(model)
    with torch.no_grad():
        getter = getattr(inner, "get_image_features", None)
        if getter is not None:
            out = getter(pixel_values, image_grid_thw=image_grid_thw, return_dict=True)
            return _tensor_from_vision_output(out)

        visual = getattr(inner, "visual", None)
        if visual is None:
            raise AttributeError("模型上找不到 get_image_features 或 .visual")

        out = visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        return _tensor_from_vision_output(out)


def encode_merge_batch(
    model: Any,
    processor: Any,
    pil_image: Any,
    bboxes_xyxy: list[tuple[int, int, int, int]],
    *,
    orig_hw: tuple[int, int],
    bg_merge_factor: int = 4,
    device: str = "cuda",
) -> dict[str, Any]:
    """Single framed image: processor -> visual encode -> OCR merge."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "placeholder"},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise KeyError("processor 输出缺少 pixel_values / image_grid_thw")

    embeds = extract_visual_embeddings_hf(model, pixel_values, image_grid_thw)
    n_orig = int(embeds.shape[0])
    grid = [int(x) for x in image_grid_thw[0].tolist()]

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
    if fg_mask.shape[0] != n_orig:
        if fg_mask.shape[0] > n_orig:
            fg_mask = fg_mask[:n_orig]
        else:
            pad = np.zeros(n_orig - fg_mask.shape[0], dtype=bool)
            fg_mask = np.concatenate([fg_mask, pad])

    merged, merged_fg = merge_visual_embeddings(
        embeds, fg_mask, bg_merge_factor=bg_merge_factor,
    )
    merged_grid = compute_merged_grid_thw(
        image_grid_thw[0], n_orig, merged.shape[0], merge_size=merge_size,
    )
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
            "grid_orig": grid,
            "grid_merged": [int(x) for x in grid_aligned.tolist()],
            "vllm_align": align_info,
        },
    }
