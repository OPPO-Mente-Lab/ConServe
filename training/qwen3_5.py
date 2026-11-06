# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright 2026 OPPO

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
)

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class TokenPruneMetadata:
    """Sequence rewrite produced after Qwen's vision merger.

    The FSDP engine consumes this immediately after the model forward to apply
    the same per-sample keep indices to teacher top-k tensors and nested
    offsets.  The language model itself only sees the compact tensors below.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    keep_indices: list[torch.Tensor]
    original_lengths: list[int]
    compact_lengths: list[int]
    image_stats: list[dict[str, int]]


def fast_pos_embed_interpolate(self, grid_thw):
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    # Modification: # Get device from grid_thw to avoid self.pos_embed being on CPU when FSDP2 enables cpu_offload
    device = grid_thw.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


def _maybe_compress_student_image_embeds(model, image_embeds, image_grid_thw):
    """Optional student-only spatial_quota_fps scatter."""
    try:
        import sys
        from pathlib import Path

        # The token_compress package lives at the checkout root.
        checkout_root = Path(__file__).resolve().parents[4]
        if str(checkout_root) not in sys.path:
            sys.path.insert(0, str(checkout_root))
        from token_compress import compress_image_embeds_scatter  # noqa: WPS433
        from token_compress.context import compress_context  # noqa: WPS433

        ctx = compress_context()
        if ctx is None or not ctx.config.enabled:
            return image_embeds
        if ctx.config.mode == "prune":
            # Physical pruning is handled together with placeholder and
            # position-id reconstruction below, not by this length-preserving
            # compatibility helper.
            return image_embeds
        if ctx.config.mode != "scatter":
            raise ValueError(f"Unknown compression mode {ctx.config.mode!r}; expected scatter or prune")
        visual = getattr(model, "visual", None)
        if visual is None and hasattr(model, "model"):
            visual = getattr(model.model, "visual", None)
        if visual is None:
            return image_embeds
        return compress_image_embeds_scatter(
            image_embeds, image_grid_thw, visual=visual, cfg=ctx.config
        )
    except Exception as exc:  # noqa: BLE001 — never break training if compress fails
        logger.warning("Token compression skipped: %s", exc)
        return image_embeds


def _prune_image_tokens_before_llm(
    model: Any,
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TokenPruneMetadata]:
    """Physically remove unselected image placeholders in the dense FSDP path.

    ``position_ids`` is gathered at the selected original token positions.  In
    particular, the final three rows retain Qwen3.5's original temporal/height/
    width M-RoPE coordinates instead of inventing a regular compressed grid.
    """
    import sys
    from pathlib import Path

    checkout_root = Path(__file__).resolve().parents[4]
    if str(checkout_root) not in sys.path:
        sys.path.insert(0, str(checkout_root))
    from token_compress import select_one_image_embeds  # noqa: WPS433
    from token_compress.context import compress_context  # noqa: WPS433

    ctx = compress_context()
    if ctx is None or not ctx.config.enabled or ctx.config.mode != "prune":
        raise RuntimeError("prune requested without an enabled prune CompressContext")
    if input_ids.ndim != 2 or inputs_embeds.ndim != 3 or attention_mask is None:
        raise ValueError("prune requires dense input_ids/inputs_embeds/attention_mask tensors")
    if position_ids is None or position_ids.ndim != 3 or position_ids.shape[0] != 4:
        raise ValueError("prune requires Qwen3.5 position_ids with shape (4, batch, sequence)")
    if position_ids.shape[1:] != input_ids.shape:
        raise ValueError(
            f"position_ids shape {tuple(position_ids.shape)} does not match input_ids {tuple(input_ids.shape)}"
        )

    visual = getattr(model, "visual", None)
    if visual is None:
        raise RuntimeError("Qwen3.5 model has no visual module")
    merge_size = int(getattr(visual, "spatial_merge_size", 2) or 2)
    image_sizes = (image_grid_thw.prod(dim=-1) // (merge_size * merge_size)).tolist()
    if sum(int(n) for n in image_sizes) != int(image_embeds.shape[0]):
        raise ValueError(
            f"image_grid_thw implies {sum(int(n) for n in image_sizes)} merged tokens, "
            f"but vision returned {image_embeds.shape[0]}"
        )

    selected_images = []
    image_stats: list[dict[str, int]] = []
    offset = 0
    for image_idx, n_image_tokens in enumerate(image_sizes):
        n_image_tokens = int(n_image_tokens)
        image_part = image_embeds[offset : offset + n_image_tokens]
        offset += n_image_tokens
        boxes = ctx.boxes_xyxy[image_idx] if image_idx < len(ctx.boxes_xyxy) else []
        hw = ctx.orig_hw[image_idx] if image_idx < len(ctx.orig_hw) else (1, 1)
        selected = select_one_image_embeds(
            image_part,
            grid_thw=image_grid_thw[image_idx].tolist(),
            boxes_xyxy=boxes,
            orig_hw=hw,
            visual=visual,
            cfg=ctx.config,
        )
        selected_images.append(selected)
        image_stats.append(
            {
                "original_visual_tokens": n_image_tokens,
                "foreground_visual_tokens": selected.foreground_count,
                "selected_background_tokens": selected.background_count,
                "compressed_visual_tokens": int(selected.selected_indices.numel()),
            }
        )

    image_token_id = model.config.image_token_id
    batch_size, original_width = input_ids.shape
    image_cursor = 0
    compact_ids: list[torch.Tensor] = []
    compact_embeds: list[torch.Tensor] = []
    compact_positions: list[torch.Tensor] = []
    keep_indices: list[torch.Tensor] = []
    original_lengths: list[int] = []
    compact_lengths: list[int] = []

    for batch_idx in range(batch_size):
        valid = attention_mask[batch_idx].bool()
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        original_lengths.append(int(valid_indices.numel()))
        image_positions = torch.nonzero(
            (input_ids[batch_idx] == image_token_id) & valid,
            as_tuple=False,
        ).flatten()

        # Each image occupies one contiguous placeholder run.  A batch can
        # contain several images, while a text-only sample consumes none.
        runs: list[torch.Tensor] = []
        if image_positions.numel():
            run_start = 0
            for pos_idx in range(1, int(image_positions.numel()) + 1):
                if pos_idx == image_positions.numel() or image_positions[pos_idx] != image_positions[pos_idx - 1] + 1:
                    runs.append(image_positions[run_start:pos_idx])
                    run_start = pos_idx

        keep = valid.clone()
        sample_embeds = inputs_embeds[batch_idx].clone()
        for run in runs:
            if image_cursor >= len(selected_images):
                raise ValueError("more image placeholder runs than image_grid_thw rows")
            selected = selected_images[image_cursor]
            image_cursor += 1
            if int(run.numel()) != image_stats[image_cursor - 1]["original_visual_tokens"]:
                raise ValueError(
                    "image placeholder run does not match merger output: "
                    f"run={run.numel()}, features={image_stats[image_cursor - 1]['original_visual_tokens']}"
                )
            keep[run] = False
            kept_positions = run[selected.selected_indices]
            keep[kept_positions] = True
            sample_embeds[kept_positions] = selected.embeddings.to(sample_embeds.dtype)

        selected_positions = torch.nonzero(keep, as_tuple=False).flatten()
        compact_ids.append(input_ids[batch_idx, selected_positions])
        compact_embeds.append(sample_embeds[selected_positions])
        compact_positions.append(position_ids[:, batch_idx, selected_positions])
        keep_indices.append(selected_positions)
        compact_lengths.append(int(selected_positions.numel()))

    if image_cursor != len(selected_images):
        raise ValueError(f"image_grid_thw has {len(selected_images)} images but found {image_cursor} placeholder runs")

    compact_width = max(compact_lengths, default=0)
    hidden_size = inputs_embeds.shape[-1]
    padded_ids = torch.zeros((batch_size, compact_width), dtype=input_ids.dtype, device=input_ids.device)
    padded_embeds = torch.zeros(
        (batch_size, compact_width, hidden_size), dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    padded_attention = torch.zeros((batch_size, compact_width), dtype=attention_mask.dtype, device=input_ids.device)
    padded_positions = torch.zeros((4, batch_size, compact_width), dtype=position_ids.dtype, device=input_ids.device)
    for batch_idx, compact_len in enumerate(compact_lengths):
        padded_ids[batch_idx, :compact_len] = compact_ids[batch_idx]
        padded_embeds[batch_idx, :compact_len] = compact_embeds[batch_idx]
        padded_attention[batch_idx, :compact_len] = 1
        padded_positions[:, batch_idx, :compact_len] = compact_positions[batch_idx]

    if os.environ.get("TOKEN_COMPRESS_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        for image_idx, stats in enumerate(image_stats):
            logger.warning(
                "Token prune image=%d N=%d F=%d B=%d M=%d ratio=%.3f",
                image_idx,
                stats["original_visual_tokens"],
                stats["foreground_visual_tokens"],
                stats["selected_background_tokens"],
                stats["compressed_visual_tokens"],
                stats["original_visual_tokens"] / max(1, stats["compressed_visual_tokens"]),
            )
        logger.warning("Token prune sequence lengths: original=%s compact=%s", original_lengths, compact_lengths)

    metadata = TokenPruneMetadata(
        input_ids=padded_ids,
        attention_mask=padded_attention,
        position_ids=padded_positions,
        keep_indices=keep_indices,
        original_lengths=original_lengths,
        compact_lengths=compact_lengths,
        image_stats=image_stats,
    )
    return padded_embeds, padded_attention, padded_positions, metadata


def _get_input_embeds(
    model: "Qwen3_5CausalLMOutputWithPast",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        # ``prune`` changes both image features and sequence tensors.  It must
        # happen after the vision merger, before the language model, rather
        # than via the legacy length-preserving scatter helper.
        import sys
        from pathlib import Path

        checkout_root = Path(__file__).resolve().parents[4]
        if str(checkout_root) not in sys.path:
            sys.path.insert(0, str(checkout_root))
        from token_compress.context import compress_context  # noqa: WPS433

        ctx = compress_context()
        if ctx is not None and ctx.config.enabled and ctx.config.mode == "prune":
            if video_grid_thw is not None or pixel_values_videos is not None:
                raise NotImplementedError("Token pruning currently supports image-only batches")
            inputs_embeds, attention_mask, position_ids, prune_metadata = _prune_image_tokens_before_llm(
                model,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )
            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_prune_metadata": prune_metadata,
            }

        image_embeds = _maybe_compress_student_image_embeds(model, image_embeds, image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


def qwen3_5_base_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw,
        kwargs.get("position_ids"),
    )  # avoid lora module having multiple keyword arguments
    prune_metadata = input_kwargs.pop("token_prune_metadata", None)
    kwargs.update(input_kwargs)
    outputs = self.language_model(
        input_ids=None,
        **kwargs,
    )
    if prune_metadata is not None:
        outputs.token_prune_metadata = prune_metadata
    return outputs


@dataclass
class Qwen3_5CausalLMOutputForPPO(Qwen3_5CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def forward_with_normal_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    result = Qwen3_5CausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )
    if hasattr(outputs, "token_prune_metadata"):
        result.token_prune_metadata = outputs.token_prune_metadata
    return result


def forward_with_torch_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # See `dense_common.forward_with_torch_backend` for the `shift_labels`
    # rationale (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    vocab_weights = self.lm_head.weight
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()

    ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sequence_parallel_size > 1:
        rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
            rolled_labels, position_ids_rmpad=None, sp_size=ulysses_sequence_parallel_size
        )
    hidden_states = hidden_states.to(vocab_weights.dtype)  # bf16 to float
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=vocab_weights,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # See `dense_common.forward_with_torch_backend` for the `shift_labels`
    # rationale (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")
    ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sequence_parallel_size > 1:
        rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
            rolled_labels, position_ids_rmpad=None, sp_size=ulysses_sequence_parallel_size
        )

    vocab_weights = self.lm_head.weight
    hidden_states = hidden_states.to(vocab_weights.dtype)
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        vocab_weights,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
