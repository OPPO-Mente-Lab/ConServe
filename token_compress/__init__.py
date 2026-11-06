# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Training-time visual token compression (spatial_quota_fps).

Default OFF. Enable with env TOKEN_COMPRESS=1.

Train mode uses *scatter_centroid*: FPS/quota selects background prototypes, then
every background token is replaced by its cluster prototype so sequence length
(and flash-attn varlen packing) stay unchanged. This matches the inference
method's prototype selection while remaining drop-in for HF / verl FSDP.

Inference (vLLM merged_embeds reduce) stays in image-translation's
run_M3_multi_box_cluster.py — not replaced by this package.
"""
from __future__ import annotations

from .config import CompressConfig, load_config_from_env
from .context import CompressContext, compress_context, reset_compress_context, set_compress_context
from .apply import PrunedImageEmbeds, compress_image_embeds_scatter, select_one_image_embeds

__all__ = [
    "CompressConfig",
    "CompressContext",
    "PrunedImageEmbeds",
    "compress_context",
    "compress_image_embeds_scatter",
    "select_one_image_embeds",
    "load_config_from_env",
    "reset_compress_context",
    "set_compress_context",
]
