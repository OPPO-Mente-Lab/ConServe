# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

from .config import CompressConfig, load_config_from_env


@dataclass
class CompressContext:
    """Per-forward compression state (student only)."""

    config: CompressConfig = field(default_factory=load_config_from_env)
    # one entry per image in this forward (aligned with image_grid_thw rows)
    boxes_xyxy: list[list[tuple[int, int, int, int]]] = field(default_factory=list)
    # (H, W) original / framed image size per image
    orig_hw: list[tuple[int, int]] = field(default_factory=list)


_CTX: ContextVar[Optional[CompressContext]] = ContextVar("token_compress_ctx", default=None)


def compress_context() -> Optional[CompressContext]:
    return _CTX.get()


def set_compress_context(ctx: Optional[CompressContext]):
    return _CTX.set(ctx)


def reset_compress_context(token) -> None:
    _CTX.reset(token)


@contextmanager
def using_compress_context(ctx: Optional[CompressContext]):
    token = set_compress_context(ctx)
    try:
        yield ctx
    finally:
        reset_compress_context(token)
