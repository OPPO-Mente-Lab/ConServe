# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from .llm_client import ModelClientError
from visual_eval_agent.llm_client import VisionModelClientError

T = TypeVar("T")


def call_with_retries(
    callable_obj: Callable[[], T],
    retries: int,
    base_delay_seconds: float = 1.5,
    max_delay_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return callable_obj()
        except (ModelClientError, VisionModelClientError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = min(max_delay_seconds, base_delay_seconds * (2**attempt))
            sleep(delay)
    raise last_error or RuntimeError("Model call failed")
