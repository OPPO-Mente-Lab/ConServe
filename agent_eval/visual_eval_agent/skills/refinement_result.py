# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence, overload

from ..models import VisualError


@dataclass(frozen=True)
class RefinementResult(Sequence[VisualError]):
    """List-compatible Skill output with preserved rejection audit records."""

    errors: list[VisualError]
    rejected_errors: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self) -> Iterator[VisualError]:
        return iter(self.errors)

    @overload
    def __getitem__(self, index: int) -> VisualError: ...

    @overload
    def __getitem__(self, index: slice) -> list[VisualError]: ...

    def __getitem__(self, index: int | slice) -> VisualError | list[VisualError]:
        return self.errors[index]


def normalized_rejections(raw_items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in raw_items or []:
        if not isinstance(raw, dict) or not raw.get("original_error_id"):
            continue
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized.append(
            {
                "original_error_id": str(raw["original_error_id"]),
                "reason": str(raw.get("reason") or "unspecified"),
                "evidence": str(raw.get("evidence") or ""),
                "confidence": confidence,
                "direct_contradiction": raw.get("direct_contradiction") is True,
            }
        )
    return normalized
