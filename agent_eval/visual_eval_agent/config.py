# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualEvaluationConfig:
    detail_skill_confidence_threshold: float = 0.65
    ocr_detail_confidence_threshold: float = 0.65
    minor_skill_trigger_count: int = 3
    enable_visual_critic: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "VisualEvaluationConfig":
        data = data or {}
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})
