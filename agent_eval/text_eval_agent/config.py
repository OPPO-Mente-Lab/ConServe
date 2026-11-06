# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class TextEvaluationConfig:
    detail_skill_confidence_threshold: float = 0.65
    high_risk_cap_score_100: int = 40
    critical_cap_score_100: int = 50
    faithfulness_weight: float = 0.5
    fluency_weight: float = 0.5
    minor_skill_trigger_count: int = 3


