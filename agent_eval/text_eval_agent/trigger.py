# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .config import TextEvaluationConfig
from .models import Severity, TextDimension, TextError


def plan_text_skill_calls(errors: list[TextError], config: TextEvaluationConfig) -> list[dict]:
    plans: dict[str, dict] = {}
    for error in errors:
        skill = _skill_for_error(error)
        should_trigger = (
            error.severity in {Severity.MAJOR, Severity.CRITICAL}
            or error.confidence < config.detail_skill_confidence_threshold
            or error.risk_level == "high"
            or error.needs_skill
        )
        if not should_trigger:
            continue

        _add_error_to_plan(plans, skill, error)

    minor_groups: dict[str, list[TextError]] = {}
    for error in errors:
        if error.severity != Severity.MINOR:
            continue
        minor_groups.setdefault(_skill_for_error(error), []).append(error)

    for skill, minor_errors in minor_groups.items():
        if len(minor_errors) < config.minor_skill_trigger_count:
            continue
        for error in minor_errors:
            _add_error_to_plan(plans, skill, error)

    return list(plans.values())


def _skill_for_error(error: TextError) -> str:
    return "accuracy_detail" if error.dimension == TextDimension.FAITHFULNESS else "fluency_style"


def _add_error_to_plan(plans: dict[str, dict], skill: str, error: TextError) -> None:
    if skill not in plans:
        plans[skill] = {"skill": skill, "target_error_ids": [], "target_pair_ids": []}
    if error.error_id not in plans[skill]["target_error_ids"]:
        plans[skill]["target_error_ids"].append(error.error_id)
    if error.pair_id not in plans[skill]["target_pair_ids"]:
        plans[skill]["target_pair_ids"].append(error.pair_id)


