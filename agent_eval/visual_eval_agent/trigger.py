# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .config import VisualEvaluationConfig
from .models import Severity, VisualError


def plan_text_rendering_skill_call(
    errors: list[VisualError],
    config: VisualEvaluationConfig,
    translated_text_regions: list[dict] | None = None,
) -> dict | None:
    target_ids: list[str] = []
    trigger_reasons: list[str] = []
    for error in errors:
        if (
            error.severity in {Severity.MAJOR, Severity.CRITICAL}
            or error.confidence < config.detail_skill_confidence_threshold
            or error.risk_level == "high"
            or error.needs_skill
        ):
            target_ids.append(error.error_id)
            trigger_reasons.append("candidate_requires_detail")

    minor_errors = [error for error in errors if error.severity == Severity.MINOR]
    if len(minor_errors) >= config.minor_skill_trigger_count:
        target_ids.extend(error.error_id for error in minor_errors)
        trigger_reasons.append("repeated_minor")

    target_region_ids: list[str] = []
    for region in translated_text_regions or []:
        confidence = region.get("ocr_confidence")
        if confidence is None or float(confidence) >= config.ocr_detail_confidence_threshold:
            continue
        region_id = region.get("region_id")
        if region_id is not None:
            target_region_ids.append(str(region_id))
        trigger_reasons.append("low_ocr_confidence")

    target_ids = list(dict.fromkeys(target_ids))
    target_region_ids = list(dict.fromkeys(target_region_ids))
    trigger_reasons = list(dict.fromkeys(trigger_reasons))
    if not target_ids and not target_region_ids:
        return None
    return {
        "skill": "text_rendering_detail",
        "target_error_ids": target_ids,
        "target_region_ids": target_region_ids,
        "trigger_reasons": trigger_reasons,
    }


def plan_layout_skill_call(
    errors: list[VisualError],
    config: VisualEvaluationConfig,
) -> dict | None:
    target_ids: list[str] = []
    trigger_reasons: list[str] = []
    for error in errors:
        if error.severity in {Severity.MAJOR, Severity.CRITICAL}:
            target_ids.append(error.error_id)
            trigger_reasons.append("major_or_critical")
        if error.confidence < config.detail_skill_confidence_threshold:
            target_ids.append(error.error_id)
            trigger_reasons.append("low_confidence")
        if error.risk_level == "high":
            target_ids.append(error.error_id)
            trigger_reasons.append("high_risk")
        if error.needs_skill:
            target_ids.append(error.error_id)
            trigger_reasons.append("explicit_detail_request")

    minor_errors = [error for error in errors if error.severity == Severity.MINOR]
    if len(minor_errors) >= config.minor_skill_trigger_count:
        target_ids.extend(error.error_id for error in minor_errors)
        trigger_reasons.append("repeated_minor")

    target_ids = list(dict.fromkeys(target_ids))
    trigger_reasons = list(dict.fromkeys(trigger_reasons))
    if not target_ids:
        return None
    return {
        "skill": "layout_detail",
        "target_error_ids": target_ids,
        "trigger_reasons": trigger_reasons,
    }


def plan_background_skill_call(
    errors: list[VisualError],
    config: VisualEvaluationConfig,
    detail_risk_regions: list[dict] | None = None,
) -> dict | None:
    target_ids: list[str] = []
    trigger_reasons: list[str] = []
    for error in errors:
        if error.severity in {Severity.MAJOR, Severity.CRITICAL}:
            target_ids.append(error.error_id)
            trigger_reasons.append("major_or_critical")
        if error.confidence < config.detail_skill_confidence_threshold:
            target_ids.append(error.error_id)
            trigger_reasons.append("low_confidence")
        if error.risk_level == "high":
            target_ids.append(error.error_id)
            trigger_reasons.append("high_risk")
        if error.needs_skill:
            target_ids.append(error.error_id)
            trigger_reasons.append("explicit_detail_request")

    minor_errors = [error for error in errors if error.severity == Severity.MINOR]
    if len(minor_errors) >= config.minor_skill_trigger_count:
        target_ids.extend(error.error_id for error in minor_errors)
        trigger_reasons.append("repeated_minor")

    risk_regions = [
        region
        for region in detail_risk_regions or []
        if float(region.get("confidence", 0.0))
        >= config.detail_skill_confidence_threshold
    ]
    if risk_regions:
        trigger_reasons.append("complex_background_risk")

    target_ids = list(dict.fromkeys(target_ids))
    trigger_reasons = list(dict.fromkeys(trigger_reasons))
    if not target_ids and not risk_regions:
        return None
    return {
        "skill": "background_detail",
        "target_error_ids": target_ids,
        "detail_risk_regions": risk_regions,
        "trigger_reasons": trigger_reasons,
    }
