# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..models import (
    SEVERITY_PENALTY,
    Severity,
    VisualDimension,
    VisualError,
    ensure_unique_error_ids,
)


LAYOUT_TYPE_ALIASES = {
    "rotated_text": "reading_direction_mismatch",
    "wrong_orientation": "reading_direction_mismatch",
    "orientation_mismatch": "reading_direction_mismatch",
    "line_break_error": "unreasonable_line_break",
    "bad_line_break": "unreasonable_line_break",
    "overflow": "text_overflow",
    "occlusion": "content_occlusion",
    "wrong_position": "position_shift",
    "misalignment": "alignment_error",
    "wrong_region": "wrong_region_fill",
}

CRITICAL_IMPACT_TAGS = {
    "wrong_content_association",
    "key_content_hidden",
    "primary_content_unrecoverable",
    "page_unusable",
}

_SEVERITY_RANK = {
    Severity.MINOR: 0,
    Severity.MAJOR: 1,
    Severity.CRITICAL: 2,
}


def normalize_layout_items(
    raw_items: list[dict[str, Any]],
    task_id: str,
    default_source: str,
    default_risk_level: str = "normal",
) -> list[VisualError]:
    errors: list[VisualError] = []
    for raw_item in raw_items:
        item = dict(raw_item)
        item["type"] = LAYOUT_TYPE_ALIASES.get(str(item.get("type")), str(item.get("type")))
        comparison = item.get("comparison") or {}
        if _as_bool(comparison.get("bbox_evidence_match")) is False:
            continue
        if item["type"] == "reading_direction_mismatch" and bool(
            comparison.get("orientation_preserved")
            or comparison.get("source_arrangement_preserved")
        ):
            continue
        item["bbox_1000"] = _sanitize_model_bbox(
            _ensure_affected_area(item.get("bbox_1000"))
        )
        if item["bbox_1000"] is None:
            item["localization_status"] = "global"
            item["bbox_confidence"] = 0.0
            item["confidence"] = min(_safe_confidence(item.get("confidence")), 0.6)
            item["needs_skill"] = True
        error = VisualError.from_model_dict(
            item,
            task_id,
            default_source,
            default_risk_level,
            dimension=VisualDimension.LAYOUT,
        )
        if error.severity == Severity.CRITICAL and not (
            set(error.impact_tags) & CRITICAL_IMPACT_TAGS
        ):
            error = replace(
                error,
                severity=Severity.MAJOR,
                penalty=SEVERITY_PENALTY[Severity.MAJOR],
            )
        errors.append(error)

    consolidated = consolidate_layout_errors(errors)
    stable = [
        replace(error, error_id=_stable_error_id(task_id, error))
        for error in consolidated
    ]
    return calibrate_layout_errors(ensure_unique_error_ids(stable))


def calibrate_layout_errors(errors: list[VisualError]) -> list[VisualError]:
    """Apply deterministic V4 layout severity caps."""

    calibrated: list[VisualError] = []
    for error in errors:
        if error.type == "unreasonable_line_break":
            evidence = _nested(error.comparison, "line_break_evidence")
            isolated_split = all(
                (
                    evidence.get("scope") in {"single_word", "short_phrase"},
                    _as_bool(evidence.get("characters_complete")) is True,
                    _as_bool(evidence.get("reading_order_preserved")) is True,
                    _as_bool(evidence.get("content_association_preserved")) is True,
                    _as_bool(evidence.get("overlap_or_clipping")) is False,
                    _as_bool(evidence.get("material_reconstruction_required")) is False,
                )
            )
            if isolated_split and error.severity != Severity.MINOR:
                error = replace(
                    error,
                    severity=Severity.MINOR,
                    penalty=SEVERITY_PENALTY[Severity.MINOR],
                )
        calibrated.append(error)
    return calibrated


def filter_unstable_direction_errors(
    candidates: list[VisualError],
    refined: list[VisualError],
) -> tuple[list[VisualError], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject scored direction claims whose independent measurements conflict.

    A direction mismatch is unusually easy to hallucinate when the page itself is
    photographed sideways or when upright CJK glyphs are used in a rotated text
    flow. A Skill confirmation is accepted only when its core measurements agree
    with the Base Agent for the same error id.
    """

    unmatched_candidates = {error.error_id: error for error in candidates}
    kept: list[VisualError] = []
    rejections: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for error in refined:
        candidate = _match_candidate(error, unmatched_candidates)
        if candidate is not None:
            unmatched_candidates.pop(candidate.error_id, None)
            if error.error_id != candidate.error_id:
                error = replace(error, error_id=candidate.error_id)
        conflicts = (
            _orientation_conflicts(candidate.comparison, error.comparison)
            if candidate is not None and error.type == "reading_direction_mismatch"
            else []
        )
        if not conflicts:
            if error.type != "reading_direction_mismatch":
                kept.append(error)
                continue
            anchor_status = orientation_anchor_status(error.comparison)
            if anchor_status == "confirmed_mismatch":
                kept.append(error)
                continue
            if anchor_status == "preserved":
                evidence = (
                    "A shared non-CJK orientation anchor has equivalent source and "
                    "target rotation, so the alleged direction change is contradicted."
                )
                rejection = {
                    "original_error_id": error.error_id,
                    "reason": "orientation_anchor_preserved",
                    "evidence": evidence,
                    "confidence": 1.0,
                    "direct_contradiction": True,
                }
                rejections.append(rejection)
                warnings.append(
                    {
                        "candidate_error_id": error.error_id,
                        "dimension": error.dimension.value,
                        "severity": error.severity.value,
                        "reason": "orientation_anchor_preserved",
                        "rejection": rejection,
                    }
                )
                continue
            kept.append(
                replace(
                    error,
                    severity=Severity.MINOR,
                    penalty=SEVERITY_PENALTY[Severity.MINOR],
                )
            )
            warnings.append(
                {
                    "candidate_error_id": error.error_id,
                    "dimension": error.dimension.value,
                    "severity": error.severity.value,
                    "reason": "orientation_anchor_insufficient",
                    "rejection": None,
                }
            )
            continue
        evidence = (
            "Base Agent and Layout Detail Skill disagree on orientation fields: "
            + ", ".join(conflicts)
            + ". The direction claim requires human review and is not scored."
        )
        rejection = {
            "original_error_id": error.error_id,
            "reason": "orientation_measurement_conflict",
            "evidence": evidence,
            "confidence": 1.0,
            "direct_contradiction": True,
        }
        rejections.append(rejection)
        warnings.append(
            {
                "candidate_error_id": error.error_id,
                "dimension": error.dimension.value,
                "severity": error.severity.value,
                "reason": "orientation_measurement_conflict",
                "rejection": rejection,
            }
        )
    return kept, rejections, warnings


def _match_candidate(
    refined: VisualError,
    unmatched: dict[str, VisualError],
) -> VisualError | None:
    exact = unmatched.get(refined.error_id)
    if exact is not None:
        return exact
    same_type = [candidate for candidate in unmatched.values() if candidate.type == refined.type]
    same_region = [candidate for candidate in same_type if _same_region(candidate, refined)]
    if len(same_region) == 1:
        return same_region[0]
    if len(same_type) == 1:
        return same_type[0]
    return None


def orientation_anchor_status(comparison: dict[str, Any]) -> str:
    """Classify whether non-CJK evidence proves a source-relative rotation change."""

    evidence = _nested(comparison, "orientation_anchor_evidence")
    kind = str(evidence.get("anchor_kind") or "").strip().lower()
    supported_kinds = {
        "latin_text",
        "number",
        "punctuation",
        "container_axis",
        "ui_geometry",
        "mixed",
    }
    source_rotation = _rotation(evidence.get("source_anchor_rotation_degrees"))
    target_rotation = _rotation(evidence.get("target_anchor_rotation_degrees"))
    sufficient = all(
        (
            kind in supported_kinds,
            _as_bool(evidence.get("source_anchor_visible")) is True,
            _as_bool(evidence.get("target_anchor_visible")) is True,
            _as_bool(evidence.get("cjk_glyph_shape_only")) is False,
            source_rotation is not None,
            target_rotation is not None,
        )
    )
    if not sufficient:
        return "insufficient"
    mismatch = _as_bool(evidence.get("anchor_mismatch_confirmed"))
    rotation_changed = _rotation_distance(source_rotation, target_rotation) > 15
    if mismatch is True and rotation_changed:
        return "confirmed_mismatch"
    if mismatch is False or not rotation_changed:
        return "preserved"
    return "insufficient"


def _orientation_conflicts(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    conflicts: list[str] = []
    for field in ("source_page_rotation", "target_page_rotation"):
        left = _rotation(first.get(field))
        right = _rotation(second.get(field))
        if left is not None and right is not None and _rotation_distance(left, right) > 15:
            conflicts.append(field)
    for field in (
        "source_block_axis",
        "target_block_axis",
        "source_arrangement_preserved",
        "new_head_rotation_required",
    ):
        left = first.get(field)
        right = second.get(field)
        if left is not None and right is not None and str(left).lower() != str(right).lower():
            conflicts.append(field)
    first_anchor = _nested(first, "orientation_anchor_evidence")
    second_anchor = _nested(second, "orientation_anchor_evidence")
    for field in (
        "anchor_kind",
        "anchor_mismatch_confirmed",
        "cjk_glyph_shape_only",
    ):
        left = first_anchor.get(field)
        right = second_anchor.get(field)
        if left is not None and right is not None and str(left).lower() != str(right).lower():
            conflicts.append(f"orientation_anchor_evidence.{field}")
    for field in (
        "source_anchor_rotation_degrees",
        "target_anchor_rotation_degrees",
    ):
        left = _rotation(first_anchor.get(field))
        right = _rotation(second_anchor.get(field))
        if left is not None and right is not None and _rotation_distance(left, right) > 15:
            conflicts.append(f"orientation_anchor_evidence.{field}")
    return conflicts


def _rotation(value: Any) -> float | None:
    try:
        return float(value) % 360
    except (TypeError, ValueError):
        return None


def _rotation_distance(first: float, second: float) -> float:
    difference = abs(first - second) % 360
    return min(difference, 360 - difference)


def consolidate_layout_errors(errors: list[VisualError]) -> list[VisualError]:
    ordered = sorted(errors, key=_spatial_sort_key)
    consolidated: list[VisualError] = []
    for error in ordered:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(consolidated)
                if existing.type == error.type and _same_region(existing, error)
            ),
            None,
        )
        if duplicate_index is None:
            consolidated.append(error)
            continue
        consolidated[duplicate_index] = _merge_errors(
            consolidated[duplicate_index], error
        )
    return sorted(consolidated, key=_spatial_sort_key)


def _ensure_affected_area(bbox_1000: Any) -> Any:
    if not isinstance(bbox_1000, dict) or "affected_area" in bbox_1000:
        return bbox_1000
    fallback = bbox_1000.get("text_area") or bbox_1000.get("container_area")
    if fallback is None:
        return bbox_1000
    return {"affected_area": list(fallback), **bbox_1000}


def _sanitize_model_bbox(bbox_1000: Any) -> dict[str, list[int]] | None:
    if not isinstance(bbox_1000, dict):
        return None
    sanitized: dict[str, list[int]] = {}
    for name, coordinates in bbox_1000.items():
        normalized = _sanitize_coordinates(coordinates)
        if normalized is not None:
            sanitized[str(name)] = normalized
    if "affected_area" not in sanitized:
        fallback = sanitized.get("text_area") or sanitized.get("container_area")
        if fallback is None:
            return None
        sanitized["affected_area"] = list(fallback)
    return sanitized


def _sanitize_coordinates(coordinates: Any) -> list[int] | None:
    if (
        not isinstance(coordinates, list)
        or len(coordinates) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in coordinates
        )
    ):
        return None
    values = [max(0, min(1000, round(float(value)))) for value in coordinates]
    ymin, ymax = sorted((values[0], values[2]))
    xmin, xmax = sorted((values[1], values[3]))
    if ymin == ymax:
        ymin, ymax = _expand_zero_span(ymin, ymax)
    if xmin == xmax:
        xmin, xmax = _expand_zero_span(xmin, xmax)
    return [ymin, xmin, ymax, xmax]


def _expand_zero_span(lower: int, upper: int) -> tuple[int, int]:
    if upper < 1000:
        return lower, upper + 1
    return max(0, lower - 1), upper


def _safe_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.6
    return max(0.0, min(1.0, number))


def _nested(value: dict[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key)
    return dict(nested) if isinstance(nested, dict) else {}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _same_region(first: VisualError, second: VisualError) -> bool:
    if first.region_id and second.region_id and first.region_id == second.region_id:
        return True
    first_box = _affected_box(first)
    second_box = _affected_box(second)
    return bool(first_box and second_box and _iou(first_box, second_box) >= 0.5)


def _merge_errors(first: VisualError, second: VisualError) -> VisualError:
    preferred = max(
        (first, second),
        key=lambda error: (_SEVERITY_RANK[error.severity], error.confidence),
    )
    severity = max((first.severity, second.severity), key=_SEVERITY_RANK.__getitem__)
    impact_tags = list(dict.fromkeys(first.impact_tags + second.impact_tags))
    suggestion_error = next(
        (error for error in (preferred, first, second) if error.suggestion),
        None,
    )
    suggestion = suggestion_error.suggestion if suggestion_error else None
    return replace(
        preferred,
        severity=severity,
        penalty=SEVERITY_PENALTY[severity],
        impact_tags=impact_tags,
        confidence=max(first.confidence, second.confidence),
        bbox_confidence=max(first.bbox_confidence, second.bbox_confidence),
        suggestion=suggestion,
        source=suggestion_error.source if suggestion_error else preferred.source,
    )


def _stable_error_id(task_id: str, error: VisualError) -> str:
    box = _affected_box(error)
    if box:
        y_bucket = ((box[0] + box[2]) // 2) // 100
        x_bucket = ((box[1] + box[3]) // 2) // 100
        anchor = f"{y_bucket}:{x_bucket}"
    elif error.region_id:
        anchor = str(error.region_id).replace(":", "_")
    else:
        anchor = "global"
    return f"{task_id}:layout:{error.type}:{anchor}"


def _spatial_sort_key(error: VisualError) -> tuple[int, int, int, str]:
    box = _affected_box(error)
    if box:
        return (box[0], box[1], -_SEVERITY_RANK[error.severity], error.error_id)
    return (1001, 1001, -_SEVERITY_RANK[error.severity], error.error_id)


def _affected_box(error: VisualError) -> list[int] | None:
    if not error.bbox_1000:
        return None
    return error.bbox_1000.get("affected_area")


def _iou(first: list[int], second: list[int]) -> float:
    ymin = max(first[0], second[0])
    xmin = max(first[1], second[1])
    ymax = min(first[2], second[2])
    xmax = min(first[3], second[3])
    intersection = max(0, ymax - ymin) * max(0, xmax - xmin)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
