# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..models import (
    BACKGROUND_IMPACT_TAGS,
    SEVERITY_PENALTY,
    Severity,
    VisualDimension,
    VisualError,
    ensure_unique_error_ids,
)


BACKGROUND_IMPACT_TAG_ALIASES = {
    "ghost_text": "local_scene_degradation",
    "diagram_structure_broken": "local_scene_degradation",
    "scene_element_damage": "local_scene_degradation",
    "ui_element_lost": "content_missing",
    "unnatural_appearance": "local_scene_degradation",
    "object_missing": "content_missing",
    "false_shadow": "false_content_introduced",
}


BACKGROUND_TYPE_ALIASES = {
    "smearing": "smearing_artifact",
    "smear": "smearing_artifact",
    "color_break": "color_discontinuity",
    "gradient_break": "gradient_discontinuity",
    "texture_break": "texture_discontinuity",
    "hard_edge": "hard_inpainting_edge",
    "residue": "inpainting_residue",
    "object_damage": "scene_element_damage",
    "icon_damage": "scene_element_damage",
    "missing_content": "missing_scene_content",
    "extra_content": "unexpected_generated_content",
    "repeated_pattern": "pattern_repetition_artifact",
}

CRITICAL_IMPACT_TAGS = {
    "primary_subject_unrecoverable",
    "key_icon_unrecognizable",
    "critical_region_untrustworthy",
    "scene_unusable",
}

SURFACE_REPAIR_TYPES = {
    "smearing_artifact",
    "color_discontinuity",
    "gradient_discontinuity",
    "texture_discontinuity",
    "hard_inpainting_edge",
    "inpainting_residue",
    "pattern_repetition_artifact",
}

CAPTURE_ARTIFACT_KINDS = {"moire", "scan_lines", "glare", "compression"}

TYPE_PRIORITY = {
    "scene_element_damage": 10,
    "missing_scene_content": 9,
    "unexpected_generated_content": 8,
    "smearing_artifact": 7,
    "hard_inpainting_edge": 6,
    "inpainting_residue": 5,
    "gradient_discontinuity": 4,
    "texture_discontinuity": 3,
    "pattern_repetition_artifact": 2,
    "color_discontinuity": 1,
}

SEVERITY_RANK = {
    Severity.MINOR: 0,
    Severity.MAJOR: 1,
    Severity.CRITICAL: 2,
}


def normalize_background_items(
    raw_items: list[dict[str, Any]],
    task_id: str,
    default_source: str,
    default_risk_level: str = "normal",
    calibrate_severity: bool = True,
) -> list[VisualError]:
    errors: list[VisualError] = []
    for raw_item in raw_items:
        item = dict(raw_item)
        item["type"] = BACKGROUND_TYPE_ALIASES.get(
            str(item.get("type")), str(item.get("type"))
        )
        item["bbox_1000"] = _ensure_affected_area(item.get("bbox_1000"))
        item["bbox_1000"] = _sanitize_model_bbox(item.get("bbox_1000"))
        item["severity_factors"] = _normalize_severity_factors(
            item.get("severity_factors")
        )
        if item["bbox_1000"] is None:
            item["localization_status"] = "global"
            item["bbox_confidence"] = 0.0
            item["confidence"] = min(_safe_confidence(item.get("confidence")), 0.6)
            item["needs_skill"] = True
        item["impact_tags"] = _normalize_impact_tags(item.get("impact_tags"))
        error = VisualError.from_model_dict(
            item,
            task_id,
            default_source,
            default_risk_level,
            dimension=VisualDimension.BACKGROUND,
        )
        errors.append(error)

    consolidated = consolidate_background_errors(errors)
    stable = [
        replace(error, error_id=_stable_error_id(task_id, error))
        for error in consolidated
    ]
    unique = ensure_unique_error_ids(stable)
    return calibrate_background_errors(unique) if calibrate_severity else unique


def calibrate_background_errors(errors: list[VisualError]) -> list[VisualError]:
    """Apply deterministic severity gates after model and Skill review."""

    calibrated: list[VisualError] = []
    for error in errors:
        if _comparison_bool(error.comparison.get("bbox_evidence_match")) is False:
            continue

        if error.type == "unexpected_generated_content":
            evidence = _nested(error.comparison, "unexpected_content_evidence")
            confirmed_new_content = all(
                (
                    bool(str(evidence.get("target_new_content") or "").strip()),
                    _comparison_bool(evidence.get("target_new_content_visible")) is True,
                    _comparison_bool(evidence.get("source_counterpart_absent")) is True,
                    _comparison_bool(evidence.get("bbox_evidence_match")) is True,
                )
            )
            if not confirmed_new_content:
                continue

        if error.type == "scene_element_damage":
            evidence = _nested(error.comparison, "scene_element_evidence")
            target_damage = _comparison_bool(
                evidence.get("target_element_missing_or_malformed")
            )
            if target_damage is False:
                continue
        calibrated_error = _calibrate_background_error(error)
        if is_imperceptible_surface_minor(calibrated_error):
            continue
        calibrated.append(calibrated_error)
    return calibrated


def is_imperceptible_surface_minor(error: VisualError) -> bool:
    """Return true for zoom-only cosmetic traces below the scoring floor."""

    return has_imperceptible_surface_trace_evidence(
        error_type=error.type,
        severity=error.severity,
        severity_factors=error.severity_factors,
        comparison=error.comparison,
        impact_tags=error.impact_tags,
    )


def has_imperceptible_surface_trace_evidence(
    *,
    error_type: str,
    severity: Severity | str,
    severity_factors: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    impact_tags: list[str] | None,
) -> bool:
    if (
        error_type not in SURFACE_REPAIR_TYPES
        or str(getattr(severity, "value", severity)) != Severity.MINOR.value
    ):
        return False

    factors = dict(severity_factors or {})
    comparison_data = dict(comparison or {})
    normalized_impact_tags = set(impact_tags or [])
    only_generic_surface_impact = normalized_impact_tags <= {
        "local_scene_degradation"
    }
    return all(
        (
            factors.get("visible_at_normal_scale") is False,
            factors.get("requires_side_by_side_comparison") is True,
            factors.get("readability_impact") == "none",
            factors.get("scene_element_impact") == "none",
            factors.get("operation_impact") == "none",
            factors.get("affected_scope") in {"small", "medium"},
            factors.get("contrast_with_surroundings") == "low",
            factors.get("source_background_overwritten") is False,
            only_generic_surface_impact,
            _comparison_bool(
                comparison_data.get("independent_non_text_damage")
            )
            is not True,
        )
    )


def _calibrate_background_error(error: VisualError) -> VisualError:
    severity = error.severity
    impact_tags = set(error.impact_tags)

    capture_artifact_kind = str(
        error.comparison.get("capture_artifact_kind") or ""
    ).strip().lower()
    independent_non_text_damage = _comparison_bool(
        error.comparison.get("independent_non_text_damage")
    )
    independent_scene_content = str(
        error.comparison.get("independent_scene_content") or ""
    ).strip().lower()
    scene_evidence = _nested(error.comparison, "scene_element_evidence")
    target_scene_damage = _comparison_bool(
        scene_evidence.get("target_element_missing_or_malformed")
    )
    artifact_terms = {
        "moire": ("moire",),
        "scan_lines": ("scan line", "refresh band"),
        "glare": ("glare", "reflection"),
        "compression": ("compression", "block artifact"),
    }
    meaningful_scene_terms = (
        "hair",
        "fabric",
        "icon",
        "object",
        "diagram",
        "photograph",
        "product",
        "subject",
        "outline",
    )
    describes_capture_artifact = any(
        term in independent_scene_content
        for term in artifact_terms.get(capture_artifact_kind, ())
    )
    artifact_only_description = bool(
        describes_capture_artifact
        and (
            target_scene_damage is False
            or not any(
                term in independent_scene_content for term in meaningful_scene_terms
            )
        )
    )
    if (
        severity == Severity.MAJOR
        and capture_artifact_kind in CAPTURE_ARTIFACT_KINDS
        and (
            independent_non_text_damage is not True
            or artifact_only_description
        )
    ):
        severity = Severity.MINOR

    if severity == Severity.MAJOR and error.type == "texture_discontinuity":
        texture_evidence = _nested(error.comparison, "texture_evidence")
        source_primitive = texture_evidence.get("source_primitive")
        source_visible = _comparison_bool(
            texture_evidence.get("source_primitive_visible")
        )
        if texture_evidence and (not source_primitive or source_visible is False):
            severity = Severity.MINOR

    if severity == Severity.CRITICAL and not (impact_tags & CRITICAL_IMPACT_TAGS):
        severity = Severity.MAJOR

    if severity == Severity.MAJOR and error.type in SURFACE_REPAIR_TYPES:
        factors = error.severity_factors
        visible_at_normal_scale = factors.get("visible_at_normal_scale") is True
        requires_comparison = factors.get("requires_side_by_side_comparison") is True
        high_contrast_fill_overwrite = all(
            (
                factors.get("rectangular_fill_visible") is True,
                factors.get("contrast_with_surroundings") == "high",
                factors.get("source_background_overwritten") is True,
            )
        )
        repeated_rectangular_fill = all(
            (
                factors.get("rectangular_fill_visible") is True,
                int(factors.get("repeated_rectangular_regions") or 0) >= 2,
                factors.get("source_background_overwritten") is True,
                factors.get("affected_scope") in {"medium", "large"},
                factors.get("contrast_with_surroundings") in {"medium", "high"},
            )
        )
        material_impact = any(
            (
                factors.get("readability_impact") == "material",
                factors.get("scene_element_impact") in {"material", "unrecoverable"},
                factors.get("operation_impact") == "material",
                factors.get("affected_scope") == "large",
                high_contrast_fill_overwrite,
                repeated_rectangular_fill,
                bool(
                    impact_tags
                    & {
                        "large_area_corruption",
                        "critical_region_untrustworthy",
                        "scene_unusable",
                    }
                ),
            )
        )
        if not visible_at_normal_scale or requires_comparison or not material_impact:
            severity = Severity.MINOR

    if severity == error.severity:
        return error
    return replace(error, severity=severity, penalty=SEVERITY_PENALTY[severity])


def _normalize_severity_factors(raw_factors: Any) -> dict[str, Any]:
    factors = dict(raw_factors or {})
    for key in (
        "visible_at_normal_scale",
        "requires_side_by_side_comparison",
        "rectangular_fill_visible",
        "source_background_overwritten",
    ):
        if key in factors:
            factors[key] = _normalize_boolean_factor(factors[key])

    if "repeated_rectangular_regions" in factors:
        try:
            factors["repeated_rectangular_regions"] = max(
                0, int(factors["repeated_rectangular_regions"])
            )
        except (TypeError, ValueError):
            factors["repeated_rectangular_regions"] = 0

    if "contrast_with_surroundings" in factors:
        contrast = str(factors["contrast_with_surroundings"]).strip().lower()
        factors["contrast_with_surroundings"] = (
            contrast if contrast in {"low", "medium", "high", "uncertain"} else "uncertain"
        )
    return factors


def _normalize_boolean_factor(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return "uncertain"


def _normalize_impact_tags(raw_tags: Any) -> list[str]:
    tags: list[str] = []
    for raw_tag in raw_tags or []:
        tag = BACKGROUND_IMPACT_TAG_ALIASES.get(str(raw_tag), str(raw_tag))
        if tag in BACKGROUND_IMPACT_TAGS and tag not in tags:
            tags.append(tag)
    return tags


def consolidate_background_errors(errors: list[VisualError]) -> list[VisualError]:
    ordered = sorted(errors, key=_spatial_sort_key)
    consolidated: list[VisualError] = []
    for error in ordered:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(consolidated)
                if _same_root_region(existing, error)
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


def _same_root_region(first: VisualError, second: VisualError) -> bool:
    same_type_or_surface_family = first.type == second.type or (
        first.type in SURFACE_REPAIR_TYPES and second.type in SURFACE_REPAIR_TYPES
    )
    if not same_type_or_surface_family:
        return False
    if first.region_id and second.region_id and first.region_id == second.region_id:
        return True
    first_box = _affected_box(first)
    second_box = _affected_box(second)
    threshold = 0.65 if first.type != second.type else 0.5
    return bool(
        first_box
        and second_box
        and (
            _iou(first_box, second_box) >= threshold
            or _overlap_over_smaller(first_box, second_box) >= 0.8
        )
    )


def _merge_errors(first: VisualError, second: VisualError) -> VisualError:
    preferred = max(
        (first, second),
        key=lambda error: (
            SEVERITY_RANK[error.severity],
            TYPE_PRIORITY.get(error.type, 0),
            error.confidence,
        ),
    )
    severity = max((first.severity, second.severity), key=SEVERITY_RANK.__getitem__)
    suggestion_error = next(
        (error for error in (preferred, first, second) if error.suggestion), None
    )
    return replace(
        preferred,
        severity=severity,
        penalty=SEVERITY_PENALTY[severity],
        impact_tags=list(dict.fromkeys(first.impact_tags + second.impact_tags)),
        confidence=max(first.confidence, second.confidence),
        bbox_confidence=max(first.bbox_confidence, second.bbox_confidence),
        suggestion=suggestion_error.suggestion if suggestion_error else None,
        source=suggestion_error.source if suggestion_error else preferred.source,
    )


def _ensure_affected_area(bbox_1000: Any) -> Any:
    if not isinstance(bbox_1000, dict) or "affected_area" in bbox_1000:
        return bbox_1000
    fallback = bbox_1000.get("scene_element_area") or bbox_1000.get("repair_area")
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
        return None
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
    return f"{task_id}:background:{error.type}:{anchor}"


def _spatial_sort_key(error: VisualError) -> tuple[int, int, int, str]:
    box = _affected_box(error)
    if box:
        return (box[0], box[1], -SEVERITY_RANK[error.severity], error.error_id)
    return (1001, 1001, -SEVERITY_RANK[error.severity], error.error_id)


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


def _overlap_over_smaller(first: list[int], second: list[int]) -> float:
    ymin = max(first[0], second[0])
    xmin = max(first[1], second[1])
    ymax = min(first[2], second[2])
    xmax = min(first[3], second[3])
    intersection = max(0, ymax - ymin) * max(0, xmax - xmin)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller else 0.0


def _comparison_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _nested(value: dict[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key)
    return dict(nested) if isinstance(nested, dict) else {}
