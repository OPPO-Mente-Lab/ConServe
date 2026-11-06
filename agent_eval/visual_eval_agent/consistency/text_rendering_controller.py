# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..models import SEVERITY_PENALTY, Severity, VisualError


DECORATIVE_STYLE_TYPES = {
    "font_mismatch",
    "color_mismatch",
    "stroke_shadow_mismatch",
    "font_weight_mismatch",
    "font_color_mismatch",
}

FONT_SIZE_TYPES = {
    "font_size_inconsistency",
    "size_mismatch",
}


def calibrate_text_rendering_errors(errors: list[VisualError]) -> list[VisualError]:
    """Require direct visual evidence for residue and broken-stroke findings."""

    calibrated: list[VisualError] = []
    for error in errors:
        if _as_bool(error.comparison.get("bbox_evidence_match")) is False:
            continue

        if error.type == "source_text_residue":
            evidence = _nested(error.comparison, "residue_evidence")
            identifiable = _as_bool(evidence.get("identifiable_source_strokes"))
            matched_fragment = str(evidence.get("matched_source_fragment") or "").strip()
            visible_at_normal_scale = _as_bool(
                evidence.get("visible_at_normal_scale")
            )
            salient_without_bbox_guidance = _as_bool(
                evidence.get("salient_without_bbox_guidance")
            )
            contiguous_source_fragment = _as_bool(
                evidence.get("contiguous_source_fragment")
            )
            if not (
                identifiable is True
                and matched_fragment
                and visible_at_normal_scale is True
                and salient_without_bbox_guidance is True
                and contiguous_source_fragment is True
            ):
                continue

        if error.type == "broken_stroke":
            evidence = _nested(error.comparison, "glyph_damage_evidence")
            interrupted = _as_bool(
                evidence.get("target_glyph_interruption_visible")
            )
            capture_only = _as_bool(evidence.get("capture_artifact_only"))
            if interrupted is False or capture_only is True:
                continue
            major_supported = all(
                (
                    interrupted is True,
                    capture_only is False,
                    _as_bool(evidence.get("visible_at_normal_scale")) is True,
                    _as_bool(evidence.get("material_readability_impact")) is True,
                )
            )
            if error.severity == Severity.MAJOR and not major_supported:
                error = _with_severity(error, Severity.MINOR)

        if error.type == "too_small_to_read":
            evidence = _nested(error.comparison, "readability_evidence")
            source_readable = _as_bool(
                evidence.get("source_readable_at_normal_scale")
            )
            target_readable = _as_bool(
                evidence.get("target_readable_at_normal_scale")
            )
            introduced_reduction = _as_bool(
                evidence.get("translation_introduced_size_reduction")
            )
            unreadable_fragment = str(
                evidence.get("unreadable_target_fragment") or ""
            ).strip()
            if not (
                source_readable is True
                and target_readable is False
                and introduced_reduction is True
                and unreadable_fragment
            ):
                continue

        if error.type in {"missing_glyph", "garbled_text"}:
            if has_critical_font_fallback_evidence(
                error_type=error.type,
                comparison=error.comparison,
                evidence=error.evidence,
            ):
                error = _with_severity(error, Severity.CRITICAL)

        if error.type in DECORATIVE_STYLE_TYPES:
            evidence = _nested(error.comparison, "style_loss_evidence")
            no_material_impact = evidence and all(
                str(evidence.get(key) or "").strip().lower() == "none"
                for key in (
                    "readability_impact",
                    "hierarchy_impact",
                    "functional_state_impact",
                    "brand_identity_impact",
                )
            )
            if no_material_impact and error.severity != Severity.MINOR:
                error = _with_severity(error, Severity.MINOR)

        if (
            error.type in FONT_SIZE_TYPES
            and error.severity != Severity.MINOR
            and not supports_major_font_size(error.comparison)
        ):
            error = _with_severity(error, Severity.MINOR)

        calibrated.append(error)
    return calibrated


def has_critical_font_fallback_evidence(
    *,
    error_type: str,
    comparison: dict[str, Any],
    evidence: str,
) -> bool:
    """Return true only for an unrecoverable run of visible fallback glyphs."""

    if error_type not in {"missing_glyph", "garbled_text"}:
        return False

    fallback = _nested(comparison, "font_fallback_evidence")
    if fallback:
        try:
            placeholder_count = int(
                fallback.get("consecutive_placeholder_count") or 0
            )
        except (TypeError, ValueError):
            placeholder_count = 0
        return all(
            (
                _as_bool(fallback.get("placeholder_glyphs_visible")) is True,
                _as_bool(fallback.get("visible_at_normal_scale")) is True,
                placeholder_count >= 3,
                _as_bool(fallback.get("meaningful_text_region")) is True,
                _as_bool(fallback.get("intended_text_recoverable")) is False,
            )
        )

    normalized = str(evidence or "").strip().lower()
    placeholder_terms = ("带叉方框", "叉框", "方框占位", "tofu", "placeholder")
    scope_terms = ("一行", "整行", "整段", "多个", "连续")
    unreadable_terms = ("不可读", "无法识别", "无法表达", "无法恢复")
    return (
        any(term in normalized for term in placeholder_terms)
        and any(term in normalized for term in scope_terms)
        and any(term in normalized for term in unreadable_terms)
    )


def supports_major_font_size(comparison: dict[str, Any]) -> bool:
    """Return true only when a font-size defect has concrete material impact."""

    evidence = _nested(comparison, "typography_major_evidence")
    if not evidence:
        return False

    return any(
        (
            _as_bool(evidence.get("target_readable_at_normal_scale")) is False,
            _as_bool(evidence.get("characters_complete")) is False,
            _as_bool(evidence.get("content_association_preserved")) is False,
            _as_bool(evidence.get("overlap_or_clipping")) is True,
            _as_bool(evidence.get("functional_state_changed")) is True,
            _as_bool(evidence.get("brand_identity_lost")) is True,
        )
    )

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


def _with_severity(error: VisualError, severity: Severity) -> VisualError:
    return replace(error, severity=severity, penalty=SEVERITY_PENALTY[severity])
