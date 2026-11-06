# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from ..consistency.layout_controller import orientation_anchor_status
from ..consistency.text_rendering_controller import supports_major_font_size
from ..llm_client import VisionModelClient
from ..models import (
    BACKGROUND_ERROR_TYPES,
    LAYOUT_ERROR_TYPES,
    TEXT_RENDERING_ERROR_TYPES,
    ImagePair,
    VisualError,
)
from ..prompt_loader import load_prompt
from ..runtime import VisionRuntimeConfig


ALLOWED_TYPES = {
    "visual.text_rendering": TEXT_RENDERING_ERROR_TYPES,
    "visual.layout": LAYOUT_ERROR_TYPES,
    "visual.background": BACKGROUND_ERROR_TYPES,
}

TYPE_ALIASES = {
    "visual.text_rendering": {
        "text_blur": "blurred_text",
        "garbled": "garbled_text",
        "font_size_mismatch": "size_mismatch",
    },
    "visual.layout": {
        "overflow": "text_overflow",
        "wrong_position": "position_shift",
        "occlusion": "content_occlusion",
    },
    "visual.background": {
        "smearing": "smearing_artifact",
        "hard_edge": "hard_inpainting_edge",
        "color_break": "color_discontinuity",
    },
}

SUPPRESSION_REASONS = {
    "duplicate",
    "secondary_impact",
    "wrong_dimension",
    "insufficient_evidence",
    "source_preserved_behavior",
}

CRITICAL_TAGS = {
    "key_content_hidden",
    "primary_content_unrecoverable",
    "page_unusable",
    "primary_subject_unrecoverable",
    "key_icon_unrecognizable",
    "critical_region_untrustworthy",
    "scene_unusable",
}

CRITICAL_TEXT_TYPES = {
    "garbled_text",
    "overlapping_text",
    "cropped_text",
    "missing_text",
    "missing_glyph",
    "too_small_to_read",
}

CRITICAL_GATE_TYPES = {
    "font_fallback_unrecoverable",
    "control_labels_unrecoverable",
    "control_mapping_unrecoverable",
    "critical_structured_data_unrecoverable",
    "primary_content_unrecoverable",
    "page_unusable",
}

SOURCE_TEXT_ERASURE_BACKGROUND_TYPES = {
    "smearing_artifact",
    "color_discontinuity",
    "hard_inpainting_edge",
    "inpainting_residue",
}

SAME_STYLE_TEXT_TYPES = {
    "font_mismatch",
    "size_mismatch",
    "color_mismatch",
    "stroke_shadow_mismatch",
    "font_size_inconsistency",
    "font_weight_mismatch",
    "font_color_mismatch",
}


class VisualCritic:
    """Model-backed adjudicator for post-Skill visual errors."""

    def __init__(
        self,
        model_client: VisionModelClient,
        runtime: VisionRuntimeConfig,
    ) -> None:
        self.model_client = model_client
        self.runtime = runtime

    def review(
        self,
        image_pair: ImagePair,
        errors: list[VisualError],
    ) -> dict[str, Any]:
        response = self.model_client.complete_json(
            "visual_critic",
            [
                {"role": "system", "content": load_prompt("visual_critic")},
                {"role": "user", "content": _critic_payload(image_pair, errors)},
            ],
            [image_pair.source_image, image_pair.translated_image],
            self.runtime,
        )
        return normalize_visual_critic_response(response, errors)


def normalize_visual_critic_response(
    response: dict[str, Any],
    candidates: list[VisualError],
) -> dict[str, Any]:
    candidate_by_id = {error.error_id: error for error in candidates}
    notes = [str(note) for note in response.get("notes", [])]
    severity_changes = [
        dict(change) for change in response.get("severity_changes", []) if isinstance(change, dict)
    ]
    adjudicated: list[dict[str, Any]] = []
    merged_ids: set[str] = set()

    for raw_error in response.get("adjudicated_errors", response.get("errors", [])):
        if not isinstance(raw_error, dict):
            continue
        raw_id = str(raw_error.get("error_id") or "")
        requested_ids = [
            str(error_id)
            for error_id in raw_error.get("merged_from", [])
            if str(error_id) in candidate_by_id and str(error_id) not in merged_ids
        ]
        if (
            raw_id in candidate_by_id
            and raw_id not in merged_ids
            and raw_id not in requested_ids
        ):
            requested_ids.append(raw_id)
        if not requested_ids:
            notes.append("Ignored an adjudicated error without a valid candidate source.")
            continue

        merged_candidates = [candidate_by_id[error_id] for error_id in requested_ids]
        primary_dimension = _primary_dimension(raw_error, merged_candidates)
        error_type = _normalized_type(
            primary_dimension,
            str(raw_error.get("type") or ""),
            merged_candidates,
        )
        severity = str(raw_error.get("severity") or merged_candidates[0].severity.value)
        if severity not in {"Minor", "Major", "Critical"}:
            severity = merged_candidates[0].severity.value
        v4_cap = _v4_severity_cap(primary_dimension, error_type, merged_candidates)
        if v4_cap == "Minor" and severity != "Minor":
            severity_changes.append(
                {
                    "candidate_error_id": requested_ids[0],
                    "from": severity,
                    "to": "Minor",
                    "reason": "V4 structured evidence caps this candidate at Minor.",
                }
            )
            severity = "Minor"
        impact_tags = list(
            dict.fromkeys(
                str(tag)
                for tag in raw_error.get("impact_tags", [])
                if str(tag).strip()
            )
        )
        evidence = str(raw_error.get("evidence") or merged_candidates[0].evidence)
        critical_gate = _normalize_critical_gate(raw_error.get("critical_gate"))
        if severity == "Critical" and not _supports_critical(
            primary_dimension,
            error_type,
            impact_tags,
            evidence,
            merged_candidates,
            critical_gate=critical_gate,
            gate_was_supplied="critical_gate" in raw_error,
        ):
            severity = "Major"
            critical_gate = None
            severity_changes.append(
                {
                    "candidate_error_id": requested_ids[0],
                    "from": "Critical",
                    "to": "Major",
                    "reason": "Critical requires an explicit unusable or untrustworthy visual consequence.",
                }
            )

        reference = _reference_candidate(primary_dimension, merged_candidates)
        bbox = reference.bbox_1000
        localization = reference.localization_status
        if localization == "global":
            bbox = None
        elif not _valid_bbox(bbox):
            critic_bbox = raw_error.get("bbox_1000")
            critic_localization = str(
                raw_error.get("localization_status") or "uncertain"
            )
            if critic_localization != "global" and _valid_bbox(critic_bbox):
                bbox = critic_bbox
                localization = critic_localization
            else:
                bbox = None
                localization = "global"

        impact_dimensions = [
            dimension
            for dimension in dict.fromkeys(
                str(value) for value in raw_error.get("impact_dimensions", [])
            )
            if dimension in ALLOWED_TYPES and dimension != primary_dimension
        ]
        confidence = _confidence(raw_error.get("confidence", reference.confidence))
        suggestion = raw_error.get("suggestion") or next(
            (
                candidate.suggestion
                for candidate in merged_candidates
                if candidate.suggestion
            ),
            None,
        )
        final_error = {
            "error_id": raw_id or requested_ids[0],
            "primary_dimension": primary_dimension,
            "type": error_type,
            "severity": severity,
            "bbox_1000": bbox,
            "localization_status": localization,
            "evidence": evidence,
            "impact_dimensions": impact_dimensions,
            "impact_tags": impact_tags,
            "merged_from": requested_ids,
            "confidence": confidence,
            "suggestion": suggestion,
        }
        if severity == "Critical" and critical_gate is not None:
            final_error["critical_gate"] = critical_gate
        duplicate = _find_duplicate(adjudicated, final_error)
        if duplicate is None:
            adjudicated.append(final_error)
        else:
            _merge_adjudicated_error(duplicate, final_error)
        merged_ids.update(requested_ids)

    suppressed: list[dict[str, Any]] = []
    suppressed_ids: set[str] = set()
    for raw_item in response.get("suppressed_errors", []):
        if not isinstance(raw_item, dict):
            continue
        candidate_id = str(raw_item.get("candidate_error_id") or "")
        if (
            candidate_id not in candidate_by_id
            or candidate_id in merged_ids
            or candidate_id in suppressed_ids
        ):
            continue
        reason = str(raw_item.get("reason") or "insufficient_evidence")
        if reason not in SUPPRESSION_REASONS:
            reason = "insufficient_evidence"
        suppressed.append(
            {
                "candidate_error_id": candidate_id,
                "reason": reason,
                "merged_into": raw_item.get("merged_into"),
            }
        )
        suppressed_ids.add(candidate_id)

    for candidate in candidates:
        if candidate.error_id in merged_ids or candidate.error_id in suppressed_ids:
            continue
        adjudicated.append(_candidate_fallback(candidate))
        merged_ids.add(candidate.error_id)
        notes.append(
            f"Candidate {candidate.error_id} was unaccounted for and retained unchanged."
        )

    _merge_same_erasure_failures(
        adjudicated,
        candidate_by_id,
        severity_changes,
        notes,
    )
    _merge_same_style_errors(adjudicated, candidate_by_id, notes)

    return {
        "adjudicated_errors": adjudicated,
        "suppressed_errors": suppressed,
        "severity_changes": severity_changes,
        "notes": notes,
    }


def _critic_payload(image_pair: ImagePair, errors: list[VisualError]) -> str:
    payload = {
        "task_id": image_pair.task_id,
        "source_language": image_pair.source_language,
        "target_language": image_pair.target_language,
        "output_language": image_pair.output_language,
        "risk_regions": image_pair.risk_regions,
        "candidate_errors": [error.to_dict() for error in errors],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _primary_dimension(
    raw_error: dict[str, Any],
    candidates: list[VisualError],
) -> str:
    requested = str(raw_error.get("primary_dimension") or "")
    if requested in ALLOWED_TYPES:
        return requested
    return candidates[0].dimension.value


def _normalized_type(
    dimension: str,
    raw_type: str,
    candidates: list[VisualError],
) -> str:
    normalized = TYPE_ALIASES.get(dimension, {}).get(raw_type, raw_type)
    if normalized in ALLOWED_TYPES[dimension]:
        return normalized
    for candidate in candidates:
        if candidate.dimension.value == dimension:
            return candidate.type
    return candidates[0].type


def _reference_candidate(
    dimension: str,
    candidates: list[VisualError],
) -> VisualError:
    return next(
        (candidate for candidate in candidates if candidate.dimension.value == dimension),
        candidates[0],
    )


def _supports_critical(
    dimension: str,
    error_type: str,
    impact_tags: list[str],
    evidence: str,
    candidates: list[VisualError],
    *,
    critical_gate: dict[str, Any] | None = None,
    gate_was_supplied: bool = False,
) -> bool:
    if critical_gate is not None:
        return True
    if gate_was_supplied:
        return False
    candidate_tags = {
        tag for candidate in candidates for tag in candidate.impact_tags
    }
    if (set(impact_tags) | candidate_tags) & CRITICAL_TAGS:
        return bool(evidence.strip())
    if dimension == "visual.text_rendering" and error_type in CRITICAL_TEXT_TYPES:
        return bool(
            evidence.strip()
            and any(candidate.severity.value == "Critical" for candidate in candidates)
        )
    return False


def _normalize_critical_gate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    gate_type = str(value.get("gate_type") or "").strip()
    normalized = {
        "triggered": _as_bool(value.get("triggered")),
        "gate_type": gate_type,
        "source_target_comparison_confirmed": _as_bool(
            value.get("source_target_comparison_confirmed")
        ),
        "visible_at_normal_scale": _as_bool(
            value.get("visible_at_normal_scale")
        ),
        "main_region_affected": _as_bool(value.get("main_region_affected")),
        "content_recoverable": _as_bool(value.get("content_recoverable")),
        "task_completable": _as_bool(value.get("task_completable")),
        "materially_untrustworthy": _as_bool(
            value.get("materially_untrustworthy")
        ),
    }
    if (
        normalized["triggered"] is not True
        or gate_type not in CRITICAL_GATE_TYPES
        or normalized["source_target_comparison_confirmed"] is not True
        or normalized["visible_at_normal_scale"] is not True
        or normalized["main_region_affected"] is not True
        or normalized["content_recoverable"] is not False
        or not (
            normalized["task_completable"] is False
            or normalized["materially_untrustworthy"] is True
        )
    ):
        return None
    return normalized


def _v4_severity_cap(
    dimension: str,
    error_type: str,
    candidates: list[VisualError],
) -> str | None:
    for candidate in candidates:
        if candidate.dimension.value != dimension:
            continue
        if dimension == "visual.layout" and error_type == "unreasonable_line_break":
            evidence = _nested(candidate.comparison, "line_break_evidence")
            if all(
                (
                    evidence.get("scope") in {"single_word", "short_phrase"},
                    _as_bool(evidence.get("characters_complete")) is True,
                    _as_bool(evidence.get("reading_order_preserved")) is True,
                    _as_bool(evidence.get("content_association_preserved")) is True,
                    _as_bool(evidence.get("overlap_or_clipping")) is False,
                    _as_bool(evidence.get("material_reconstruction_required")) is False,
                )
            ):
                return "Minor"
        if dimension == "visual.layout" and error_type == "reading_direction_mismatch":
            if orientation_anchor_status(candidate.comparison) != "confirmed_mismatch":
                return "Minor"
        if dimension == "visual.text_rendering" and error_type in SAME_STYLE_TEXT_TYPES:
            if error_type in {"font_size_inconsistency", "size_mismatch"}:
                if not supports_major_font_size(candidate.comparison):
                    return "Minor"

            evidence = _nested(candidate.comparison, "style_loss_evidence")
            if evidence and all(
                str(evidence.get(key) or "").strip().lower() == "none"
                for key in (
                    "readability_impact",
                    "hierarchy_impact",
                    "functional_state_impact",
                    "brand_identity_impact",
                )
            ):
                return "Minor"
        if dimension == "visual.background" and error_type == "texture_discontinuity":
            evidence = _nested(candidate.comparison, "texture_evidence")
            if evidence and (
                not evidence.get("source_primitive")
                or _as_bool(evidence.get("source_primitive_visible")) is False
            ):
                return "Minor"
    return None


def _nested(value: dict[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key)
    return dict(nested) if isinstance(nested, dict) else {}


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, dict) or "affected_area" not in value:
        return False
    coordinates = value["affected_area"]
    return bool(
        isinstance(coordinates, list)
        and len(coordinates) == 4
        and all(isinstance(number, int) and not isinstance(number, bool) for number in coordinates)
        and all(0 <= number <= 1000 for number in coordinates)
        and coordinates[0] < coordinates[2]
        and coordinates[1] < coordinates[3]
    )


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _find_duplicate(
    adjudicated: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    candidate_key = _spatial_error_key(candidate)
    for existing in adjudicated:
        if existing["error_id"] == candidate["error_id"]:
            return existing
        if candidate_key is not None and _spatial_error_key(existing) == candidate_key:
            return existing
    return None


def _spatial_error_key(error: dict[str, Any]) -> tuple[Any, ...] | None:
    bbox = error.get("bbox_1000")
    area = bbox.get("affected_area") if isinstance(bbox, dict) else None
    if not isinstance(area, list) or len(area) != 4:
        return None
    return (
        error.get("primary_dimension"),
        error.get("type"),
        tuple(area),
    )


def _merge_adjudicated_error(
    target: dict[str, Any],
    duplicate: dict[str, Any],
) -> None:
    for field in ("merged_from", "impact_dimensions", "impact_tags"):
        target[field] = list(dict.fromkeys([*target.get(field, []), *duplicate.get(field, [])]))
    severity_rank = {"Minor": 1, "Major": 2, "Critical": 3}
    if severity_rank.get(duplicate["severity"], 0) > severity_rank.get(target["severity"], 0):
        target["severity"] = duplicate["severity"]
    target["confidence"] = max(target.get("confidence", 0.0), duplicate.get("confidence", 0.0))
    if not target.get("suggestion") and duplicate.get("suggestion"):
        target["suggestion"] = duplicate["suggestion"]


def _candidate_fallback(candidate: VisualError) -> dict[str, Any]:
    return {
        "error_id": candidate.error_id,
        "primary_dimension": candidate.dimension.value,
        "type": candidate.type,
        "severity": candidate.severity.value,
        "bbox_1000": candidate.bbox_1000,
        "localization_status": candidate.localization_status,
        "evidence": candidate.evidence,
        "impact_dimensions": [],
        "impact_tags": list(candidate.impact_tags),
        "merged_from": [candidate.error_id],
        "confidence": candidate.confidence,
        "suggestion": candidate.suggestion,
    }


def _merge_same_erasure_failures(
    adjudicated: list[dict[str, Any]],
    candidate_by_id: dict[str, VisualError],
    severity_changes: list[dict[str, Any]],
    notes: list[str],
) -> None:
    """Conservatively merge explicitly classified same-footprint erasure symptoms."""

    residue_errors = [
        error
        for error in adjudicated
        if error.get("primary_dimension") == "visual.text_rendering"
        and error.get("type") == "source_text_residue"
    ]
    if not residue_errors:
        return

    remove_ids: set[int] = set()
    for background_error in adjudicated:
        if background_error.get("primary_dimension") != "visual.background":
            continue
        if background_error.get("type") not in SOURCE_TEXT_ERASURE_BACKGROUND_TYPES:
            continue

        merged_candidates = [
            candidate_by_id[error_id]
            for error_id in background_error.get("merged_from", [])
            if error_id in candidate_by_id
        ]
        if not merged_candidates or not all(
            _is_same_footprint_erasure_symptom(candidate)
            for candidate in merged_candidates
        ):
            continue

        residue_error = next(
            (
                error
                for error in residue_errors
                if _overlap_over_smaller(error, background_error) >= 0.5
            ),
            None,
        )
        if residue_error is None:
            continue

        previous_severity = str(background_error.get("severity") or "")
        retained_severity = str(residue_error.get("severity") or "")
        residue_error["merged_from"] = list(
            dict.fromkeys(
                [
                    *residue_error.get("merged_from", []),
                    *background_error.get("merged_from", []),
                ]
            )
        )
        residue_error["impact_dimensions"] = list(
            dict.fromkeys(
                [*residue_error.get("impact_dimensions", []), "visual.background"]
            )
        )
        residue_error["confidence"] = max(
            float(residue_error.get("confidence") or 0.0),
            float(background_error.get("confidence") or 0.0),
        )
        if not residue_error.get("suggestion") and background_error.get("suggestion"):
            residue_error["suggestion"] = background_error["suggestion"]
        remove_ids.add(id(background_error))
        if previous_severity and previous_severity != retained_severity:
            severity_changes.append(
                {
                    "candidate_error_id": background_error.get("error_id"),
                    "from": previous_severity,
                    "to": retained_severity,
                    "reason": (
                        "Merged a same-footprint Background symptom into the "
                        "source-text-residue root cause; severity follows the "
                        "retained root cause rather than the highest candidate."
                    ),
                }
            )
        notes.append(
            "Merged a same-footprint Background erasure symptom into "
            f"{residue_error.get('error_id')}."
        )

    if remove_ids:
        adjudicated[:] = [error for error in adjudicated if id(error) not in remove_ids]


def _is_same_footprint_erasure_symptom(candidate: VisualError) -> bool:
    if candidate.dimension.value != "visual.background":
        return False
    comparison = candidate.comparison
    return bool(
        comparison.get("relation_to_source_text_footprint") == "inside"
        and _as_bool(comparison.get("recognizable_source_glyphs")) is True
        and _as_bool(comparison.get("independent_non_text_damage")) is False
        and _as_bool(comparison.get("independent_repair_required")) is False
    )


def _merge_same_style_errors(
    adjudicated: list[dict[str, Any]],
    candidate_by_id: dict[str, VisualError],
    notes: list[str],
) -> None:
    """Merge repeated typography symptoms only when candidates declare one cause."""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    remove_ids: set[int] = set()
    for error in adjudicated:
        if error.get("primary_dimension") != "visual.text_rendering":
            continue
        error_type = str(error.get("type") or "")
        if error_type not in SAME_STYLE_TEXT_TYPES:
            continue
        candidates = [
            candidate_by_id[error_id]
            for error_id in error.get("merged_from", [])
            if error_id in candidate_by_id
        ]
        group_ids = {
            str(candidate.comparison.get("style_group_id") or "").strip()
            for candidate in candidates
            if _as_bool(candidate.comparison.get("shared_rendering_cause")) is True
        }
        group_ids.discard("")
        if len(group_ids) != 1:
            continue
        key = (error_type, next(iter(group_ids)))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = error
            continue
        _merge_adjudicated_error(existing, error)
        remove_ids.add(id(error))
        notes.append(
            f"Merged repeated {error_type} errors from style group {key[1]}."
        )
    if remove_ids:
        adjudicated[:] = [error for error in adjudicated if id(error) not in remove_ids]


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _overlap_over_smaller(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    first_area = _affected_area(first)
    second_area = _affected_area(second)
    if first_area is None or second_area is None:
        return 0.0
    y1 = max(first_area[0], second_area[0])
    x1 = max(first_area[1], second_area[1])
    y2 = min(first_area[2], second_area[2])
    x2 = min(first_area[3], second_area[3])
    if y1 >= y2 or x1 >= x2:
        return 0.0
    intersection = (y2 - y1) * (x2 - x1)
    first_size = (first_area[2] - first_area[0]) * (first_area[3] - first_area[1])
    second_size = (second_area[2] - second_area[0]) * (second_area[3] - second_area[1])
    return intersection / min(first_size, second_size)


def _affected_area(error: dict[str, Any]) -> list[int] | None:
    bbox = error.get("bbox_1000")
    area = bbox.get("affected_area") if isinstance(bbox, dict) else None
    if not isinstance(area, list) or len(area) != 4:
        return None
    return area
