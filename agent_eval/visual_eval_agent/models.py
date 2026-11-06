# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(str, Enum):
    MINOR = "Minor"
    MAJOR = "Major"
    CRITICAL = "Critical"


class VisualDimension(str, Enum):
    TEXT_RENDERING = "visual.text_rendering"
    LAYOUT = "visual.layout"
    BACKGROUND = "visual.background"


SEVERITY_PENALTY = {
    Severity.MINOR: 1,
    Severity.MAJOR: 5,
    Severity.CRITICAL: 25,
}


TEXT_RENDERING_ERROR_TYPES = {
    "blurred_text",
    "jagged_text",
    "too_small_to_read",
    "cropped_text",
    "missing_text",
    "missing_glyph",
    "broken_stroke",
    "overlapping_text",
    "source_text_residue",
    "garbled_text",
    "font_mismatch",
    "size_mismatch",
    "color_mismatch",
    "stroke_shadow_mismatch",
    "font_size_inconsistency",
    "font_weight_mismatch",
    "font_color_mismatch",
}


LAYOUT_ERROR_TYPES = {
    "position_shift",
    "alignment_error",
    "text_overflow",
    "unreasonable_line_break",
    "reading_direction_mismatch",
    "reading_order_error",
    "text_region_crowding",
    "content_occlusion",
    "wrong_region_fill",
}


LAYOUT_IMPACT_TAGS = {
    "text_overflow",
    "content_occlusion",
    "requires_head_rotation",
    "disrupted_reading_flow",
    "rereading_required",
    "partial_content_hidden",
    "wrong_content_association",
    "key_content_hidden",
    "primary_content_unrecoverable",
    "page_unusable",
}


BACKGROUND_ERROR_TYPES = {
    "smearing_artifact",
    "color_discontinuity",
    "gradient_discontinuity",
    "texture_discontinuity",
    "hard_inpainting_edge",
    "inpainting_residue",
    "scene_element_damage",
    "missing_scene_content",
    "unexpected_generated_content",
    "pattern_repetition_artifact",
}


BACKGROUND_IMPACT_TAGS = {
    "local_scene_degradation",
    "large_area_corruption",
    "content_missing",
    "false_content_introduced",
    "primary_subject_unrecoverable",
    "key_icon_unrecognizable",
    "critical_region_untrustworthy",
    "scene_unusable",
}


@dataclass(frozen=True)
class ImagePair:
    task_id: str
    source_image: Path
    translated_image: Path
    source_language: str | None = None
    target_language: str | None = None
    output_language: str = "zh-CN"
    original_text_regions: list[dict[str, Any]] = field(default_factory=list)
    translated_text_regions: list[dict[str, Any]] = field(default_factory=list)
    risk_regions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> "ImagePair":
        source_value = request.get("source_image") or request.get("original_image")
        translated_value = request.get("translated_image")
        if not source_value:
            raise ValueError("source_image is required")
        if not translated_value:
            raise ValueError("translated_image is required")
        source = Path(source_value).expanduser().resolve()
        translated = Path(translated_value).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"source_image does not exist: {source}")
        if not translated.is_file():
            raise ValueError(f"translated_image does not exist: {translated}")
        if source == translated:
            raise ValueError("source_image and translated_image must be distinct files")
        return cls(
            task_id=str(request.get("task_id") or source.stem),
            source_image=source,
            translated_image=translated,
            source_language=request.get("source_language"),
            target_language=request.get("target_language"),
            output_language=str(request.get("output_language") or "zh-CN"),
            original_text_regions=list(request.get("original_text_regions") or []),
            translated_text_regions=list(request.get("translated_text_regions") or []),
            risk_regions=list(request.get("risk_regions") or []),
            metadata=dict(request.get("metadata") or {}),
        )


@dataclass
class VisualError:
    error_id: str
    dimension: VisualDimension
    type: str
    severity: Severity
    evidence: str
    region_id: str | None = None
    bbox_1000: dict[str, list[int]] | None = None
    bbox_confidence: float = 0.0
    localization_status: str = "uncertain"
    confidence: float = 0.8
    risk_level: str = "normal"
    impact_tags: list[str] = field(default_factory=list)
    suggestion: str | None = None
    source: str = "unknown"
    needs_skill: bool = False
    penalty: int | None = None
    comparison: dict[str, Any] = field(default_factory=dict)
    severity_factors: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, VisualDimension):
            self.dimension = VisualDimension(self.dimension)
        if not isinstance(self.severity, Severity):
            self.severity = Severity(self.severity)
        allowed_types = {
            VisualDimension.TEXT_RENDERING: TEXT_RENDERING_ERROR_TYPES,
            VisualDimension.LAYOUT: LAYOUT_ERROR_TYPES,
            VisualDimension.BACKGROUND: BACKGROUND_ERROR_TYPES,
        }[self.dimension]
        if self.type not in allowed_types:
            label = {
                VisualDimension.TEXT_RENDERING: "Text Rendering",
                VisualDimension.LAYOUT: "Layout",
                VisualDimension.BACKGROUND: "Background",
            }[self.dimension]
            raise ValueError(f"Unsupported {label} error type: {self.type}")
        self.impact_tags = list(dict.fromkeys(str(tag) for tag in self.impact_tags))
        if self.dimension == VisualDimension.LAYOUT:
            unknown_tags = [tag for tag in self.impact_tags if tag not in LAYOUT_IMPACT_TAGS]
            if unknown_tags:
                raise ValueError(f"Unsupported Layout impact tag: {unknown_tags[0]}")
        elif self.dimension == VisualDimension.BACKGROUND:
            self.impact_tags = [
                tag for tag in self.impact_tags if tag in BACKGROUND_IMPACT_TAGS
            ]
        elif self.impact_tags:
            raise ValueError(
                "impact_tags are only supported for Layout or Background errors"
            )
        self.comparison = dict(self.comparison or {})
        self.severity_factors = dict(self.severity_factors or {})
        self.confidence = _confidence(self.confidence, "confidence")
        self.bbox_confidence = _confidence(self.bbox_confidence, "bbox_confidence")
        if self.localization_status not in {"exact", "approximate", "uncertain", "global"}:
            raise ValueError(f"Invalid localization_status: {self.localization_status}")
        self.bbox_1000 = _validate_bbox_1000(self.bbox_1000, self.localization_status)
        if self.penalty is None:
            self.penalty = SEVERITY_PENALTY[self.severity]

    @classmethod
    def from_model_dict(
        cls,
        data: dict[str, Any],
        task_id: str,
        default_source: str,
        default_risk_level: str = "normal",
        dimension: VisualDimension = VisualDimension.TEXT_RENDERING,
    ) -> "VisualError":
        error_type = str(data["type"])
        dimension = VisualDimension(dimension)
        dimension_name = dimension.value.rsplit(".", 1)[-1]
        impact_tags = list(data.get("impact_tags") or [])
        if dimension == VisualDimension.BACKGROUND:
            impact_tags = [
                str(tag) for tag in impact_tags if str(tag) in BACKGROUND_IMPACT_TAGS
            ]
        return cls(
            error_id=str(data.get("error_id") or f"{task_id}:{dimension_name}:{error_type}"),
            dimension=dimension,
            type=error_type,
            severity=Severity(data["severity"]),
            evidence=str(data.get("evidence") or ""),
            region_id=data.get("region_id"),
            bbox_1000=data.get("bbox_1000"),
            bbox_confidence=float(data.get("bbox_confidence", 0.0)),
            localization_status=str(data.get("localization_status") or "uncertain"),
            confidence=float(data.get("confidence", 0.8)),
            risk_level=str(data.get("risk_level") or default_risk_level),
            impact_tags=impact_tags,
            suggestion=data.get("suggestion"),
            source=str(data.get("source") or default_source),
            needs_skill=bool(data.get("needs_skill", False)),
            comparison=dict(data.get("comparison") or {}),
            severity_factors=dict(data.get("severity_factors") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "error_id": self.error_id,
            "branch": "visual",
            "dimension": self.dimension.value,
            "type": self.type,
            "severity": self.severity.value,
            "penalty": self.penalty,
            "region_id": self.region_id,
            "bbox_1000": self.bbox_1000,
            "bbox_confidence": self.bbox_confidence,
            "localization_status": self.localization_status,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "impact_tags": list(self.impact_tags),
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "source": self.source,
            "needs_skill": self.needs_skill,
        }
        if self.comparison:
            payload["comparison"] = dict(self.comparison)
        if self.severity_factors:
            payload["severity_factors"] = dict(self.severity_factors)
        return payload


def ensure_unique_error_ids(errors: list[VisualError]) -> list[VisualError]:
    seen: dict[str, int] = {}
    result: list[VisualError] = []
    for error in errors:
        count = seen.get(error.error_id, 0) + 1
        seen[error.error_id] = count
        result.append(error if count == 1 else replace(error, error_id=f"{error.error_id}:{count}"))
    return result


def _confidence(value: float, name: str) -> float:
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _validate_bbox_1000(
    bbox_1000: dict[str, list[int]] | None,
    localization_status: str,
) -> dict[str, list[int]] | None:
    if localization_status == "global":
        if bbox_1000 is not None:
            raise ValueError("bbox_1000 must be null when localization_status is global")
        return None
    if bbox_1000 is None:
        return None
    if not isinstance(bbox_1000, dict) or "affected_area" not in bbox_1000:
        raise ValueError("bbox_1000 must contain affected_area")
    for name, coordinates in bbox_1000.items():
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinates)
            or any(value < 0 or value > 1000 for value in coordinates)
            or coordinates[0] >= coordinates[2]
            or coordinates[1] >= coordinates[3]
        ):
            raise ValueError(
                f"Invalid bbox_1000 for {name}; expected [ymin, xmin, ymax, xmax]"
            )
    return bbox_1000
