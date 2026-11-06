# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from enum import Enum
from typing import Any

from .policy import affected_fragment_key, locate_fragment


class Severity(str, Enum):
    MINOR = "Minor"
    MAJOR = "Major"
    CRITICAL = "Critical"


class TextDimension(str, Enum):
    FAITHFULNESS = "text.faithfulness"
    FLUENCY = "text.fluency"


SEVERITY_PENALTY = {
    Severity.MINOR: 1,
    Severity.MAJOR: 5,
    Severity.CRITICAL: 25,
}


@dataclass
class TextPair:
    pair_id: str
    source_text: str
    translated_text: str
    source_language: str | None = None
    target_language: str | None = None
    source_bbox: list[int] | None = None
    translated_bbox: list[int] | None = None
    pair_confidence: float = 1.0
    source_ocr_confidence: float | None = None
    translated_ocr_confidence: float | None = None
    risk_level: str = "normal"
    risk_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextPair":
        source = data.get("source")
        translated = data.get("translated")
        if isinstance(source, dict):
            source_text = source.get("text", "")
            source_bbox = source.get("bbox")
            source_ocr_confidence = source.get("ocr_confidence")
        else:
            source_text = data.get("source_text", "")
            source_bbox = data.get("source_bbox")
            source_ocr_confidence = data.get("source_ocr_confidence")

        if isinstance(translated, dict):
            translated_text = translated.get("text", "")
            translated_bbox = translated.get("bbox")
            translated_ocr_confidence = translated.get("ocr_confidence")
        else:
            translated_text = data.get("translated_text", "")
            translated_bbox = data.get("translated_bbox")
            translated_ocr_confidence = data.get("translated_ocr_confidence")

        return cls(
            pair_id=str(data["pair_id"]),
            source_text=str(source_text or ""),
            translated_text=str(translated_text or ""),
            source_language=data.get("source_language"),
            target_language=data.get("target_language"),
            source_bbox=source_bbox,
            translated_bbox=translated_bbox,
            pair_confidence=float(data.get("pair_confidence", 1.0)),
            source_ocr_confidence=source_ocr_confidence,
            translated_ocr_confidence=translated_ocr_confidence,
            risk_level=data.get("risk_level", "normal"),
            risk_type=data.get("risk_type"),
            metadata=data.get("metadata", {}),
        )


ALLOWED_IMAGE_CONTEXT_KEYS = {
    "scene_type",
    "document_type",
    "domain",
    "user_intent",
    "context_summary",
    "terminology_clues",
    "surrounding_text_clues",
    "confidence",
}


def image_context_for_prompt(pair: TextPair) -> dict[str, Any] | None:
    value = pair.metadata.get("image_context")
    if not isinstance(value, dict):
        return None
    context = {key: value[key] for key in ALLOWED_IMAGE_CONTEXT_KEYS if key in value}
    return context or None


@dataclass
class TextError:
    error_id: str
    dimension: TextDimension
    type: str
    severity: Severity
    pair_id: str
    evidence: str
    source_fragment: str | None = None
    target_fragment: str | None = None
    suggestion: str | None = None
    penalty: int | None = None
    region_id: str | None = None
    bbox: dict[str, list[int]] | None = None
    confidence: float = 0.8
    risk_level: str = "normal"
    source: str = "unknown"
    needs_skill: bool = False
    source_fragment_span: dict[str, int] | None = None
    target_fragment_span: dict[str, int] | None = None
    affected_fragment_key: str | None = None
    related_evidence: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, TextDimension):
            self.dimension = TextDimension(self.dimension)
        if not isinstance(self.severity, Severity):
            self.severity = Severity(self.severity)
        if self.penalty is None:
            self.penalty = SEVERITY_PENALTY[self.severity]

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_id": self.error_id,
            "branch": "text",
            "dimension": self.dimension.value,
            "type": self.type,
            "severity": self.severity.value,
            "penalty": self.penalty,
            "pair_id": self.pair_id,
            "region_id": self.region_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "evidence": self.evidence,
            "source_fragment": self.source_fragment,
            "target_fragment": self.target_fragment,
            "suggestion": self.suggestion,
            "source": self.source,
            "needs_skill": self.needs_skill,
            "source_fragment_span": self.source_fragment_span,
            "target_fragment_span": self.target_fragment_span,
            "affected_fragment_key": self.affected_fragment_key,
            "related_evidence": self.related_evidence,
        }

    @classmethod
    def from_model_dict(
        cls,
        data: dict[str, Any],
        pair: TextPair | None,
        default_dimension: TextDimension,
        default_source: str,
    ) -> "TextError":
        pair_id = str(data.get("pair_id") or (pair.pair_id if pair else "unknown_pair"))
        error_type = str(data["type"])
        dimension = TextDimension(data.get("dimension", default_dimension.value))
        severity = Severity(data["severity"])
        error_id = str(data.get("error_id") or f"{pair_id}:{dimension.value.split('.')[-1]}:{error_type}")
        bbox = data.get("bbox")
        if bbox is None and pair and pair.source_bbox:
            bbox = {"source_text_area": pair.source_bbox}
        if pair and pair.translated_bbox:
            bbox = dict(bbox or {})
            bbox.setdefault("translated_text_area", pair.translated_bbox)
        source_fragment = data.get("source_fragment")
        target_fragment = data.get("target_fragment")
        source_span = locate_fragment(pair.source_text, source_fragment) if pair else None
        target_span = locate_fragment(pair.translated_text, target_fragment) if pair else None
        return cls(
            error_id=error_id,
            dimension=dimension,
            type=error_type,
            severity=severity,
            pair_id=pair_id,
            evidence=str(data.get("evidence", "")),
            source_fragment=source_fragment,
            target_fragment=target_fragment,
            suggestion=data.get("suggestion"),
            region_id=data.get("region_id"),
            bbox=bbox,
            confidence=float(data.get("confidence", 0.8)),
            risk_level=str(data.get("risk_level") or (pair.risk_level if pair else "normal")),
            source=str(data.get("source", default_source)),
            needs_skill=bool(data.get("needs_skill", severity in {Severity.MAJOR, Severity.CRITICAL})),
            source_fragment_span=source_span,
            target_fragment_span=target_span,
            affected_fragment_key=affected_fragment_key(source_span, target_span),
            related_evidence=list(data.get("related_evidence", [])),
        )


def refresh_fragment_spans(error: TextError, pair: TextPair) -> TextError:
    source_span = locate_fragment(pair.source_text, error.source_fragment)
    target_span = locate_fragment(pair.translated_text, error.target_fragment)
    return replace(
        error,
        source_fragment_span=source_span,
        target_fragment_span=target_span,
        affected_fragment_key=affected_fragment_key(source_span, target_span),
    )


@dataclass
class TextJudgeResult:
    errors: list[TextError]
    deduplicated_errors: list[dict[str, Any]] = field(default_factory=list)
    severity_updates: list[dict[str, Any]] = field(default_factory=list)
    caps: list[dict[str, Any]] = field(default_factory=list)


def ensure_unique_error_ids(errors: list[TextError]) -> list[TextError]:
    seen: dict[str, int] = {}
    unique_errors: list[TextError] = []
    for error in errors:
        count = seen.get(error.error_id, 0) + 1
        seen[error.error_id] = count
        if count == 1:
            unique_errors.append(error)
            continue
        unique_errors.append(replace(error, error_id=f"{error.error_id}:{count}"))
    return unique_errors

