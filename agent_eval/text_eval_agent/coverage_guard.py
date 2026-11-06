# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any

from .image_models import ImageTextBlock, ImageTextExtraction
from .models import Severity, TextDimension, TextError, TextPair
from .suggestion_format import format_suggestion


MIN_CLEAR_UNMATCHED_BLOCKS = 5
MIN_UNMATCHED_BLOCK_RATIO = 0.70
MIN_UNMATCHED_CHARACTER_RATIO = 0.70
MAX_TARGET_TO_SOURCE_BLOCK_RATIO = 0.40
MIN_PRIMARY_TEXT_CHARACTERS = 80
MIN_RELIABLE_PARTIAL_CONFIDENCE = 0.90
MIN_UNCHANGED_PARAGRAPH_CHARACTERS = 60
MIN_LATIN_LETTERS = 45
MIN_LATIN_LETTER_RATIO = 0.75
MIN_UNCHANGED_SIMILARITY = 0.96
PRIMARY_BLOCK_TYPES = {
    "article",
    "body",
    "body_text",
    "long_text",
    "paragraph",
}


@dataclass
class TextCoverageAudit:
    metrics: dict[str, Any]
    direct_critical: bool = False
    trigger_reasons: list[str] = field(default_factory=list)
    pairs: list[TextPair] = field(default_factory=list)
    errors: list[TextError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "direct_critical": self.direct_critical,
            "trigger_reasons": list(self.trigger_reasons),
            "coverage_pair_ids": [pair.pair_id for pair in self.pairs],
            "coverage_error_ids": [error.error_id for error in self.errors],
        }


def audit_text_coverage(
    task_id: str,
    extraction: ImageTextExtraction,
    *,
    pairs: list[TextPair] | None = None,
    source_language: str | None = None,
    target_language: str | None = None,
) -> TextCoverageAudit:
    reliable_source = [
        block for block in extraction.source_blocks if _is_reliably_readable(block)
    ]
    unmatched_ids = set(extraction.unmatched_source_block_ids)
    reliable_unmatched = [
        block for block in reliable_source if block.block_id in unmatched_ids
    ]
    source_characters = sum(_character_count(block.text) for block in reliable_source)
    unmatched_characters = sum(
        _character_count(block.text) for block in reliable_unmatched
    )
    source_block_count = len(reliable_source)
    unmatched_block_count = len(reliable_unmatched)
    unmatched_block_ratio = _ratio(unmatched_block_count, source_block_count)
    unmatched_character_ratio = _ratio(
        unmatched_characters, source_characters
    )
    target_to_source_block_ratio = _ratio(
        len(extraction.target_blocks), source_block_count
    )
    missing_primary = [
        block for block in reliable_unmatched if _is_primary_content(block)
    ]

    metrics = {
        "clear_source_block_count": sum(
            block.visibility == "clear" for block in extraction.source_blocks
        ),
        "reliable_source_block_count": source_block_count,
        "target_block_count": len(extraction.target_blocks),
        "unmatched_source_block_count": unmatched_block_count,
        "source_character_count": source_characters,
        "unmatched_source_character_count": unmatched_characters,
        "unmatched_block_ratio": round(unmatched_block_ratio, 4),
        "unmatched_character_ratio": round(unmatched_character_ratio, 4),
        "target_to_source_block_ratio": round(target_to_source_block_ratio, 4),
        "missing_primary_block_ids": [block.block_id for block in missing_primary],
    }
    unchanged_errors = _audit_unchanged_source_language_paragraphs(
        pairs or [],
        source_language=source_language,
        target_language=target_language,
    )
    metrics["unchanged_source_language_pair_count"] = len(unchanged_errors)
    metrics["unchanged_source_language_pair_ids"] = [
        error.pair_id for error in unchanged_errors
    ]
    reasons = _trigger_reasons(
        unmatched_block_count,
        unmatched_block_ratio,
        unmatched_character_ratio,
        target_to_source_block_ratio,
        missing_primary,
    )
    if len(reasons) < 5:
        return TextCoverageAudit(
            metrics=metrics,
            trigger_reasons=reasons,
            errors=unchanged_errors,
        )

    primary = max(missing_primary, key=lambda block: _character_count(block.text))
    pair_id = f"{task_id}:coverage_critical_01"
    pair = TextPair(
        pair_id=pair_id,
        source_text=primary.text,
        translated_text="",
        source_language=primary.language,
        target_language=None,
        source_bbox=list(primary.bbox_1000),
        translated_bbox=None,
        pair_confidence=min(primary.confidence, 0.99),
        source_ocr_confidence=primary.confidence,
        translated_ocr_confidence=None,
        metadata={
            "unit_type": "coverage_omission",
            "alignment_type": "source_only",
            "source_block_ids": [block.block_id for block in reliable_unmatched],
            "target_block_ids": [],
            "coverage_gate": "primary_content_missing",
            "coverage_metrics": metrics,
        },
    )
    error = TextError(
        error_id=f"{pair_id}:faithfulness:omission",
        dimension=TextDimension.FAITHFULNESS,
        type="omission",
        severity=Severity.CRITICAL,
        pair_id=pair_id,
        evidence=(
            "原图主要正文及大多数清晰可翻译文字在译图中均无对应内容，"
            "主要信息无法阅读。"
        ),
        source_fragment=primary.text,
        target_fragment="",
        suggestion=format_suggestion(primary.text, "[补全译文]"),
        bbox={"source_text_area": list(primary.bbox_1000)},
        confidence=min(primary.confidence, 0.99),
        source="text_coverage_guard",
        needs_skill=False,
    )
    return TextCoverageAudit(
        metrics=metrics,
        direct_critical=True,
        trigger_reasons=reasons,
        pairs=[pair],
        errors=[error, *unchanged_errors],
    )


def _audit_unchanged_source_language_paragraphs(
    pairs: list[TextPair],
    *,
    source_language: str | None,
    target_language: str | None,
) -> list[TextError]:
    if not _supports_latin_source_guard(source_language, target_language):
        return []

    errors: list[TextError] = []
    for pair in pairs:
        if not _is_eligible_unchanged_paragraph(pair):
            continue
        bbox: dict[str, list[int]] = {}
        if pair.source_bbox:
            bbox["source_text_area"] = list(pair.source_bbox)
        if pair.translated_bbox:
            bbox["translated_text_area"] = list(pair.translated_bbox)
        errors.append(
            TextError(
                error_id=f"{pair.pair_id}:faithfulness:under_translation",
                dimension=TextDimension.FAITHFULNESS,
                type="under_translation",
                severity=Severity.MAJOR,
                pair_id=pair.pair_id,
                evidence=(
                    "该完整英文正文段落在译图中仍基本原样保留，仅有换行、空格或标点差异，"
                    "未转换为目标语言。"
                ),
                source_fragment=pair.source_text,
                target_fragment=pair.translated_text,
                suggestion=format_suggestion(pair.source_text, "[译为目标语言]"),
                bbox=bbox or None,
                confidence=min(max(pair.pair_confidence, 0.95), 0.99),
                source="unchanged_language_guard",
                needs_skill=False,
            )
        )
    return errors


def _supports_latin_source_guard(
    source_language: str | None,
    target_language: str | None,
) -> bool:
    source = _language_family(source_language)
    target = _language_family(target_language)
    return source == "en" and bool(target) and target != source


def _is_eligible_unchanged_paragraph(pair: TextPair) -> bool:
    unit_type = str(pair.metadata.get("unit_type", "")).casefold()
    if unit_type not in PRIMARY_BLOCK_TYPES:
        return False
    source = pair.source_text.strip()
    target = pair.translated_text.strip()
    normalized_source = _normalized_comparison_text(source)
    normalized_target = _normalized_comparison_text(target)
    if (
        len(normalized_source) < MIN_UNCHANGED_PARAGRAPH_CHARACTERS
        or not normalized_target
        or _looks_like_preservable_token(source)
        or not _is_predominantly_latin(source)
    ):
        return False
    return (
        normalized_source == normalized_target
        or SequenceMatcher(None, normalized_source, normalized_target).ratio()
        >= MIN_UNCHANGED_SIMILARITY
    )


def _normalized_comparison_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _is_predominantly_latin(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    latin_letters = [
        character
        for character in letters
        if "LATIN" in unicodedata.name(character, "")
    ]
    return (
        len(latin_letters) >= MIN_LATIN_LETTERS
        and _ratio(len(latin_letters), len(letters)) >= MIN_LATIN_LETTER_RATIO
    )


def _looks_like_preservable_token(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"(?:https?://|www\.)\S+", stripped, flags=re.IGNORECASE):
        return True
    if "\n" not in stripped and len(stripped.split()) <= 5:
        return bool(re.fullmatch(r"[A-Za-z0-9_.+/#:-]+", stripped))
    return False


def _language_family(language: str | None) -> str:
    return str(language or "").strip().replace("_", "-").split("-", 1)[0].casefold()


def _trigger_reasons(
    unmatched_block_count: int,
    unmatched_block_ratio: float,
    unmatched_character_ratio: float,
    target_to_source_block_ratio: float,
    missing_primary: list[ImageTextBlock],
) -> list[str]:
    reasons: list[str] = []
    if unmatched_block_count >= MIN_CLEAR_UNMATCHED_BLOCKS:
        reasons.append("many_clear_unmatched_source_blocks")
    if unmatched_block_ratio >= MIN_UNMATCHED_BLOCK_RATIO:
        reasons.append("high_unmatched_source_block_ratio")
    if unmatched_character_ratio >= MIN_UNMATCHED_CHARACTER_RATIO:
        reasons.append("high_unmatched_source_character_ratio")
    if target_to_source_block_ratio <= MAX_TARGET_TO_SOURCE_BLOCK_RATIO:
        reasons.append("very_low_target_text_coverage")
    if missing_primary:
        reasons.append("primary_content_missing")
    return reasons


def _is_primary_content(block: ImageTextBlock) -> bool:
    character_count = _character_count(block.text)
    return (
        character_count >= MIN_PRIMARY_TEXT_CHARACTERS
        and (
            block.block_type.casefold() in PRIMARY_BLOCK_TYPES
            or _bbox_area(block.bbox_1000) >= 20_000
        )
    )


def _is_reliably_readable(block: ImageTextBlock) -> bool:
    return block.visibility == "clear" or (
        block.visibility == "partial"
        and block.confidence >= MIN_RELIABLE_PARTIAL_CONFIDENCE
    )


def _character_count(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def _bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
