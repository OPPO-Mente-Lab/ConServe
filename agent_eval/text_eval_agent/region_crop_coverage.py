# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


MIN_RELIABLE_CONFIDENCE = 0.85
MIN_CRITICAL_SOURCE_CHARACTERS = 80
MIN_CRITICAL_UNMATCHED_RATIO = 0.70
MAJOR_BLOCK_TYPES = {"body", "heading", "button", "field"}
IGNORED_BLOCK_TYPES = {"brand", "decorative"}
VISIBILITY_WARNINGS = {"cropped_text", "blurred_characters", "occluded_text"}


def build_region_coverage_pairs(
    task_id: str,
    extraction: dict[str, Any],
    task: dict[str, Any],
    paired_source_texts: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for segment in extraction.get("unmatched_source_segments", []):
        reason = _ineligibility_reason(segment)
        if reason:
            issues.append(
                {
                    "segment_id": segment["segment_id"],
                    "type": reason,
                    "source_visibility": segment["source_visibility"],
                    "source_confidence": segment["source_confidence"],
                    "warnings": list(segment["warnings"]),
                    "scored": False,
                }
            )
            continue
        eligible.append(segment)

    if not eligible:
        return [], issues

    source_text = "\n".join(segment["source_text"].strip() for segment in eligible)
    unmatched_characters = _character_count(source_text)
    paired_characters = sum(_character_count(text) for text in paired_source_texts)
    total_characters = paired_characters + unmatched_characters
    unmatched_ratio = unmatched_characters / total_characters if total_characters else 0.0
    severity = _severity(eligible, unmatched_characters, unmatched_ratio)
    bbox = _union_bbox([segment["source_bbox_1000"] for segment in eligible])
    pair_id = f"{task_id}:coverage_unmatched_01"
    pair = {
        "pair_id": pair_id,
        "source_text": source_text,
        "translated_text": "",
        "source_language": task.get("source_language") or eligible[0]["source_language"],
        "target_language": task.get("target_language"),
        "source_bbox": bbox,
        "translated_bbox": None,
        "pair_confidence": min(float(segment["source_confidence"]) for segment in eligible),
        "source_ocr_confidence": min(
            float(segment["source_confidence"]) for segment in eligible
        ),
        "metadata": {
            "unit_type": "coverage_omission",
            "coverage_severity": severity,
            "coverage_source": "region_crop_coverage_guard",
            "unmatched_segment_ids": [segment["segment_id"] for segment in eligible],
            "unmatched_character_ratio": round(unmatched_ratio, 4),
            "image_context": dict(extraction["image_context"]),
        },
    }
    return [pair], issues


def _ineligibility_reason(segment: dict[str, Any]) -> str | None:
    if not segment["translatable"] or segment["block_type"] in IGNORED_BLOCK_TYPES:
        return "unmatched_source_nontranslatable"
    if (
        segment["source_visibility"] != "clear"
        or float(segment["source_confidence"]) < MIN_RELIABLE_CONFIDENCE
        or VISIBILITY_WARNINGS.intersection(segment["warnings"])
        or not segment["source_text"].strip()
    ):
        return "unmatched_source_insufficient_evidence"
    return None


def _severity(
    segments: list[dict[str, Any]],
    unmatched_characters: int,
    unmatched_ratio: float,
) -> str:
    has_primary = any(segment["importance"] == "primary" for segment in segments)
    if (
        has_primary
        and unmatched_characters >= MIN_CRITICAL_SOURCE_CHARACTERS
        and unmatched_ratio >= MIN_CRITICAL_UNMATCHED_RATIO
    ):
        return "Critical"
    if any(
        segment["importance"] in {"primary", "supporting"}
        or segment["block_type"] in MAJOR_BLOCK_TYPES
        for segment in segments
    ):
        return "Major"
    return "Minor"


def _character_count(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def _union_bbox(boxes: list[list[int]]) -> list[int]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
