# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


_CONTEXT_FIELDS = {
    "scene_type",
    "document_type",
    "domain",
    "user_intent",
    "context_summary",
    "terminology_clues",
    "surrounding_text_clues",
    "confidence",
}
_PAIR_FIELDS = {
    "pid",
    "source_text",
    "translated_text",
    "source_bbox_1000",
    "source_language",
    "source_visibility",
    "source_confidence",
    "pair_confidence",
    "warnings",
}
_UNMATCHED_SOURCE_FIELDS = {
    "segment_id",
    "source_text",
    "source_bbox_1000",
    "source_language",
    "source_visibility",
    "source_confidence",
    "block_type",
    "translatable",
    "importance",
    "warnings",
}
_OUTPUT_FIELDS = {
    "image_context",
    "pairs",
    "missing_pids",
    "unexpected_pids",
    "unmatched_source_segments",
    "notes",
}
_VISIBILITIES = {"clear", "partial", "uncertain"}
_IMPORTANCE_LEVELS = {"primary", "supporting", "low"}
_PARTIAL_WARNINGS = {"cropped_text", "blurred_characters", "occluded_text"}


def validate_translations(translations: list[dict]) -> dict[int, dict[str, Any]]:
    if not isinstance(translations, list):
        raise ValueError("translations must be a list")
    candidates: dict[int, dict[str, Any]] = {}
    for entry in translations:
        if not isinstance(entry, dict):
            raise ValueError("translation entries must be objects")
        pid = _validate_pid(entry.get("pid"), "translation pid")
        translated_text = entry.get("translated_text")
        if not isinstance(translated_text, str):
            raise ValueError("translated_text must be a string")
        if pid in candidates:
            raise ValueError("translation pids must be unique")
        candidates[pid] = {"pid": pid, "translated_text": translated_text}
    return candidates


def validate_region_response(
    response: dict[str, Any], candidates: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != _OUTPUT_FIELDS:
        raise ValueError("region crop extraction response has an invalid schema")
    context = _validate_context(response["image_context"])
    pairs = _validate_pairs(response["pairs"], candidates)
    missing_pids = _validate_pid_list(response["missing_pids"], "missing_pids")
    unexpected_pids = _validate_pid_list(
        response["unexpected_pids"], "unexpected_pids"
    )
    unmatched = _validate_unmatched_source_segments(
        response["unmatched_source_segments"]
    )
    notes = _validate_string_list(response["notes"], "notes")

    pair_pids = {pair["pid"] for pair in pairs}
    missing_set = set(missing_pids)
    if pair_pids & missing_set or pair_pids | missing_set != set(candidates):
        raise ValueError("missing_pids must exactly reconcile requested pids")
    if any(pid in candidates for pid in unexpected_pids):
        raise ValueError("unexpected_pids must not contain requested pids")

    return {
        "image_context": context,
        "pairs": pairs,
        "missing_pids": missing_pids,
        "unexpected_pids": unexpected_pids,
        "unmatched_source_segments": unmatched,
        "notes": notes,
    }


def _validate_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONTEXT_FIELDS:
        raise ValueError("image_context has an invalid schema")
    context = dict(value)
    for field in (
        "scene_type",
        "document_type",
        "domain",
        "user_intent",
        "context_summary",
    ):
        if not isinstance(context[field], str):
            raise ValueError(f"image_context.{field} must be a string")
    context["terminology_clues"] = _validate_string_list(
        context["terminology_clues"], "image_context.terminology_clues"
    )
    context["surrounding_text_clues"] = _validate_string_list(
        context["surrounding_text_clues"], "image_context.surrounding_text_clues"
    )
    context["confidence"] = _validate_confidence(
        context["confidence"], "image_context.confidence"
    )
    return context


def _validate_pairs(
    value: Any, candidates: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("pairs must be a list")
    pairs: list[dict[str, Any]] = []
    seen_pids: set[int] = set()
    for raw_pair in value:
        if not isinstance(raw_pair, dict) or set(raw_pair) != _PAIR_FIELDS:
            raise ValueError("pair has an invalid schema")
        pair = dict(raw_pair)
        pid = _validate_pid(pair["pid"], "pair pid")
        if pid not in candidates:
            raise ValueError("pair has an unknown pid")
        if pid in seen_pids:
            raise ValueError("returned pair pids must be unique")
        seen_pids.add(pid)
        for field in ("source_text", "translated_text", "source_language"):
            if not isinstance(pair[field], str):
                raise ValueError(f"pair.{field} must be a string")
        pair["source_bbox_1000"] = _validate_bbox(pair["source_bbox_1000"])
        if pair["source_visibility"] not in _VISIBILITIES:
            raise ValueError("pair.source_visibility is invalid")
        pair["source_confidence"] = _validate_confidence(
            pair["source_confidence"], "pair.source_confidence"
        )
        pair["pair_confidence"] = _validate_confidence(
            pair["pair_confidence"], "pair.pair_confidence"
        )
        pair["warnings"] = _validate_string_list(pair["warnings"], "pair.warnings")
        if pair["source_visibility"] == "partial" and (
            pair["source_confidence"] >= 1.0
            or not _PARTIAL_WARNINGS.intersection(pair["warnings"])
        ):
            raise ValueError(
                "partial source must have reduced confidence and a visibility warning"
            )
        if not pair["source_text"] and (
            pair["source_visibility"] != "uncertain"
            or pair["source_confidence"] > 0.25
            or "source_unreadable" not in pair["warnings"]
        ):
            raise ValueError(
                "unreadable source must be uncertain with source_unreadable warning"
            )
        pair["pid"] = pid
        pair["translated_text"] = candidates[pid]["translated_text"]
        pairs.append(pair)
    return pairs


def _validate_unmatched_source_segments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("unmatched_source_segments must be a list")
    segments: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_segment in value:
        if (
            not isinstance(raw_segment, dict)
            or set(raw_segment) != _UNMATCHED_SOURCE_FIELDS
        ):
            raise ValueError("unmatched source segment has an invalid schema")
        segment = dict(raw_segment)
        segment_id = segment["segment_id"]
        if (
            not isinstance(segment_id, str)
            or not segment_id.strip()
            or segment_id in seen_ids
        ):
            raise ValueError("unmatched source segment_id must be unique and non-empty")
        seen_ids.add(segment_id)
        for field in ("source_text", "source_language", "block_type"):
            if not isinstance(segment[field], str):
                raise ValueError(f"unmatched source {field} must be a string")
        segment["source_bbox_1000"] = _validate_bbox(
            segment["source_bbox_1000"]
        )
        if segment["source_visibility"] not in _VISIBILITIES:
            raise ValueError("unmatched source visibility is invalid")
        segment["source_confidence"] = _validate_confidence(
            segment["source_confidence"], "unmatched source confidence"
        )
        if not isinstance(segment["translatable"], bool):
            raise ValueError("unmatched source translatable must be a boolean")
        if segment["importance"] not in _IMPORTANCE_LEVELS:
            raise ValueError("unmatched source importance is invalid")
        segment["warnings"] = _validate_string_list(
            segment["warnings"], "unmatched source warnings"
        )
        if not segment["source_text"].strip() and segment["source_visibility"] == "clear":
            raise ValueError("clear unmatched source text must not be empty")
        segments.append(segment)
    return segments


def _validate_pid_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    pids = [_validate_pid(pid, name) for pid in value]
    if len(set(pids)) != len(pids):
        raise ValueError(f"{name} must contain unique pids")
    return pids


def _validate_pid(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_bbox(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(item, int) or isinstance(item, bool) for item in value
        )
    ):
        raise ValueError("source_bbox_1000 must contain four integers")
    ymin, xmin, ymax, xmax = value
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        raise ValueError("source_bbox_1000 must be a normalized non-empty bbox")
    return list(value)


def _validate_confidence(value: Any, name: str) -> float | int:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a number from 0 to 1")
    return value


def _validate_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{name} must be a list of strings")
    return list(value)
