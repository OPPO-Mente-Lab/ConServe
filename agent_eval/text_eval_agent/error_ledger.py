# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import TextError, TextPair
from .policy import annotate_fragment_dict, locate_fragment


SEVERITY_RANK = {"Minor": 1, "Major": 2, "Critical": 3}
MEANING_CHANGE_TYPES = {
    "mistranslation",
    "omission",
    "under_translation",
    "over_translation",
    "addition",
    "hallucination",
    "terminology_error",
}
EMPTY_FRAGMENT_MARKERS = {"", "-", "—", "...", "…", "null", "none"}


def normalize_error_ledger(
    adjudicated_errors: list[dict[str, Any]],
    pairs: list[TextPair],
    candidate_errors: Iterable[TextError | dict[str, Any]] | None = None,
    suppressed_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    candidates = [_as_dict(item) for item in (candidate_errors or [])]
    candidate_by_id = {
        str(item.get("error_id")): item for item in candidates if item.get("error_id")
    }
    suppressed = [dict(item) for item in (suppressed_errors or [])]
    notes: list[str] = []
    validated: list[dict[str, Any]] = []

    for raw_error in adjudicated_errors:
        error = dict(raw_error)
        pair_id = str(error.get("pair_id", ""))
        pair = pair_by_id.get(pair_id)
        if pair is None:
            _suppress_sources(error, suppressed, "insufficient_evidence", None)
            notes.append(
                f"Suppressed {error.get('error_id', 'unknown_error')} because its pair_id is unknown."
            )
            continue

        error["merged_from"] = _merged_ids(error)
        _recover_fragments(error, pair, candidate_by_id)
        error = annotate_fragment_dict(error, pair.source_text, pair.translated_text)
        dimension = str(error.get("primary_dimension") or error.get("dimension") or "")
        required_span = (
            error.get("source_fragment_span")
            if dimension == "text.faithfulness"
            else error.get("target_fragment_span")
        )
        if required_span is None:
            _suppress_sources(error, suppressed, "insufficient_evidence", None)
            notes.append(
                f"Suppressed {error.get('error_id', 'unknown_error')} because required fragment evidence is missing."
            )
            continue
        validated.append(error)

    normalized: list[dict[str, Any]] = []
    for error in validated:
        match = next(
            (
                kept
                for kept in normalized
                if _same_occurrence(kept, error) and _compatible_root_cause(kept, error)
            ),
            None,
        )
        if match is None:
            normalized.append(error)
            continue
        survivor, duplicate = _choose_survivor(match, error)
        _merge_error(survivor, duplicate)
        if survivor is not match:
            normalized[normalized.index(match)] = survivor
        _suppress_sources(
            duplicate,
            suppressed,
            "duplicate" if survivor.get("primary_dimension") == duplicate.get("primary_dimension") else "secondary_impact",
            str(survivor.get("error_id")),
        )
        notes.append(
            f"Merged duplicate error {duplicate.get('error_id', 'unknown_error')} into "
            f"{survivor.get('error_id', 'unknown_error')} within pair {survivor.get('pair_id')}."
        )

    return {
        "adjudicated_errors": normalized,
        "suppressed_errors": _unique_suppressions(suppressed),
        "notes": notes,
    }


def _as_dict(value: TextError | dict[str, Any]) -> dict[str, Any]:
    return value.to_dict() if isinstance(value, TextError) else dict(value)


def _meaningful_fragment(value: object) -> str | None:
    text = str(value or "").strip()
    return None if text.casefold() in EMPTY_FRAGMENT_MARKERS else text


def _merged_ids(error: dict[str, Any]) -> list[str]:
    values = error.get("merged_from") or [error.get("error_id")]
    return list(dict.fromkeys(str(value) for value in values if value))


def _recover_fragments(
    error: dict[str, Any],
    pair: TextPair,
    candidate_by_id: dict[str, dict[str, Any]],
) -> None:
    for field, text in (
        ("source_fragment", pair.source_text),
        ("target_fragment", pair.translated_text),
    ):
        fragment = _meaningful_fragment(error.get(field))
        exact_fragment = _locate_whitespace_tolerant_excerpt(text, fragment)
        if exact_fragment:
            error[field] = exact_fragment
            continue
        error[field] = "" if field == "target_fragment" and error.get("type") == "omission" else None
        for candidate_id in _merged_ids(error):
            candidate = candidate_by_id.get(candidate_id)
            candidate_fragment = _meaningful_fragment(
                candidate.get(field) if candidate else None
            )
            exact_candidate_fragment = _locate_whitespace_tolerant_excerpt(
                text, candidate_fragment
            )
            if exact_candidate_fragment:
                error[field] = exact_candidate_fragment
                break


def _locate_whitespace_tolerant_excerpt(
    text: str, fragment: str | None
) -> str | None:
    if not fragment:
        return None
    if locate_fragment(text, fragment):
        return fragment

    normalized_text_chars: list[str] = []
    original_indexes: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        normalized_text_chars.append(char)
        original_indexes.append(index)
    normalized_fragment = "".join(char for char in fragment if not char.isspace())
    if not normalized_fragment:
        return None
    start = "".join(normalized_text_chars).find(normalized_fragment)
    if start < 0:
        return None
    end = start + len(normalized_fragment) - 1
    return text[original_indexes[start] : original_indexes[end] + 1]


def _spans_overlap(left: dict[str, int] | None, right: dict[str, int] | None) -> bool:
    if left is None or right is None:
        return False
    return max(left["start"], right["start"]) < min(left["end"], right["end"])


def _same_occurrence(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("pair_id") != right.get("pair_id"):
        return False
    source_overlap = _spans_overlap(
        left.get("source_fragment_span"), right.get("source_fragment_span")
    )
    target_overlap = _spans_overlap(
        left.get("target_fragment_span"), right.get("target_fragment_span")
    )
    if left.get("primary_dimension") != right.get("primary_dimension"):
        return target_overlap
    return source_overlap or target_overlap


def _compatible_root_cause(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_dimension = str(left.get("primary_dimension"))
    right_dimension = str(right.get("primary_dimension"))
    if left_dimension != right_dimension:
        return {
            left_dimension,
            right_dimension,
        } == {"text.faithfulness", "text.fluency"}
    if left_dimension == "text.fluency":
        return True
    left_type = str(left.get("type"))
    right_type = str(right.get("type"))
    return left_type == right_type or {
        left_type,
        right_type,
    }.issubset(MEANING_CHANGE_TYPES)


def _choose_survivor(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    def rank(error: dict[str, Any]) -> tuple[int, int]:
        return (
            1 if error.get("primary_dimension") == "text.faithfulness" else 0,
            SEVERITY_RANK.get(str(error.get("severity")), 0),
        )

    return (left, right) if rank(left) >= rank(right) else (right, left)


def _merge_error(survivor: dict[str, Any], duplicate: dict[str, Any]) -> None:
    survivor["merged_from"] = list(
        dict.fromkeys(_merged_ids(survivor) + _merged_ids(duplicate))
    )
    if SEVERITY_RANK.get(str(duplicate.get("severity")), 0) > SEVERITY_RANK.get(
        str(survivor.get("severity")), 0
    ):
        survivor["severity"] = duplicate.get("severity")
    survivor["confidence"] = max(
        float(survivor.get("confidence", 0.0)),
        float(duplicate.get("confidence", 0.0)),
    )
    if (
        survivor.get("primary_dimension") == "text.faithfulness"
        and {str(survivor.get("type")), str(duplicate.get("type"))}
        <= MEANING_CHANGE_TYPES
    ):
        has_target = bool(
            _meaningful_fragment(survivor.get("target_fragment"))
            or _meaningful_fragment(duplicate.get("target_fragment"))
        )
        survivor["type"] = "mistranslation" if has_target else "omission"


def _suppress_sources(
    error: dict[str, Any],
    suppressed: list[dict[str, Any]],
    reason: str,
    merged_into: str | None,
) -> None:
    for candidate_id in _merged_ids(error):
        suppressed.append(
            {
                "candidate_error_id": candidate_id,
                "reason": reason,
                "merged_into": merged_into,
            }
        )


def _unique_suppressions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for item in items:
        key = (
            str(item.get("candidate_error_id", "")),
            str(item.get("reason", "")),
            item.get("merged_into"),
        )
        unique[key] = item
    return list(unique.values())
