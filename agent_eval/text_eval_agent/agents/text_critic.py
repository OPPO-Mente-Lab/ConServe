# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from ..llm_client import ModelClient
from ..models import TextError, TextPair, image_context_for_prompt
from ..error_ledger import normalize_error_ledger
from ..prompt_loader import load_prompt
from ..policy import POLICY_VERSION, annotate_fragment_dict, is_non_scored_error_type
from ..runtime import ModelRuntimeConfig


FAITHFULNESS_TYPES = {
    "mistranslation",
    "omission",
    "under_translation",
    "over_translation",
    "addition",
    "hallucination",
    "repetition",
    "number_error",
    "unit_error",
    "date_error",
    "negation_error",
    "relation_error",
    "terminology_error",
    "text_pair_uncertain",
}

FLUENCY_TYPES = {
    "grammar_error",
    "awkward_expression",
    "word_order_error",
    "collocation_error",
    "translationese",
    "unclear_expression",
    "style_mismatch",
    "audience_mismatch",
    "garbled_expression",
}

TYPE_ALIASES = {
    "text.faithfulness": {
        "inaccurate_translation": "mistranslation",
        "wrong_translation": "mistranslation",
    },
    "text.fluency": {
        "unnatural_wording": "awkward_expression",
        "awkward_word_usage": "awkward_expression",
        "grammar_and_syntax": "grammar_error",
    },
}


class TextCritic:
    """Model-backed adjudicator for candidate Faithfulness/Fluency errors."""

    def __init__(self, model_client: ModelClient, runtime: ModelRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def review(self, pair: TextPair, errors: list[TextError]) -> dict[str, Any]:
        response = self.model_client.complete_json(
            "text_critic",
            [
                {"role": "system", "content": load_prompt("text_critic")},
                {"role": "user", "content": _critic_prompt(pair, errors)},
            ],
            self.runtime,
        )
        return normalize_critic_response(response, pair, errors)

    def review_many(self, pairs: list[TextPair], errors: list[TextError]) -> dict[str, Any]:
        if not errors:
            return {
                "policy_version": POLICY_VERSION,
                "adjudicated_errors": [],
                "suppressed_errors": [],
                "severity_changes": [],
                "caps": [],
                "notes": [],
            }
        response = self.model_client.complete_json(
            "text_critic",
            [
                {"role": "system", "content": load_prompt("text_critic")},
                {"role": "user", "content": _critic_many_prompt(pairs, errors)},
            ],
            self.runtime,
        )
        return normalize_critic_response_many(response, pairs, errors)


def apply_deterministic_coverage_gate(
    review: dict[str, Any],
    coverage_errors: list[TextError],
) -> dict[str, Any]:
    if not coverage_errors:
        return review
    updated = dict(review)
    adjudicated = [dict(error) for error in review.get("adjudicated_errors", [])]
    existing_sources = {
        str(source_id)
        for error in adjudicated
        for source_id in error.get("merged_from", [])
    }
    existing_pair_types = {
        (str(error.get("pair_id", "")), str(error.get("type", "")))
        for error in adjudicated
    }
    for coverage_error in coverage_errors:
        pair_type = (coverage_error.pair_id, coverage_error.type)
        if (
            coverage_error.error_id in existing_sources
            or pair_type in existing_pair_types
        ):
            continue
        adjudicated.append(
            {
                "error_id": coverage_error.error_id,
                "primary_dimension": coverage_error.dimension.value,
                "type": coverage_error.type,
                "severity": coverage_error.severity.value,
                "pair_id": coverage_error.pair_id,
                "source_fragment": coverage_error.source_fragment,
                "target_fragment": coverage_error.target_fragment,
                "evidence": coverage_error.evidence,
                "suggestion": coverage_error.suggestion,
                "merged_from": [coverage_error.error_id],
                "confidence": coverage_error.confidence,
            }
        )
        existing_pair_types.add(pair_type)
    updated["adjudicated_errors"] = adjudicated
    gate_note = "已应用确定性全图文字覆盖门控；高置信度主要内容缺失保留为 Critical。"
    notes = list(review.get("notes", []))
    if gate_note not in notes:
        notes.append(gate_note)
    updated["notes"] = notes
    return updated


def normalize_critic_response(
    response: dict[str, Any],
    pair: TextPair | None = None,
    candidate_errors: list[TextError] | None = None,
) -> dict[str, Any]:
    notes = list(response.get("notes", []))
    severity_changes = list(response.get("severity_changes", []))
    suppressed_errors = list(response.get("suppressed_errors", []))
    adjudicated_errors = []
    for raw_error in response.get("adjudicated_errors", response.get("errors", [])):
        error = dict(raw_error)
        if is_non_scored_error_type(error.get("type")):
            source_ids = error.get("merged_from") or [error.get("error_id", "unknown_error")]
            existing_ids = {item.get("candidate_error_id") for item in suppressed_errors}
            for source_id in source_ids:
                if source_id not in existing_ids:
                    suppressed_errors.append(
                        {
                            "candidate_error_id": str(source_id),
                            "reason": "non_scored_format_or_metadata",
                            "merged_into": None,
                        }
                    )
            notes.append(f"Filtered non-scored error type {error.get('type')!r}.")
            continue
        dimension = str(error.get("primary_dimension", ""))
        normalized_type = _normalize_error_type(dimension, str(error.get("type", "")))
        if normalized_type != error.get("type"):
            notes.append(
                f"Normalized critic type {error.get('type')!r} to {normalized_type!r} "
                f"for {dimension or 'unknown dimension'}."
            )
            error["type"] = normalized_type

        if error.get("severity") == "Critical" and not _supports_critical(error, pair):
            error["severity"] = "Major"
            severity_changes.append(
                {
                    "candidate_error_id": str(error.get("error_id", "unknown_error")),
                    "from": "Critical",
                    "to": "Major",
                    "reason": "Critical requires explicit high-risk context or an unusable garbled target.",
                }
            )
        if pair:
            error = annotate_fragment_dict(error, pair.source_text, pair.translated_text)
        adjudicated_errors.append(error)

    if pair is not None and candidate_errors is not None:
        ledger = normalize_error_ledger(
            adjudicated_errors,
            [pair],
            candidate_errors,
            suppressed_errors,
        )
        adjudicated_errors = ledger["adjudicated_errors"]
        suppressed_errors = ledger["suppressed_errors"]
        notes.extend(ledger["notes"])

    caps = list(response.get("caps", []))
    if caps and not any(error.get("severity") == "Critical" for error in adjudicated_errors):
        caps = []
        notes.append("Removed score caps because no adjudicated Critical error remained.")

    return {
        "policy_version": POLICY_VERSION,
        "adjudicated_errors": adjudicated_errors,
        "suppressed_errors": suppressed_errors,
        "severity_changes": severity_changes,
        "caps": caps,
        "notes": notes,
    }


def normalize_critic_response_many(
    response: dict[str, Any],
    pairs: list[TextPair],
    candidate_errors: list[TextError] | None = None,
) -> dict[str, Any]:
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    notes = list(response.get("notes", []))
    severity_changes = list(response.get("severity_changes", []))
    suppressed_errors = list(response.get("suppressed_errors", []))
    adjudicated_errors = []
    for raw_error in response.get("adjudicated_errors", response.get("errors", [])):
        error = dict(raw_error)
        pair_id = str(error.get("pair_id", ""))
        pair = pair_by_id.get(pair_id)
        if pair is None:
            notes.append(f"Dropped adjudicated error {error.get('error_id', 'unknown_error')} with unknown pair_id {pair_id!r}.")
            continue
        if is_non_scored_error_type(error.get("type")):
            notes.append(f"Filtered non-scored error type {error.get('type')!r}.")
            continue
        dimension = str(error.get("primary_dimension", ""))
        normalized_type = _normalize_error_type(dimension, str(error.get("type", "")))
        if normalized_type != error.get("type"):
            notes.append(
                f"Normalized critic type {error.get('type')!r} to {normalized_type!r} for {dimension or 'unknown dimension'}."
            )
            error["type"] = normalized_type
        if error.get("severity") == "Critical" and not _supports_critical(error, pair):
            error["severity"] = "Major"
            severity_changes.append(
                {
                    "candidate_error_id": str(error.get("error_id", "unknown_error")),
                    "from": "Critical",
                    "to": "Major",
                    "reason": "Critical requires explicit high-risk context or an unusable garbled target.",
                }
            )
        adjudicated_errors.append(annotate_fragment_dict(error, pair.source_text, pair.translated_text))
    if candidate_errors is not None:
        ledger = normalize_error_ledger(
            adjudicated_errors,
            pairs,
            candidate_errors,
            suppressed_errors,
        )
        adjudicated_errors = ledger["adjudicated_errors"]
        suppressed_errors = ledger["suppressed_errors"]
        notes.extend(ledger["notes"])

    caps = list(response.get("caps", []))
    if caps and not any(error.get("severity") == "Critical" for error in adjudicated_errors):
        caps = []
        notes.append("Removed score caps because no adjudicated Critical error remained.")
    return {
        "policy_version": POLICY_VERSION,
        "adjudicated_errors": adjudicated_errors,
        "suppressed_errors": suppressed_errors,
        "severity_changes": severity_changes,
        "caps": caps,
        "notes": notes,
    }


def _normalize_error_type(dimension: str, error_type: str) -> str:
    normalized = TYPE_ALIASES.get(dimension, {}).get(error_type, error_type)
    if dimension == "text.fluency":
        return normalized if normalized in FLUENCY_TYPES else "awkward_expression"
    return normalized if normalized in FAITHFULNESS_TYPES else "mistranslation"


def _supports_critical(error: dict[str, Any], pair: TextPair | None) -> bool:
    if error.get("primary_dimension") == "text.faithfulness" and error.get("type") == "repetition":
        return bool(str(error.get("evidence", "")).strip())
    if error.get("primary_dimension") == "text.fluency" and error.get("type") == "garbled_expression":
        return bool(str(error.get("evidence", "")).strip())
    return bool(pair and pair.risk_level == "high" and str(error.get("evidence", "")).strip())


def _critic_prompt(pair: TextPair, errors: list[TextError]) -> str:
    payload = {
        "output_language": "zh-CN",
        "pair_id": pair.pair_id,
        "source_text": pair.source_text,
        "translated_text": pair.translated_text,
        "source_language": pair.source_language,
        "target_language": pair.target_language,
        "risk_level": pair.risk_level,
        "risk_type": pair.risk_type,
        "candidate_errors": [error.to_dict() for error in errors],
    }
    context = image_context_for_prompt(pair)
    if context:
        payload["image_context"] = context
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _critic_many_prompt(pairs: list[TextPair], errors: list[TextError]) -> str:
    payload = {
        "mode": "batch",
        "output_language": "zh-CN",
        "instruction": "Adjudicate candidates across all pairs. Keep pair_id on every final error and never merge errors from different pair_id values.",
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "source_text": pair.source_text,
                "translated_text": pair.translated_text,
                "source_language": pair.source_language,
                "target_language": pair.target_language,
                "risk_level": pair.risk_level,
                "risk_type": pair.risk_type,
                **({"image_context": context} if (context := image_context_for_prompt(pair)) else {}),
            }
            for pair in pairs
        ],
        "candidate_errors": [error.to_dict() for error in errors],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
