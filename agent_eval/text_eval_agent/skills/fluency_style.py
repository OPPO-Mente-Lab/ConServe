# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

from ..llm_client import ModelClient
from ..models import TextDimension, TextError, TextPair, image_context_for_prompt, refresh_fragment_spans
from ..policy import is_non_scored_error_type
from ..prompt_loader import load_skill_instructions
from ..runtime import ModelRuntimeConfig
from ..suggestion_format import normalize_suggestion


class FluencyStyleSkill:
    """Model-backed detail skill for fluency and style errors."""

    def __init__(self, model_client: ModelClient, runtime: ModelRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def refine(self, pair: TextPair, errors: list[TextError]) -> list[TextError]:
        response = self.model_client.complete_json(
            "fluency_style",
            [
                {"role": "system", "content": load_skill_instructions("fluency-style")},
                {"role": "user", "content": _detail_prompt(pair, errors)},
            ],
            self.runtime,
        )
        updates = {item.get("original_error_id"): item for item in response.get("updated_errors", [])}
        refined: list[TextError] = []
        for error in errors:
            update = updates.get(error.error_id)
            if not update:
                refined.append(error)
                continue
            source_fragment = update.get("source_fragment", error.source_fragment)
            target_fragment = update.get("target_fragment", error.target_fragment)
            suggestion = normalize_suggestion(
                update.get("suggestion", error.suggestion),
                source_fragment=source_fragment,
                target_fragment=target_fragment,
            )
            refined.append(
                refresh_fragment_spans(replace(
                    error,
                    type=update.get("type", error.type),
                    severity=update.get("severity", error.severity),
                    penalty=None,
                    evidence=update.get("evidence", error.evidence),
                    source_fragment=source_fragment,
                    target_fragment=target_fragment,
                    suggestion=suggestion,
                    confidence=float(update.get("confidence", error.confidence)),
                    source="fluency_style_skill",
                    needs_skill=False,
                ), pair)
            )

        for item in response.get("new_errors", []):
            if not is_non_scored_error_type(item.get("type")):
                refined.append(_normalized_model_error(item, pair, TextDimension.FLUENCY, "fluency_style_skill"))
        return [error for error in refined if not is_non_scored_error_type(error.type)]

    def refine_many(self, pairs: list[TextPair], errors: list[TextError]) -> list[TextError]:
        if not errors:
            return []
        pair_by_id = {pair.pair_id: pair for pair in pairs}
        response = self.model_client.complete_json(
            "fluency_style",
            [
                {"role": "system", "content": load_skill_instructions("fluency-style")},
                {"role": "user", "content": _batch_detail_prompt(pairs, errors)},
            ],
            self.runtime,
        )
        updates = {item.get("original_error_id"): item for item in response.get("updated_errors", [])}
        refined: list[TextError] = []
        for error in errors:
            pair = pair_by_id.get(error.pair_id)
            if pair is None:
                continue
            update = updates.get(error.error_id)
            if not update or (update.get("pair_id") and update.get("pair_id") != error.pair_id):
                refined.append(error)
                continue
            source_fragment = update.get("source_fragment", error.source_fragment)
            target_fragment = update.get("target_fragment", error.target_fragment)
            suggestion = normalize_suggestion(
                update.get("suggestion", error.suggestion),
                source_fragment=source_fragment,
                target_fragment=target_fragment,
            )
            refined.append(refresh_fragment_spans(replace(
                error,
                type=update.get("type", error.type),
                severity=update.get("severity", error.severity),
                penalty=None,
                evidence=update.get("evidence", error.evidence),
                source_fragment=source_fragment,
                target_fragment=target_fragment,
                suggestion=suggestion,
                confidence=float(update.get("confidence", error.confidence)),
                source="fluency_style_skill",
                needs_skill=False,
            ), pair))
        for item in response.get("new_errors", []):
            pair_id = str(item.get("pair_id") or (pairs[0].pair_id if len(pairs) == 1 else ""))
            pair = pair_by_id.get(pair_id)
            if pair and not is_non_scored_error_type(item.get("type")):
                refined.append(_normalized_model_error(item, pair, TextDimension.FLUENCY, "fluency_style_skill"))
        return [error for error in refined if not is_non_scored_error_type(error.type)]


def _normalized_model_error(item: dict, pair: TextPair, dimension: TextDimension, source: str) -> TextError:
    error = TextError.from_model_dict(item, pair, dimension, source)
    return replace(
        error,
        suggestion=normalize_suggestion(
            error.suggestion,
            source_fragment=error.source_fragment,
            target_fragment=error.target_fragment,
        ),
    )


def _detail_prompt(pair: TextPair, errors: list[TextError]) -> str:
    prompt = (
        "output_language: zh-CN\n"
        f"pair_id: {pair.pair_id}\n"
        f"source_text: {pair.source_text}\n"
        f"translated_text: {pair.translated_text}\n"
        f"target_language: {pair.target_language or 'same language as translated_text'}\n"
        f"risk_level: {pair.risk_level}\n"
        f"candidate_errors: {[error.to_dict() for error in errors]}\n"
    )
    context = image_context_for_prompt(pair)
    return prompt + (f"image_context: {json.dumps(context, ensure_ascii=False)}\n" if context else "")


def _batch_detail_prompt(pairs: list[TextPair], errors: list[TextError]) -> str:
    return json.dumps(
        {
            "mode": "batch",
            "output_language": "zh-CN",
            "instruction": "Refine each candidate independently. Preserve original_error_id and include pair_id on every update or new error.",
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "source_text": pair.source_text,
                    "translated_text": pair.translated_text,
                    "target_language": pair.target_language,
                    "risk_level": pair.risk_level,
                    **({"image_context": context} if (context := image_context_for_prompt(pair)) else {}),
                }
                for pair in pairs
            ],
            "candidate_errors": [error.to_dict() for error in errors],
        },
        ensure_ascii=False,
        indent=2,
    )

