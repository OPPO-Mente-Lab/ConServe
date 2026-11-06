# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from ..llm_client import ModelClient
from ..models import TextDimension, TextError, TextPair, ensure_unique_error_ids, image_context_for_prompt
from ..prompt_loader import load_prompt
from ..policy import is_non_scored_error_type
from ..runtime import ModelRuntimeConfig


class FaithfulnessAgent:
    """Model-backed source-to-target information transfer evaluator."""

    def __init__(self, model_client: ModelClient, runtime: ModelRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def evaluate(self, pair: TextPair) -> list[TextError]:
        response = self.model_client.complete_json(
            "faithfulness",
            [
                {"role": "system", "content": load_prompt("faithfulness")},
                {"role": "user", "content": _pair_prompt(pair)},
            ],
            self.runtime,
        )
        errors = [
            TextError.from_model_dict(item, pair, TextDimension.FAITHFULNESS, "faithfulness_agent")
            for item in response.get("errors", [])
            if not is_non_scored_error_type(item.get("type"))
        ]
        return ensure_unique_error_ids(errors)

    def evaluate_many(self, pairs: list[TextPair]) -> list[TextError]:
        if not pairs:
            return []
        pair_by_id = {pair.pair_id: pair for pair in pairs}
        response = self.model_client.complete_json(
            "faithfulness",
            [
                {"role": "system", "content": load_prompt("faithfulness")},
                {"role": "user", "content": _pairs_prompt(pairs)},
            ],
            self.runtime,
        )
        errors = []
        for item in response.get("errors", []):
            pair_id = str(item.get("pair_id") or (pairs[0].pair_id if len(pairs) == 1 else ""))
            pair = pair_by_id.get(pair_id)
            if pair is None or is_non_scored_error_type(item.get("type")):
                continue
            errors.append(TextError.from_model_dict(item, pair, TextDimension.FAITHFULNESS, "faithfulness_agent"))
        return ensure_unique_error_ids(errors)

def _pair_prompt(pair: TextPair) -> str:
    prompt = (
        "output_language: zh-CN\n"
        f"pair_id: {pair.pair_id}\n"
        f"source_text: {pair.source_text}\n"
        f"translated_text: {pair.translated_text}\n"
        f"pair_confidence: {pair.pair_confidence}\n"
        f"risk_level: {pair.risk_level}\n"
        f"risk_type: {pair.risk_type}\n"
    )
    context = image_context_for_prompt(pair)
    return prompt + (f"image_context: {json.dumps(context, ensure_ascii=False)}\n" if context else "")


def _pairs_prompt(pairs: list[TextPair]) -> str:
    return json.dumps(
        {
            "mode": "batch",
            "output_language": "zh-CN",
            "instruction": "Evaluate every pair independently. Every returned error must include its exact pair_id.",
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "source_text": pair.source_text,
                    "translated_text": pair.translated_text,
                    "source_language": pair.source_language,
                    "target_language": pair.target_language,
                    "pair_confidence": pair.pair_confidence,
                    "risk_level": pair.risk_level,
                    "risk_type": pair.risk_type,
                    **({"image_context": context} if (context := image_context_for_prompt(pair)) else {}),
                }
                for pair in pairs
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

