# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

from ..llm_client import ModelClient
from ..models import TextDimension, TextError, TextJudgeResult
from ..prompt_loader import load_prompt
from ..runtime import ModelRuntimeConfig


class TextJudge:
    """Model-backed text ledger reviewer for deduplication, attribution, severity, and caps."""

    def __init__(self, model_client: ModelClient, runtime: ModelRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def review(self, errors: list[TextError]) -> TextJudgeResult:
        response = self.model_client.complete_json(
            "text_judge",
            [
                {"role": "system", "content": load_prompt("text_judge")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "output_language": "zh-CN",
                            "candidate_errors": [error.to_dict() for error in errors],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            self.runtime,
        )
        original_by_id = {error.error_id: error for error in errors}
        final_errors: list[TextError] = []
        for item in response.get("errors", []):
            final_error = TextError.from_model_dict(item, None, TextDimension.FAITHFULNESS, "text_judge")
            original = original_by_id.get(final_error.error_id)
            if original:
                final_error = replace(
                    final_error,
                    source_fragment=final_error.source_fragment or original.source_fragment,
                    target_fragment=final_error.target_fragment or original.target_fragment,
                    suggestion=final_error.suggestion or original.suggestion,
                )
            final_errors.append(final_error)
        return TextJudgeResult(
            errors=final_errors,
            deduplicated_errors=list(response.get("deduplicated_errors", [])),
            severity_updates=list(response.get("severity_updates", [])),
            caps=list(response.get("caps", [])),
        )
