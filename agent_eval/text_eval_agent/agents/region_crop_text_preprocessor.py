# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visual_eval_agent.llm_client import VisionModelClient
from visual_eval_agent.runtime import VisionRuntimeConfig

from ..prompt_loader import load_prompt
from ..region_crop_validation import (
    validate_region_response,
    validate_translations,
)


class RegionCropTextPreprocessorAgent:
    def __init__(self, model_client: VisionModelClient, runtime: VisionRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def extract(
        self,
        image_path: Path,
        task_id: str,
        translations: list[dict],
        *,
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise ValueError("region crop must be an existing image")
        candidates = validate_translations(translations)
        ordered = [
            {
                **candidates[pid],
                "sequence_index": index,
            }
            for index, pid in enumerate(candidates)
        ]
        payload = {
            "task_id": task_id,
            "source_language": source_language,
            "target_language": target_language,
            "translations": ordered,
            "instruction": "Return only the strict JSON object defined by the system prompt.",
        }
        response = self.model_client.complete_json(
            "region_crop_text_preprocessor",
            [
                {
                    "role": "system",
                    "content": load_prompt(
                        "region_crop_text_preprocessor", include_policy=False
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            [path],
            self.runtime,
        )
        return validate_region_response(response, candidates)
