# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from ..consistency.layout_controller import normalize_layout_items
from ..llm_client import VisionModelClient
from ..models import ImagePair, VisualError
from ..prompt_loader import load_prompt
from ..runtime import VisionRuntimeConfig


class LayoutAgent:
    """Lightweight model-backed evaluator for translated-text layout."""

    def __init__(self, model_client: VisionModelClient, runtime: VisionRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def evaluate(self, image_pair: ImagePair) -> list[VisualError]:
        response = self.model_client.complete_json(
            "layout",
            [
                {"role": "system", "content": load_prompt("layout")},
                {"role": "user", "content": _image_pair_prompt(image_pair)},
            ],
            [image_pair.source_image, image_pair.translated_image],
            self.runtime,
        )
        items = []
        for raw_item in response.get("errors", []):
            item = dict(raw_item)
            item.pop("suggestion", None)
            item.pop("needs_skill", None)
            item.pop("source", None)
            item["risk_level"] = _risk_level_for_region(
                image_pair, item.get("region_id")
            )
            items.append(item)
        return normalize_layout_items(
            items,
            task_id=image_pair.task_id,
            default_source="layout_agent",
        )


def _image_pair_prompt(image_pair: ImagePair) -> str:
    context = {
        "task_id": image_pair.task_id,
        "image_order": ["original_image", "translated_image"],
        "source_language": image_pair.source_language,
        "target_language": image_pair.target_language,
        "output_language": image_pair.output_language,
        "original_text_regions": image_pair.original_text_regions,
        "translated_text_regions": image_pair.translated_text_regions,
        "risk_regions": image_pair.risk_regions,
        "metadata": image_pair.metadata,
    }
    return "Evaluate the attached image pair using this request context:\n" + json.dumps(
        context, ensure_ascii=False, indent=2
    )


def _risk_level_for_region(image_pair: ImagePair, region_id: str | None) -> str:
    if region_id is None:
        return "normal"
    for region in image_pair.risk_regions:
        if region.get("region_id") == region_id:
            return str(region.get("risk_level") or "normal")
    return "normal"
