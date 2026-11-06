# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from visual_eval_agent.llm_client import VisionModelClient
from visual_eval_agent.runtime import VisionRuntimeConfig

from ..agents.image_text_preprocessor import _normalize_ids
from ..image_models import ImageTextExtraction
from ..prompt_loader import load_skill_instructions


PAIRING_CONFIDENCE_THRESHOLD = 0.80
MAX_UNEXPLAINED_BLOCK_RATIO = 2.0


def pairing_trigger_reasons(
    extraction: ImageTextExtraction,
    confidence_threshold: float = PAIRING_CONFIDENCE_THRESHOLD,
) -> list[str]:
    reasons: list[str] = []
    if any(unit.pair_confidence < confidence_threshold for unit in extraction.evaluation_units):
        reasons.append("low_pair_confidence")
    if _has_clear_unmatched(extraction, "source"):
        reasons.append("unmatched_clear_source_blocks")
    if _has_clear_unmatched(extraction, "target"):
        reasons.append("unmatched_clear_target_blocks")
    if extraction.reading_order_uncertain:
        reasons.append("reading_order_uncertain")
    if extraction.complex_layout:
        reasons.append("complex_layout")
    if _has_unexplained_block_ratio(extraction):
        reasons.append("unexplained_block_count_ratio")
    return reasons


class TextPairingDetailSkill:
    """Refine extraction, grouping, reading order, and alignment only."""

    def __init__(self, model_client: VisionModelClient, runtime: VisionRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def refine(
        self,
        source_image: Path,
        translated_image: Path,
        task_id: str,
        candidate: ImageTextExtraction,
        trigger_reasons: list[str],
    ) -> ImageTextExtraction:
        image_paths = [Path(source_image), Path(translated_image)]
        if any(not path.is_file() for path in image_paths):
            raise ValueError("Text pairing detail requires exactly two existing images")
        response = self.model_client.complete_json(
            "text_pairing_detail",
            [
                {
                    "role": "system",
                    "content": load_skill_instructions("text-pairing-detail", include_policy=False),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task_id": task_id,
                            "image_order": ["source_image", "translated_image"],
                            "trigger_reasons": trigger_reasons,
                            "candidate_extraction": candidate.to_dict(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            image_paths,
            self.runtime,
        )
        return ImageTextExtraction.from_dict(_normalize_ids(response))


def _has_clear_unmatched(extraction: ImageTextExtraction, side: str) -> bool:
    blocks = extraction.source_blocks if side == "source" else extraction.target_blocks
    unmatched = (
        set(extraction.unmatched_source_block_ids)
        if side == "source"
        else set(extraction.unmatched_target_block_ids)
    )
    return any(block.block_id in unmatched and block.visibility == "clear" for block in blocks)


def _has_unexplained_block_ratio(extraction: ImageTextExtraction) -> bool:
    source_count = len(extraction.source_blocks)
    target_count = len(extraction.target_blocks)
    if not source_count or not target_count:
        return False
    ratio = max(source_count, target_count) / min(source_count, target_count)
    if ratio <= MAX_UNEXPLAINED_BLOCK_RATIO:
        return False
    return all(unit.alignment_type == "one_to_one" for unit in extraction.evaluation_units)
