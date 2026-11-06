# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

from ..llm_client import VisionModelClient, VisionModelClientError
from ..models import ImagePair, Severity, VisualError, ensure_unique_error_ids
from ..prompt_loader import load_skill_instructions
from ..runtime import VisionRuntimeConfig
from ..consistency.text_rendering_controller import calibrate_text_rendering_errors
from .refinement_result import RefinementResult, normalized_rejections


class TextRenderingDetailSkill:
    """Model-backed detail review for Text Rendering candidate errors."""

    def __init__(self, model_client: VisionModelClient, runtime: VisionRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def refine(
        self,
        image_pair: ImagePair,
        errors: list[VisualError],
        target_region_ids: list[str] | None = None,
    ) -> RefinementResult:
        response = self.model_client.complete_json(
            "text_rendering_detail",
            [
                {
                    "role": "system",
                    "content": load_skill_instructions(
                        "text-rendering-detail", include_references=True
                    ),
                },
                {
                    "role": "user",
                    "content": _detail_prompt(image_pair, errors, target_region_ids or []),
                },
            ],
            [image_pair.source_image, image_pair.translated_image],
            self.runtime,
        )
        rejection_records = normalized_rejections(response.get("rejected_errors", []))
        rejected_ids = {
            str(item.get("original_error_id"))
            for item in rejection_records
            if item.get("original_error_id")
        }
        updates = {
            str(item.get("original_error_id")): item
            for item in response.get("updated_errors", [])
            if item.get("original_error_id")
        }
        refined: list[VisualError] = []
        for error in errors:
            if error.error_id in rejected_ids:
                continue
            update = updates.get(error.error_id)
            if not update:
                refined.append(replace(error, needs_skill=False))
                continue
            suggestion = _required_suggestion(update)
            refined.append(
                replace(
                    error,
                    type=str(update.get("type", error.type)),
                    severity=Severity(update.get("severity", error.severity.value)),
                    penalty=None,
                    region_id=update.get("region_id", error.region_id),
                    bbox_1000=update.get("bbox_1000", error.bbox_1000),
                    bbox_confidence=float(
                        update.get("bbox_confidence", error.bbox_confidence)
                    ),
                    localization_status=str(
                        update.get("localization_status", error.localization_status)
                    ),
                    evidence=str(update.get("evidence", error.evidence)),
                    suggestion=suggestion,
                    confidence=float(update.get("confidence", error.confidence)),
                    comparison=dict(update.get("comparison", error.comparison)),
                    severity_factors=dict(
                        update.get("severity_factors", error.severity_factors)
                    ),
                    source="text_rendering_detail_skill",
                    needs_skill=False,
                )
            )

        for item in response.get("new_errors", []):
            new_item = dict(item)
            new_item["suggestion"] = _required_suggestion(new_item)
            refined.append(
                VisualError.from_model_dict(
                    new_item,
                    image_pair.task_id,
                    "text_rendering_detail_skill",
                )
            )
        return RefinementResult(
            errors=calibrate_text_rendering_errors(ensure_unique_error_ids(refined)),
            rejected_errors=rejection_records,
            notes=[str(note) for note in response.get("notes", [])],
        )


def _required_suggestion(item: dict) -> str:
    suggestion = item.get("suggestion")
    if not isinstance(suggestion, str) or not suggestion.strip():
        raise VisionModelClientError(
            "Text Rendering Detail Skill must provide suggestion for every updated or new error"
        )
    return suggestion.strip()


def _detail_prompt(
    image_pair: ImagePair,
    errors: list[VisualError],
    target_region_ids: list[str],
) -> str:
    context = {
        "task_id": image_pair.task_id,
        "image_order": ["original_image", "translated_image"],
        "source_language": image_pair.source_language,
        "target_language": image_pair.target_language,
        "output_language": image_pair.output_language,
        "risk_regions": image_pair.risk_regions,
        "translated_text_regions": image_pair.translated_text_regions,
        "target_region_ids": target_region_ids,
        "candidate_errors": [error.to_dict() for error in errors],
    }
    return "Refine the candidate errors using the attached image pair:\n" + json.dumps(
        context, ensure_ascii=False, indent=2
    )
