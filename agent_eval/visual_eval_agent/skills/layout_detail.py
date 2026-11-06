# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

from ..consistency.layout_controller import normalize_layout_items
from ..llm_client import VisionModelClient, VisionModelClientError
from ..models import ImagePair, Severity, VisualError
from ..prompt_loader import load_skill_instructions
from ..runtime import VisionRuntimeConfig
from .refinement_result import RefinementResult, normalized_rejections


class LayoutDetailSkill:
    """Model-backed detailed review for Layout candidate errors."""

    def __init__(self, model_client: VisionModelClient, runtime: VisionRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def refine(
        self,
        image_pair: ImagePair,
        errors: list[VisualError],
        review_focus: str | None = None,
    ) -> RefinementResult:
        response = self.model_client.complete_json(
            "layout_detail",
            [
                {
                    "role": "system",
                    "content": load_skill_instructions(
                        "layout-detail", include_references=True
                    ),
                },
                {
                    "role": "user",
                    "content": _detail_prompt(
                        image_pair,
                        errors,
                        review_focus=review_focus,
                    ),
                },
            ],
            [image_pair.source_image, image_pair.translated_image],
            self.runtime,
        )
        rejection_records = normalized_rejections(response.get("rejected_errors", []))
        rejections = {
            str(item.get("original_error_id")): item
            for item in rejection_records
            if item.get("original_error_id")
        }
        updates = {
            str(item.get("original_error_id")): item
            for item in response.get("updated_errors", [])
            if item.get("original_error_id")
        }

        refined_items: list[dict] = []
        for error in errors:
            rejection = rejections.get(error.error_id)
            if rejection:
                if _supported_rejection(error, rejection):
                    continue
                preserved = replace(
                    error,
                    suggestion=_fallback_suggestion(
                        error.type, image_pair.output_language
                    ),
                    source="layout_consistency_controller",
                    needs_skill=False,
                )
                refined_items.append(preserved.to_dict())
                continue
            update = updates.get(error.error_id)
            if not update:
                refined_items.append(replace(error, needs_skill=False).to_dict())
                continue
            payload = error.to_dict()
            payload.update(update)
            payload.pop("original_error_id", None)
            payload["suggestion"] = _required_suggestion(update)
            payload["source"] = "layout_detail_skill"
            payload["needs_skill"] = False
            refined_items.append(payload)

        for item in response.get("new_errors", []):
            payload = dict(item)
            payload["suggestion"] = _required_suggestion(payload)
            payload["source"] = "layout_detail_skill"
            payload["needs_skill"] = False
            refined_items.append(payload)

        return RefinementResult(
            errors=normalize_layout_items(
                refined_items,
                task_id=image_pair.task_id,
                default_source="layout_detail_skill",
            ),
            rejected_errors=rejection_records,
            notes=[str(note) for note in response.get("notes", [])],
        )


def _required_suggestion(item: dict) -> str:
    suggestion = item.get("suggestion")
    if not isinstance(suggestion, str) or not suggestion.strip():
        raise VisionModelClientError(
            "Layout Detail Skill must provide suggestion for every updated or new error"
        )
    return suggestion.strip()


def _supported_rejection(error: VisualError, rejection: dict) -> bool:
    evidence = rejection.get("evidence")
    try:
        confidence = float(rejection.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    threshold = 0.85
    if error.severity in {Severity.MAJOR, Severity.CRITICAL} and error.confidence >= 0.9:
        threshold = 0.95
    return confidence >= threshold


def _fallback_suggestion(error_type: str, output_language: str) -> str:
    if output_language.lower().startswith("zh"):
        suggestions = {
            "reading_direction_mismatch": "保持字符正向，并按容器形状和画面阅读流重新排列文字。",
            "reading_order_error": "重新排列文本块，使阅读顺序连续且明确。",
            "unreasonable_line_break": "调整换行位置，保持词组和句法结构完整。",
            "text_overflow": "调整文本块或容器尺寸，使译文完整保留在目标区域内。",
            "position_shift": "将译文块移回与原文对应的目标区域。",
            "alignment_error": "按目标容器重新对齐译文块。",
            "text_region_crowding": "增加文本区域的行距、字距或块间距。",
            "content_occlusion": "移动或重排译文，避免遮挡受影响的视觉内容。",
            "wrong_region_fill": "将译文移动到与源文本正确对应的区域。",
        }
        return suggestions[error_type]
    return "Reposition and arrange the translated text to restore a natural, unobstructed reading flow."


def _detail_prompt(
    image_pair: ImagePair,
    errors: list[VisualError],
    review_focus: str | None = None,
) -> str:
    context = {
        "task_id": image_pair.task_id,
        "image_order": ["original_image", "translated_image"],
        "source_language": image_pair.source_language,
        "target_language": image_pair.target_language,
        "output_language": image_pair.output_language,
        "risk_regions": image_pair.risk_regions,
        "original_text_regions": image_pair.original_text_regions,
        "translated_text_regions": image_pair.translated_text_regions,
        "candidate_errors": [error.to_dict() for error in errors],
    }
    if review_focus:
        context["review_focus"] = review_focus
    return "Refine the Layout candidates using the attached image pair:\n" + json.dumps(
        context, ensure_ascii=False, indent=2
    )
