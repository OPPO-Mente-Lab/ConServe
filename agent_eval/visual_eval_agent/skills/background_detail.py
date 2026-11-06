# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

from ..consistency.background_controller import normalize_background_items
from ..llm_client import VisionModelClient, VisionModelClientError
from ..models import ImagePair, Severity, VisualError
from ..prompt_loader import load_skill_instructions
from ..runtime import VisionRuntimeConfig
from .refinement_result import RefinementResult, normalized_rejections


class BackgroundDetailSkill:
    """Model-backed detail review for Background candidates and risk regions."""

    def __init__(self, model_client: VisionModelClient, runtime: VisionRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def refine(
        self,
        image_pair: ImagePair,
        errors: list[VisualError],
        detail_risk_regions: list[dict] | None = None,
    ) -> RefinementResult:
        response = self.model_client.complete_json(
            "background_detail",
            [
                {
                    "role": "system",
                    "content": load_skill_instructions(
                        "background-detail", include_references=True
                    ),
                },
                {
                    "role": "user",
                    "content": _detail_prompt(
                        image_pair, errors, detail_risk_regions or []
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
        merged_error_ids = {
            str(error_id)
            for group in response.get("root_cause_groups", [])
            for error_id in group.get("merged_error_ids", [])
            if error_id and str(error_id) != str(group.get("primary_error_id"))
        }

        refined_items: list[dict] = []
        for error in errors:
            if error.error_id in merged_error_ids:
                continue
            rejection = rejections.get(error.error_id)
            if rejection and _supported_rejection(error, rejection):
                continue
            update = updates.get(error.error_id)
            if update:
                payload = error.to_dict()
                payload.update(update)
                payload.pop("original_error_id", None)
                payload["suggestion"] = _required_suggestion(update)
                payload["source"] = "background_detail_skill"
                payload["needs_skill"] = False
                refined_items.append(payload)
                continue
            refined_items.append(
                replace(
                    error,
                    suggestion=_fallback_suggestion(
                        error.type, image_pair.output_language
                    ),
                    source="background_consistency_controller",
                    needs_skill=False,
                ).to_dict()
            )

        for item in response.get("new_errors", []):
            payload = dict(item)
            payload["suggestion"] = _required_suggestion(payload)
            payload["source"] = "background_detail_skill"
            payload["needs_skill"] = False
            refined_items.append(payload)

        return RefinementResult(
            errors=normalize_background_items(
                refined_items,
                task_id=image_pair.task_id,
                default_source="background_detail_skill",
                calibrate_severity=False,
            ),
            rejected_errors=rejection_records,
            notes=[str(note) for note in response.get("notes", [])],
        )


def _required_suggestion(item: dict) -> str:
    suggestion = item.get("suggestion")
    if not isinstance(suggestion, str) or not suggestion.strip():
        raise VisionModelClientError(
            "Background Detail Skill must provide suggestion for every updated or new error"
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
            "smearing_artifact": "重新修复涂抹区域，使颜色与纹理自然延续。",
            "color_discontinuity": "匹配周围背景颜色，消除局部色块和颜色断层。",
            "gradient_discontinuity": "按原图方向重建连续渐变，避免亮度或色相突变。",
            "texture_discontinuity": "延续邻近纹理的方向、密度和尺度，修复中断区域。",
            "hard_inpainting_edge": "柔化修复边界并与周围像素自然融合。",
            "inpainting_residue": "清除残影或边缘痕迹，并恢复原有背景细节。",
            "scene_element_damage": "依据原图恢复受损的主体、图标或装饰边缘。",
            "missing_scene_content": "依据原图补回被误删的场景内容。",
            "unexpected_generated_content": "移除异常生成内容，并重建与原图一致的背景。",
            "pattern_repetition_artifact": "打散异常重复纹理，并恢复自然的图案连续性。",
        }
        return suggestions[error_type]
    return "Reconstruct the affected region to match the original scene and surrounding texture."


def _detail_prompt(
    image_pair: ImagePair,
    errors: list[VisualError],
    detail_risk_regions: list[dict],
) -> str:
    context = {
        "task_id": image_pair.task_id,
        "image_order": ["original_image", "translated_image"],
        "output_language": image_pair.output_language,
        "risk_regions": image_pair.risk_regions,
        "candidate_errors": [error.to_dict() for error in errors],
        "detail_risk_regions": detail_risk_regions,
    }
    return "Refine the Background candidates and risk regions using the attached image pair:\n" + json.dumps(
        context, ensure_ascii=False, indent=2
    )
