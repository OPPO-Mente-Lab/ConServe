# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..consistency.background_controller import normalize_background_items
from ..llm_client import VisionModelClient
from ..models import ImagePair, VisualError
from ..prompt_loader import load_prompt
from ..runtime import VisionRuntimeConfig


BACKGROUND_RISK_REASONS = {
    "primary_subject_edge",
    "scene_element_edge",
    "icon_or_decoration_edge",
    "complex_texture",
    "gradient_background",
    "patterned_background",
}


@dataclass(frozen=True)
class BackgroundAssessment:
    errors: list[VisualError]
    detail_risk_regions: list[dict[str, Any]]


class BackgroundAgent:
    """Lightweight model-backed evaluator for repaired image backgrounds."""

    def __init__(self, model_client: VisionModelClient, runtime: VisionRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def evaluate(self, image_pair: ImagePair) -> BackgroundAssessment:
        response = self.model_client.complete_json(
            "background",
            [
                {"role": "system", "content": load_prompt("background")},
                {"role": "user", "content": _image_pair_prompt(image_pair)},
            ],
            [image_pair.source_image, image_pair.translated_image],
            self.runtime,
        )
        items = []
        for raw_item in response.get("errors", []):
            item = dict(raw_item)
            item.pop("suggestion", None)
            item.pop("source", None)
            item["risk_level"] = _risk_level_for_region(
                image_pair, item.get("region_id")
            )
            items.append(item)
        errors = normalize_background_items(
            items,
            task_id=image_pair.task_id,
            default_source="background_agent",
            calibrate_severity=False,
        )
        detail_risk_regions = _normalize_detail_risk_regions(
            response.get("detail_risk_regions", [])
        )
        return BackgroundAssessment(errors, detail_risk_regions)


def _normalize_detail_risk_regions(raw_regions: list[dict]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for raw in raw_regions or []:
        reason = str(raw.get("reason") or "")
        if reason not in BACKGROUND_RISK_REASONS:
            continue
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < 0.0 or confidence > 1.0:
            continue
        bbox = raw.get("bbox_1000")
        if not _valid_risk_bbox(bbox):
            continue
        regions.append(
            {
                "region_id": raw.get("region_id"),
                "reason": reason,
                "bbox_1000": bbox,
                "confidence": confidence,
            }
        )
    return regions


def _valid_risk_bbox(bbox: object) -> bool:
    if not isinstance(bbox, dict):
        return False
    coordinates = bbox.get("affected_area")
    return bool(
        isinstance(coordinates, list)
        and len(coordinates) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in coordinates)
        and all(0 <= value <= 1000 for value in coordinates)
        and coordinates[0] < coordinates[2]
        and coordinates[1] < coordinates[3]
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
