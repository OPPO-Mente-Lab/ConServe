# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from visual_eval_agent.llm_client import VisionModelClient, VisionModelClientError
from visual_eval_agent.runtime import VisionRuntimeConfig

from ..image_models import ImageTextExtraction
from ..image_transport import prepare_image_pair
from ..prompt_loader import load_prompt


class ImageTextPreprocessorAgent:
    """Extract and align visible text from one source/translated image pair."""

    def __init__(self, model_client: VisionModelClient, runtime: VisionRuntimeConfig):
        self.model_client = model_client
        self.runtime = runtime

    def extract(self, source_image: Path, translated_image: Path, task_id: str) -> ImageTextExtraction:
        image_paths = [Path(source_image), Path(translated_image)]
        if any(not path.is_file() for path in image_paths):
            raise ValueError("Image text preprocessing requires exactly two existing images")
        with tempfile.TemporaryDirectory(prefix="image-text-transport-") as directory:
            transport_paths = (
                prepare_image_pair(
                    image_paths,
                    Path(directory),
                    max_edge=self.runtime.image_max_edge,
                    jpeg_quality=self.runtime.image_jpeg_quality,
                )
                if self.runtime.optimize_image_transport
                else image_paths
            )
            try:
                response = self._request(
                    "image_text_preprocessor", transport_paths, task_id
                )
                return ImageTextExtraction.from_dict(_normalize_ids(response))
            except (VisionModelClientError, OSError, ValueError):
                response = self._request_segmented(transport_paths, task_id)
                return ImageTextExtraction.from_dict(_normalize_ids(response))

    def _request(
        self,
        component: str,
        image_paths: list[Path],
        task_id: str,
        focus_region: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_payload: dict[str, Any] = {
            "task_id": task_id,
            "image_order": ["source_image", "translated_image"],
            "instruction": "Return only the strict JSON object defined by the system prompt.",
        }
        if focus_region is not None:
            user_payload.update(
                {
                    "mode": "focused_region",
                    "focus_region": focus_region,
                    "focus_instruction": (
                        "Extract only blocks whose bbox center lies inside focus_region.bbox_1000. "
                        "Keep bbox coordinates normalized to the full image, not the region."
                    ),
                }
            )
        return self.model_client.complete_json(
            component,
            [
                {"role": "system", "content": load_prompt("image_text_preprocessor", include_policy=False)},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            image_paths,
            self.runtime,
        )

    def _request_segmented(
        self, image_paths: list[Path], task_id: str
    ) -> dict[str, Any]:
        payloads = [
            self._request(
                "image_text_preprocessor_segment",
                image_paths,
                task_id,
                focus_region=region,
            )
            for region in FOCUS_REGIONS
        ]
        return _merge_segment_payloads(payloads)


FOCUS_REGIONS = [
    {"region_id": "top_left", "bbox_1000": [0, 0, 500, 500]},
    {"region_id": "top_right", "bbox_1000": [0, 500, 500, 1000]},
    {"region_id": "bottom_left", "bbox_1000": [500, 0, 1000, 500]},
    {"region_id": "bottom_right", "bbox_1000": [500, 500, 1000, 1000]},
]


def _merge_segment_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "source_blocks": [],
        "target_blocks": [],
        "evaluation_units": [],
        "unmatched_source_block_ids": [],
        "unmatched_target_block_ids": [],
        "reading_order_uncertain": False,
        "complex_layout": True,
        "notes": ["segmented_fallback"],
    }
    for segment_index, payload in enumerate(payloads, start=1):
        tagged = _tag_segment_payload(payload, segment_index)
        for key in (
            "source_blocks",
            "target_blocks",
            "evaluation_units",
            "unmatched_source_block_ids",
            "unmatched_target_block_ids",
        ):
            merged[key].extend(tagged.get(key, []))
        merged["reading_order_uncertain"] = bool(
            merged["reading_order_uncertain"] or payload.get("reading_order_uncertain")
        )
        merged["notes"].extend(str(note) for note in payload.get("notes", []))
    return merged


def _tag_segment_payload(payload: dict[str, Any], segment_index: int) -> dict[str, Any]:
    tagged = dict(payload)
    source_blocks, source_map = _tag_blocks(payload.get("source_blocks", []), segment_index, "source")
    target_blocks, target_map = _tag_blocks(payload.get("target_blocks", []), segment_index, "target")
    tagged["source_blocks"] = source_blocks
    tagged["target_blocks"] = target_blocks
    units = []
    for unit_index, raw_unit in enumerate(payload.get("evaluation_units", []), start=1):
        unit = dict(raw_unit)
        unit["unit_id"] = f"seg{segment_index}-unit-{unit_index}"
        unit["source_sequence"] = _remap_sequence(raw_unit.get("source_sequence", []), source_map)
        unit["target_sequence"] = _remap_sequence(raw_unit.get("target_sequence", []), target_map)
        units.append(unit)
    tagged["evaluation_units"] = units
    tagged["unmatched_source_block_ids"] = [
        source_map.get(str(block_id), str(block_id))
        for block_id in payload.get("unmatched_source_block_ids", [])
    ]
    tagged["unmatched_target_block_ids"] = [
        target_map.get(str(block_id), str(block_id))
        for block_id in payload.get("unmatched_target_block_ids", [])
    ]
    return tagged


def _tag_blocks(
    blocks: list[dict[str, Any]], segment_index: int, side: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    id_map: dict[str, str] = {}
    tagged = []
    for local_index, raw_block in enumerate(blocks, start=1):
        old_id = str(raw_block.get("block_id", f"missing-{local_index}"))
        new_id = f"seg{segment_index}-{side}-{local_index}"
        id_map[old_id] = new_id
        block = dict(raw_block)
        block["block_id"] = new_id
        block["reading_order"] = segment_index * 1000 + int(raw_block.get("reading_order", local_index))
        tagged.append(block)
    return tagged, id_map


def _remap_sequence(
    sequence: list[dict[str, Any]], id_map: dict[str, str]
) -> list[dict[str, Any]]:
    result = []
    for raw_reference in sequence:
        reference = dict(raw_reference)
        old_id = str(reference.get("block_id", ""))
        reference["block_id"] = id_map.get(old_id, old_id)
        result.append(reference)
    return result


def _normalize_ids(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    source_blocks, source_map = _normalize_blocks(payload.get("source_blocks", []), "S")
    target_blocks, target_map = _normalize_blocks(payload.get("target_blocks", []), "T")
    normalized["source_blocks"] = source_blocks
    normalized["target_blocks"] = target_blocks
    units = []
    for index, raw_unit in enumerate(payload.get("evaluation_units", []), start=1):
        unit = dict(raw_unit)
        unit["unit_id"] = f"unit_{index:02d}"
        unit["source_sequence"] = _normalize_sequence(raw_unit.get("source_sequence", []), source_map)
        unit["target_sequence"] = _normalize_sequence(raw_unit.get("target_sequence", []), target_map)
        units.append(unit)
    normalized["evaluation_units"] = units
    normalized["unmatched_source_block_ids"] = [
        source_map.get(str(block_id), str(block_id))
        for block_id in payload.get("unmatched_source_block_ids", [])
    ]
    normalized["unmatched_target_block_ids"] = [
        target_map.get(str(block_id), str(block_id))
        for block_id in payload.get("unmatched_target_block_ids", [])
    ]
    return normalized


def _normalize_blocks(raw_blocks: list[dict[str, Any]], prefix: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    ordered = sorted(
        enumerate(raw_blocks),
        key=lambda item: (int(item[1].get("reading_order", item[0] + 1)), item[0]),
    )
    id_map: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for new_index, (_, raw_block) in enumerate(ordered, start=1):
        old_id = str(raw_block.get("block_id", f"{prefix}-missing-{new_index}"))
        new_id = f"{prefix}{new_index:02d}"
        id_map[old_id] = new_id
        block = dict(raw_block)
        block["block_id"] = new_id
        block["reading_order"] = new_index
        block["bbox_1000"] = _normalize_bbox(block.get("bbox_1000"))
        normalized.append(block)
    return normalized, id_map


def _normalize_sequence(raw_sequence: list[dict[str, Any]], id_map: dict[str, str]) -> list[dict[str, Any]]:
    result = []
    for raw_reference in raw_sequence:
        reference = dict(raw_reference)
        old_id = str(reference.get("block_id", ""))
        reference["block_id"] = id_map.get(old_id, old_id)
        if not str(reference.get("join_after", "")).strip():
            reference["join_after"] = "none"
        result.append(reference)
    return result


def _normalize_bbox(value: Any) -> Any:
    if not isinstance(value, list) or len(value) != 4:
        return value
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        return value
    ymin, xmin, ymax, xmax = [max(0, min(1000, int(round(item)))) for item in value]
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin == ymax:
        ymin, ymax = ((ymin - 1, ymax) if ymax == 1000 else (ymin, ymax + 1))
    if xmin == xmax:
        xmin, xmax = ((xmin - 1, xmax) if xmax == 1000 else (xmin, xmax + 1))
    return [ymin, xmin, ymax, xmax]
