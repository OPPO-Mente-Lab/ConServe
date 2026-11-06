# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .image_models import BlockJoin, EvaluationUnitCandidate, ImageTextBlock, ImageTextExtraction
from .models import TextPair


SEPARATORS = {
    "none": "",
    "space": " ",
    "newline": "\n",
    "paragraph": "\n\n",
}


def assemble_sequence(
    blocks_by_id: dict[str, ImageTextBlock], sequence: list[BlockJoin]
) -> str:
    pieces: list[str] = []
    for index, reference in enumerate(sequence):
        pieces.append(blocks_by_id[reference.block_id].text)
        if index < len(sequence) - 1:
            pieces.append(SEPARATORS[reference.join_after])
    return "".join(pieces)


def union_bbox(blocks: list[ImageTextBlock]) -> list[int] | None:
    if not blocks:
        return None
    return [
        min(block.bbox_1000[0] for block in blocks),
        min(block.bbox_1000[1] for block in blocks),
        max(block.bbox_1000[2] for block in blocks),
        max(block.bbox_1000[3] for block in blocks),
    ]


def build_text_pairs(
    task_id: str,
    extraction: ImageTextExtraction,
    source_language: str | None = None,
    target_language: str | None = None,
) -> list[TextPair]:
    source_by_id = {block.block_id: block for block in extraction.source_blocks}
    target_by_id = {block.block_id: block for block in extraction.target_blocks}
    units = sorted(
        extraction.evaluation_units,
        key=lambda unit: _unit_reading_order(unit, source_by_id),
    )
    pairs: list[TextPair] = []
    for unit in units:
        source_blocks = [source_by_id[item.block_id] for item in unit.source_sequence]
        target_blocks = [target_by_id[item.block_id] for item in unit.target_sequence]
        pairs.append(
            TextPair(
                pair_id=f"{task_id}:{unit.unit_id}",
                source_text=assemble_sequence(source_by_id, unit.source_sequence),
                translated_text=assemble_sequence(target_by_id, unit.target_sequence),
                source_language=source_language or _dominant_language(source_blocks),
                target_language=target_language or _dominant_language(target_blocks),
                source_bbox=union_bbox(source_blocks),
                translated_bbox=union_bbox(target_blocks),
                pair_confidence=unit.pair_confidence,
                source_ocr_confidence=_minimum_confidence(source_blocks),
                translated_ocr_confidence=_minimum_confidence(target_blocks),
                metadata={
                    "unit_id": unit.unit_id,
                    "unit_type": unit.unit_type,
                    "alignment_type": unit.alignment_type,
                    "source_block_ids": [block.block_id for block in source_blocks],
                    "target_block_ids": [block.block_id for block in target_blocks],
                    "source_reading_order": [block.reading_order for block in source_blocks],
                    "target_reading_order": [block.reading_order for block in target_blocks],
                    "pairing_warnings": list(unit.warnings),
                },
            )
        )
    return pairs


def _unit_reading_order(
    unit: EvaluationUnitCandidate, source_by_id: dict[str, ImageTextBlock]
) -> int:
    return min(source_by_id[item.block_id].reading_order for item in unit.source_sequence)


def _minimum_confidence(blocks: list[ImageTextBlock]) -> float | None:
    return min((block.confidence for block in blocks), default=None)


def _dominant_language(blocks: list[ImageTextBlock]) -> str | None:
    for block in blocks:
        if block.language:
            return block.language
    return None
