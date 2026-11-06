# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JOIN_OPERATIONS = {"none", "space", "newline", "paragraph"}
ALIGNMENT_TYPES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}
VISIBILITY_LEVELS = {"clear", "partial", "uncertain"}


def _confidence(value: Any, field_name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_1000 must contain four integers")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError("bbox_1000 must contain four integers")
    ymin, xmin, ymax, xmax = value
    if any(item < 0 or item > 1000 for item in value) or ymin >= ymax or xmin >= xmax:
        raise ValueError("bbox_1000 must be ordered within 0..1000")
    return list(value)


@dataclass(frozen=True)
class ImageTextBlock:
    block_id: str
    text: str
    bbox_1000: list[int]
    language: str | None
    reading_order: int
    block_type: str
    visibility: str
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], side: str) -> "ImageTextBlock":
        block_id = str(data.get("block_id", "")).strip()
        expected_prefix = "S" if side == "source" else "T"
        if not block_id.startswith(expected_prefix):
            raise ValueError(f"{side} block_id must start with {expected_prefix}")
        text = data.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{block_id} text must be a non-empty string")
        reading_order = data.get("reading_order")
        if not isinstance(reading_order, int) or isinstance(reading_order, bool) or reading_order < 1:
            raise ValueError(f"{block_id} reading_order must be a positive integer")
        visibility = str(data.get("visibility", "uncertain"))
        if visibility not in VISIBILITY_LEVELS:
            raise ValueError(f"{block_id} visibility is invalid")
        return cls(
            block_id=block_id,
            text=text,
            bbox_1000=_bbox(data.get("bbox_1000")),
            language=str(data["language"]) if data.get("language") is not None else None,
            reading_order=reading_order,
            block_type=str(data.get("block_type", "unknown")),
            visibility=visibility,
            confidence=_confidence(data.get("confidence", 0.0), f"{block_id} confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "text": self.text,
            "bbox_1000": list(self.bbox_1000),
            "language": self.language,
            "reading_order": self.reading_order,
            "block_type": self.block_type,
            "visibility": self.visibility,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class BlockJoin:
    block_id: str
    join_after: str = "none"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockJoin":
        join_after = str(data.get("join_after", "none"))
        if join_after not in JOIN_OPERATIONS:
            raise ValueError(f"Invalid join operation: {join_after}")
        block_id = str(data.get("block_id", "")).strip()
        if not block_id:
            raise ValueError("Block reference requires block_id")
        return cls(block_id=block_id, join_after=join_after)

    def to_dict(self) -> dict[str, str]:
        return {"block_id": self.block_id, "join_after": self.join_after}


@dataclass(frozen=True)
class EvaluationUnitCandidate:
    unit_id: str
    unit_type: str
    source_sequence: list[BlockJoin]
    target_sequence: list[BlockJoin]
    alignment_type: str
    pair_confidence: float
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationUnitCandidate":
        unit_id = str(data.get("unit_id", "")).strip()
        if not unit_id:
            raise ValueError("Evaluation unit requires unit_id")
        alignment_type = str(data.get("alignment_type", ""))
        if alignment_type not in ALIGNMENT_TYPES:
            raise ValueError(f"{unit_id} alignment_type is invalid")
        source_sequence = [BlockJoin.from_dict(item) for item in data.get("source_sequence", [])]
        target_sequence = [BlockJoin.from_dict(item) for item in data.get("target_sequence", [])]
        if not source_sequence or not target_sequence:
            raise ValueError(f"{unit_id} must reference at least one source and target block")
        return cls(
            unit_id=unit_id,
            unit_type=str(data.get("unit_type", "unknown")),
            source_sequence=source_sequence,
            target_sequence=target_sequence,
            alignment_type=alignment_type,
            pair_confidence=_confidence(data.get("pair_confidence", 0.0), f"{unit_id} pair_confidence"),
            warnings=[str(item) for item in data.get("warnings", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "source_sequence": [item.to_dict() for item in self.source_sequence],
            "target_sequence": [item.to_dict() for item in self.target_sequence],
            "alignment_type": self.alignment_type,
            "pair_confidence": self.pair_confidence,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ImageTextExtraction:
    source_blocks: list[ImageTextBlock]
    target_blocks: list[ImageTextBlock]
    evaluation_units: list[EvaluationUnitCandidate]
    unmatched_source_block_ids: list[str] = field(default_factory=list)
    unmatched_target_block_ids: list[str] = field(default_factory=list)
    reading_order_uncertain: bool = False
    complex_layout: bool = False
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageTextExtraction":
        extraction = cls(
            source_blocks=[ImageTextBlock.from_dict(item, "source") for item in data.get("source_blocks", [])],
            target_blocks=[ImageTextBlock.from_dict(item, "target") for item in data.get("target_blocks", [])],
            evaluation_units=[EvaluationUnitCandidate.from_dict(item) for item in data.get("evaluation_units", [])],
            unmatched_source_block_ids=[str(item) for item in data.get("unmatched_source_block_ids", [])],
            unmatched_target_block_ids=[str(item) for item in data.get("unmatched_target_block_ids", [])],
            reading_order_uncertain=bool(data.get("reading_order_uncertain", False)),
            complex_layout=bool(data.get("complex_layout", False)),
            notes=[str(item) for item in data.get("notes", [])],
        )
        extraction.validate()
        return extraction

    def validate(self) -> None:
        source_ids = self._validate_block_side(self.source_blocks, "source")
        target_ids = self._validate_block_side(self.target_blocks, "target")
        unit_ids = [unit.unit_id for unit in self.evaluation_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Evaluation unit IDs must be unique")

        assigned_source: set[str] = set()
        assigned_target: set[str] = set()
        for unit in self.evaluation_units:
            self._validate_references(unit.source_sequence, source_ids, assigned_source, unit.unit_id)
            self._validate_references(unit.target_sequence, target_ids, assigned_target, unit.unit_id)

        self._validate_unmatched(self.unmatched_source_block_ids, source_ids, assigned_source, "source")
        self._validate_unmatched(self.unmatched_target_block_ids, target_ids, assigned_target, "target")

    @staticmethod
    def _validate_block_side(blocks: list[ImageTextBlock], side: str) -> set[str]:
        ids = [block.block_id for block in blocks]
        orders = [block.reading_order for block in blocks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{side} block IDs must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError(f"{side} reading_order values must be unique")
        return set(ids)

    @staticmethod
    def _validate_references(
        sequence: list[BlockJoin],
        known_ids: set[str],
        assigned_ids: set[str],
        unit_id: str,
    ) -> None:
        local_ids: set[str] = set()
        for reference in sequence:
            if reference.block_id not in known_ids:
                raise ValueError(f"{unit_id} references unknown block {reference.block_id}")
            if reference.block_id in local_ids or reference.block_id in assigned_ids:
                raise ValueError(f"Block {reference.block_id} is assigned more than once")
            local_ids.add(reference.block_id)
        assigned_ids.update(local_ids)

    @staticmethod
    def _validate_unmatched(
        unmatched_ids: list[str], known_ids: set[str], assigned_ids: set[str], side: str
    ) -> None:
        if len(unmatched_ids) != len(set(unmatched_ids)):
            raise ValueError(f"Unmatched {side} block IDs must be unique")
        unknown = set(unmatched_ids) - known_ids
        if unknown:
            raise ValueError(f"Unmatched {side} references unknown block {sorted(unknown)[0]}")
        overlap = set(unmatched_ids) & assigned_ids
        if overlap:
            raise ValueError(f"Block {sorted(overlap)[0]} cannot be matched and unmatched")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_blocks": [block.to_dict() for block in self.source_blocks],
            "target_blocks": [block.to_dict() for block in self.target_blocks],
            "evaluation_units": [unit.to_dict() for unit in self.evaluation_units],
            "unmatched_source_block_ids": list(self.unmatched_source_block_ids),
            "unmatched_target_block_ids": list(self.unmatched_target_block_ids),
            "reading_order_uncertain": self.reading_order_uncertain,
            "complex_layout": self.complex_layout,
            "notes": list(self.notes),
        }
