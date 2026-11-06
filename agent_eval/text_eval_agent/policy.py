# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


POLICY_PATH = Path(__file__).resolve().parent / "policies" / "text_evaluation_policy.yaml"


@lru_cache(maxsize=1)
def load_text_evaluation_policy() -> dict[str, Any]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "version",
        "light_naturalness",
        "small_semantic_shift",
        "counting_unit",
        "score_formatting",
        "score_metadata",
        "non_scored_error_types",
    }
    missing = sorted(required.difference(policy))
    if missing:
        raise ValueError(f"Text evaluation policy is missing keys: {', '.join(missing)}")
    if not isinstance(policy["non_scored_error_types"], list):
        raise ValueError("non_scored_error_types must be a list")
    return policy


POLICY_VERSION = str(load_text_evaluation_policy()["version"])


def render_policy_prompt() -> str:
    policy = load_text_evaluation_policy()
    ignored = ", ".join(policy["non_scored_error_types"])
    return (
        f"Evaluation Policy: {policy['version']}\n"
        f"- Light naturalness defects: {policy['light_naturalness']}.\n"
        f"- Small semantic shifts: {policy['small_semantic_shift']}.\n"
        f"- Counting unit: {policy['counting_unit']}; count distinct affected fragments, not repair operations.\n"
        f"- Score formatting: {str(policy['score_formatting']).lower()}.\n"
        f"- Score metadata: {str(policy['score_metadata']).lower()}.\n"
        f"- Non-scored error types: {ignored}. Do not return these as scored errors."
    )


def is_non_scored_error_type(error_type: object) -> bool:
    value = str(error_type or "").strip().casefold()
    ignored = {str(item).casefold() for item in load_text_evaluation_policy()["non_scored_error_types"]}
    return value in ignored


def locate_fragment(text: str, fragment: object) -> dict[str, int] | None:
    value = str(fragment or "")
    if not value:
        return None
    start = text.find(value)
    if start < 0:
        return None
    return {"start": start, "end": start + len(value)}


def affected_fragment_key(
    source_span: dict[str, int] | None,
    target_span: dict[str, int] | None,
) -> str | None:
    parts: list[str] = []
    if source_span:
        parts.append(f"source:{source_span['start']}:{source_span['end']}")
    if target_span:
        parts.append(f"target:{target_span['start']}:{target_span['end']}")
    return "|".join(parts) or None


def annotate_fragment_dict(error: dict[str, Any], source_text: str, target_text: str) -> dict[str, Any]:
    annotated = dict(error)
    source_span = locate_fragment(source_text, annotated.get("source_fragment"))
    target_span = locate_fragment(target_text, annotated.get("target_fragment"))
    annotated["source_fragment_span"] = source_span
    annotated["target_fragment_span"] = target_span
    annotated["affected_fragment_key"] = affected_fragment_key(source_span, target_span)
    return annotated
