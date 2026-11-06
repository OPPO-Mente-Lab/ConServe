# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from visual_eval_agent.llm_client import VisionModelClient
from visual_eval_agent.runtime import VisionRuntimeConfig

from .agents.faithfulness import FaithfulnessAgent
from .agents.fluency import FluencyAgent
from .agents.image_text_preprocessor import ImageTextPreprocessorAgent
from .agents.text_critic import TextCritic, apply_deterministic_coverage_gate
from .config import TextEvaluationConfig
from .coverage_guard import audit_text_coverage
from .image_text_assembler import build_text_pairs
from .llm_client import ModelClient
from .models import TextDimension, TextError, TextPair
from .policy import POLICY_VERSION
from .retrying import call_with_retries
from .runtime import ModelRuntimeConfig
from .scoring import score_text_errors
from .skills.accuracy_detail import AccuracyDetailSkill
from .skills.fluency_style import FluencyStyleSkill
from .skills.text_pairing_detail import (
    PAIRING_CONFIDENCE_THRESHOLD,
    TextPairingDetailSkill,
    pairing_trigger_reasons,
)
from .trigger import plan_text_skill_calls


IMAGE_TEXT_POLICY_VERSION = "image-text-policy-v1.2"


def run_image_text_task(
    task: dict[str, Any],
    vision_runtime: VisionRuntimeConfig,
    text_runtime: ModelRuntimeConfig,
    vision_client: VisionModelClient,
    text_client: ModelClient,
    config: TextEvaluationConfig | None = None,
    retries: int = 3,
    enable_skills: bool = True,
) -> dict[str, Any]:
    started = time.time()
    result = _base_result(task)
    evaluation_config = config or TextEvaluationConfig()
    stage = "preprocess"
    extraction = None
    coverage_audit = None
    try:
        source_image = Path(str(task["source_image"]))
        translated_image = Path(str(task["translated_image"]))
        task_id = str(task["task_id"])
        extraction = call_with_retries(
            lambda: ImageTextPreprocessorAgent(vision_client, vision_runtime).extract(
                source_image, translated_image, task_id
            ),
            retries,
        )
        initial_reasons = pairing_trigger_reasons(extraction)
        pairing_triggered = bool(initial_reasons and enable_skills)
        if pairing_triggered:
            stage = "pairing_detail"
            extraction = call_with_retries(
                lambda: TextPairingDetailSkill(vision_client, vision_runtime).refine(
                    source_image,
                    translated_image,
                    task_id,
                    extraction,
                    initial_reasons,
                ),
                retries,
            )
        unresolved_reasons = pairing_trigger_reasons(extraction)
        pairs = build_text_pairs(
            task_id,
            extraction,
            source_language=task.get("source_language"),
            target_language=task.get("target_language"),
        )
        scorable_pairs = [
            pair for pair in pairs if pair.pair_confidence >= PAIRING_CONFIDENCE_THRESHOLD
        ]
        unscorable_pairs = [
            pair for pair in pairs if pair.pair_confidence < PAIRING_CONFIDENCE_THRESHOLD
        ]
        pairing_issues = _pairing_issues(extraction, unscorable_pairs, unresolved_reasons)
        coverage_audit = audit_text_coverage(
            task_id,
            extraction,
            pairs=pairs,
            source_language=task.get("source_language"),
            target_language=task.get("target_language"),
        )
        all_pairs = pairs + coverage_audit.pairs

        candidate_errors: list[TextError] = []
        if scorable_pairs:
            stage = "faithfulness"
            candidate_errors.extend(
                call_with_retries(
                    lambda: FaithfulnessAgent(text_client, text_runtime).evaluate_many(scorable_pairs),
                    retries,
                )
            )
            stage = "fluency"
            candidate_errors.extend(
                call_with_retries(
                    lambda: FluencyAgent(text_client, text_runtime).evaluate_many(scorable_pairs),
                    retries,
                )
            )
        candidate_errors, uncertain_agent_issues = _separate_pairing_uncertainty(candidate_errors)
        pairing_issues.extend(uncertain_agent_issues)

        if enable_skills and candidate_errors:
            stage = "detail_skills"
            refined_errors, skill_plan = _run_detail_skills(
                scorable_pairs,
                candidate_errors,
                evaluation_config,
                text_client,
                text_runtime,
                retries,
            )
        else:
            refined_errors = candidate_errors
            skill_plan = []

        if refined_errors:
            stage = "text_critic"
            critic_review = call_with_retries(
                lambda: TextCritic(text_client, text_runtime).review_many(
                    scorable_pairs, refined_errors
                ),
                retries,
            )
            critic_review["review_status"] = "reviewed"
        else:
            critic_review = TextCritic(text_client, text_runtime).review_many(
                scorable_pairs, []
            )
            critic_review["review_status"] = "skipped_no_candidate_errors"
        critic_review = apply_deterministic_coverage_gate(
            critic_review, coverage_audit.errors
        )
        if coverage_audit.direct_critical and not refined_errors:
            critic_review["review_status"] = "deterministic_coverage_gate"

        all_candidate_errors = candidate_errors + coverage_audit.errors
        all_refined_errors = refined_errors + coverage_audit.errors
        pair_by_id = {pair.pair_id: pair for pair in all_pairs}
        refined_by_id = {
            error.error_id: error for error in all_refined_errors
        }
        final_errors = [
            _enrich_final_error(
                error,
                pair_by_id.get(str(error.get("pair_id", ""))),
                refined_by_id,
            )
            for error in critic_review.get("adjudicated_errors", [])
            if str(error.get("pair_id", "")) in pair_by_id
        ]
        scored_errors = _to_scored_errors(final_errors, pair_by_id)

        result.update(
            {
                "extraction": extraction.to_dict(),
                "pairing_skill": {
                    "triggered": pairing_triggered,
                    "trigger_reasons": initial_reasons,
                    "remaining_reasons": unresolved_reasons,
                },
                "pairing_issues": pairing_issues,
                "coverage_audit": coverage_audit.to_dict(),
                "unit_results": _unit_results(
                    all_pairs,
                    all_candidate_errors,
                    all_refined_errors,
                    final_errors,
                ),
                "candidate_errors": [
                    error.to_dict() for error in all_candidate_errors
                ],
                "refined_errors": [error.to_dict() for error in all_refined_errors],
                "triggered_skills": [plan["skill"] for plan in skill_plan],
                "skill_plan": skill_plan,
                "critic_review": critic_review,
                "final_errors": final_errors,
                "text_score": score_text_errors(
                    scored_errors, evaluation_config, critic_review.get("caps", [])
                ),
                "failed_stage": None,
                "failed": False,
            }
        )
    except Exception as exc:
        result.update(
            {
                "extraction": extraction.to_dict() if extraction is not None else None,
                "pairing_skill": {"triggered": False, "trigger_reasons": [], "remaining_reasons": []},
                "pairing_issues": [],
                "coverage_audit": (
                    coverage_audit.to_dict() if coverage_audit is not None else None
                ),
                "unit_results": [],
                "candidate_errors": [],
                "refined_errors": [],
                "triggered_skills": [],
                "skill_plan": [],
                "critic_review": {"review_status": "failed", "adjudicated_errors": []},
                "final_errors": [],
                "text_score": None,
                "failed": True,
                "failed_stage": stage,
                "error_message": str(exc),
            }
        )
    result["latency_seconds"] = round(time.time() - started, 3)
    return result


def _base_result(task: dict[str, Any]) -> dict[str, Any]:
    return {
        **task,
        "image_text_policy_version": IMAGE_TEXT_POLICY_VERSION,
        "text_policy_version": POLICY_VERSION,
    }


def _run_detail_skills(
    pairs: list[TextPair],
    errors: list[TextError],
    config: TextEvaluationConfig,
    client: ModelClient,
    runtime: ModelRuntimeConfig,
    retries: int,
) -> tuple[list[TextError], list[dict[str, Any]]]:
    plans = plan_text_skill_calls(errors, config)
    current = list(errors)
    for plan in plans:
        target_ids = set(plan["target_error_ids"])
        target_errors = [error for error in current if error.error_id in target_ids]
        untouched = [error for error in current if error.error_id not in target_ids]
        if plan["skill"] == "accuracy_detail":
            refined = call_with_retries(
                lambda: AccuracyDetailSkill(client, runtime).refine_many(pairs, target_errors),
                retries,
            )
        elif plan["skill"] == "fluency_style":
            refined = call_with_retries(
                lambda: FluencyStyleSkill(client, runtime).refine_many(pairs, target_errors),
                retries,
            )
        else:
            refined = target_errors
        current = untouched + refined
    return current, plans


def _separate_pairing_uncertainty(
    errors: list[TextError],
) -> tuple[list[TextError], list[dict[str, Any]]]:
    scored: list[TextError] = []
    issues: list[dict[str, Any]] = []
    for error in errors:
        if error.type != "text_pair_uncertain":
            scored.append(error)
            continue
        issues.append(
            {
                "type": "text_pair_uncertain",
                "pair_id": error.pair_id,
                "reasons": [error.evidence] if error.evidence else [],
                "confidence": error.confidence,
                "scored": False,
            }
        )
    return scored, issues


def _pairing_issues(extraction, unscorable_pairs, unresolved_reasons):
    issues = [
        {
            "type": "text_pair_uncertain",
            "pair_id": pair.pair_id,
            "reasons": list(pair.metadata.get("pairing_warnings", [])) or ["low_pair_confidence"],
            "confidence": pair.pair_confidence,
            "scored": False,
        }
        for pair in unscorable_pairs
    ]
    if extraction.unmatched_source_block_ids or extraction.unmatched_target_block_ids:
        issues.append(
            {
                "type": "text_pair_uncertain",
                "pair_id": None,
                "reasons": unresolved_reasons or ["unmatched_blocks"],
                "source_block_ids": list(extraction.unmatched_source_block_ids),
                "target_block_ids": list(extraction.unmatched_target_block_ids),
                "scored": False,
            }
        )
    return issues


def _enrich_final_error(
    error: dict[str, Any],
    pair: TextPair | None,
    refined_by_id: dict[str, TextError] | None = None,
) -> dict[str, Any]:
    enriched = dict(error)
    if not enriched.get("suggestion") and refined_by_id:
        for source_id in enriched.get("merged_from", []):
            refined = refined_by_id.get(str(source_id))
            if refined is not None and refined.suggestion:
                enriched["suggestion"] = refined.suggestion
                break
    if pair is None:
        return enriched
    enriched["bbox"] = {
        "source_text_area": pair.source_bbox,
        "translated_text_area": pair.translated_bbox,
    }
    enriched["source_block_ids"] = list(pair.metadata.get("source_block_ids", []))
    enriched["target_block_ids"] = list(pair.metadata.get("target_block_ids", []))
    return enriched


def _to_scored_errors(
    final_errors: list[dict[str, Any]], pair_by_id: dict[str, TextPair]
) -> list[TextError]:
    result = []
    for error in final_errors:
        pair = pair_by_id.get(str(error.get("pair_id", "")))
        if pair is None or error.get("type") == "text_pair_uncertain":
            continue
        payload = dict(error)
        payload["dimension"] = payload.get("primary_dimension", TextDimension.FAITHFULNESS.value)
        result.append(
            TextError.from_model_dict(
                payload,
                pair,
                TextDimension(payload["dimension"]),
                "text_critic",
            )
        )
    return result


def _unit_results(
    pairs: list[TextPair],
    candidate_errors: list[TextError],
    refined_errors: list[TextError],
    final_errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair.pair_id,
            "source_text": pair.source_text,
            "translated_text": pair.translated_text,
            "source_bbox": pair.source_bbox,
            "translated_bbox": pair.translated_bbox,
            "pair_confidence": pair.pair_confidence,
            "source_block_ids": list(pair.metadata.get("source_block_ids", [])),
            "target_block_ids": list(pair.metadata.get("target_block_ids", [])),
            "unit_type": pair.metadata.get("unit_type"),
            "alignment_type": pair.metadata.get("alignment_type"),
            "candidate_errors": [error.to_dict() for error in candidate_errors if error.pair_id == pair.pair_id],
            "refined_errors": [error.to_dict() for error in refined_errors if error.pair_id == pair.pair_id],
            "final_errors": [error for error in final_errors if error.get("pair_id") == pair.pair_id],
        }
        for pair in pairs
    ]
