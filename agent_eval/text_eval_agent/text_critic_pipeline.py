# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any

from .agents.faithfulness import FaithfulnessAgent
from .agents.fluency import FluencyAgent
from .agents.text_critic import TextCritic
from .config import TextEvaluationConfig
from .image_pipeline import (
    _enrich_final_error,
    _run_detail_skills,
    _to_scored_errors,
    _unit_results,
)
from .llm_client import ModelClient, OpenAICompatibleClient
from .models import TextPair
from .policy import POLICY_VERSION
from .retrying import call_with_retries
from .runtime import ModelRuntimeConfig
from .scoring import score_text_errors

TEXT_CRITIC_POLICY_VERSION = "text-critic-policy-v1.0"


def run_text_critic_task(
    request: dict[str, Any],
    *,
    text_client: ModelClient | None = None,
    text_runtime: ModelRuntimeConfig | None = None,
    config: TextEvaluationConfig | None = None,
    retries: int = 3,
    enable_skills: bool = True,
) -> dict[str, Any]:
    started = time.time()
    evaluation_config = config or _config_from_request(request)
    runtime = text_runtime or ModelRuntimeConfig.from_dict(request.get("runtime"))
    client = text_client or OpenAICompatibleClient()
    pairs = pairs_from_request(request)
    result = _base_result(request, pairs)
    stage = "faithfulness"

    try:
        if not pairs:
            raise ValueError("request must include at least one text pair")

        stage = "faithfulness"
        candidate_errors = call_with_retries(
            lambda: FaithfulnessAgent(client, runtime).evaluate_many(pairs),
            retries,
        )
        stage = "fluency"
        candidate_errors.extend(
            call_with_retries(
                lambda: FluencyAgent(client, runtime).evaluate_many(pairs),
                retries,
            )
        )

        if enable_skills and candidate_errors:
            stage = "detail_skills"
            refined_errors, skill_plan = _run_detail_skills(
                pairs,
                candidate_errors,
                evaluation_config,
                client,
                runtime,
                retries,
            )
        else:
            refined_errors = candidate_errors
            skill_plan = []

        stage = "text_critic"
        if refined_errors:
            critic_review = call_with_retries(
                lambda: TextCritic(client, runtime).review_many(pairs, refined_errors),
                retries,
            )
            critic_review["review_status"] = "reviewed"
        else:
            critic_review = TextCritic(client, runtime).review_many(pairs, [])
            critic_review["review_status"] = "skipped_no_candidate_errors"

        pair_by_id = {pair.pair_id: pair for pair in pairs}
        refined_by_id = {error.error_id: error for error in refined_errors}
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
                "unit_results": _unit_results(
                    pairs,
                    candidate_errors,
                    refined_errors,
                    final_errors,
                ),
                "candidate_errors": [error.to_dict() for error in candidate_errors],
                "refined_errors": [error.to_dict() for error in refined_errors],
                "triggered_skills": [plan["skill"] for plan in skill_plan],
                "skill_plan": skill_plan,
                "critic_review": critic_review,
                "final_errors": final_errors,
                "text_score": score_text_errors(
                    scored_errors,
                    evaluation_config,
                    critic_review.get("caps", []),
                ),
                "failed_stage": None,
                "failed": False,
                "status": "completed",
            }
        )
    except Exception as exc:
        result.update(
            {
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
                "status": "failed",
            }
        )

    result["token_usage"] = _aggregate_token_usage(client)
    # Exact component prompts and model outputs for this image. The client
    # deliberately excludes authentication material from these records.
    result["agent_trace"] = list(getattr(client, "trace_records", []))
    result["latency_seconds"] = round(time.time() - started, 3)
    return result


def pairs_from_request(request: dict[str, Any]) -> list[TextPair]:
    raw_pairs = request.get("text_pairs")
    if raw_pairs is None:
        if "source_text" not in request or "translated_text" not in request:
            raise ValueError(
                "request must include text_pairs or both source_text and translated_text"
            )
        raw_pairs = [
            {
                "pair_id": request.get("pair_id") or f"{request.get('task_id', 'task')}:p1",
                "source_text": request["source_text"],
                "translated_text": request["translated_text"],
            }
        ]

    pairs: list[TextPair] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            raise ValueError("each text pair must be a JSON object")
        enriched = dict(item)
        enriched.setdefault("source_language", request.get("source_language"))
        enriched.setdefault("target_language", request.get("target_language"))
        pairs.append(TextPair.from_dict(enriched))
    return pairs


def evaluate_text_with_critic(
    request: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    evaluation_config = _config_from_request({**request, "config": request.get("config", config.get("config", {}))})
    runtime = ModelRuntimeConfig.from_dict(
        request.get("runtime") or config.get("text_runtime")
    )
    return run_text_critic_task(
        request,
        text_runtime=runtime,
        config=evaluation_config,
        retries=int(config.get("retries", 3)),
        enable_skills=bool(config.get("enable_text_skills", True)),
    )


def batch_result_envelope(task: dict[str, Any], pipeline_result: dict[str, Any]) -> dict[str, Any]:
    text_score = None
    text_score_payload = pipeline_result.get("text_score")
    if isinstance(text_score_payload, dict):
        text_score = text_score_payload.get("text_score_5")

    return {
        "task_id": pipeline_result.get("task_id") or task.get("task_id"),
        "status": pipeline_result.get("status", "failed"),
        "source_language": task.get("source_language"),
        "target_language": task.get("target_language"),
        "scores": {"text_score_5": text_score},
        "token_usage": pipeline_result.get("token_usage"),
        "text_result": pipeline_result,
        "failures": (
            [
                {
                    "branch": "text",
                    "stage": pipeline_result.get("failed_stage") or "task_execution",
                    "error_type": "EvaluationError",
                    "message": pipeline_result.get("error_message") or "text critic evaluation failed",
                }
            ]
            if pipeline_result.get("failed")
            else []
        ),
        "latency_seconds": pipeline_result.get("latency_seconds"),
    }


def _aggregate_token_usage(client: ModelClient) -> dict[str, Any]:
    """Summarize actual provider usage for one image evaluation.

    Some gateways omit ``usage``. In that case fields remain ``None`` rather
    than introducing tokenizer-based estimates that would not match billing.
    """
    records = getattr(client, "usage_records", [])
    if not isinstance(records, list):
        records = []
    fields = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requests": len(items),
            "usage_available_requests": 0,
        }
        for field in fields:
            values = [item[field] for item in items if isinstance(item.get(field), int)]
            result[field] = sum(values) if values else None
        result["usage_available_requests"] = sum(
            any(isinstance(item.get(field), int) for field in fields)
            for item in items
        )
        return result

    normalized = [item for item in records if isinstance(item, dict)]
    by_component: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        component = str(item.get("component") or "unknown")
        by_component.setdefault(component, []).append(item)
    summary = summarize(normalized)
    summary["by_component"] = {
        component: summarize(items) for component, items in sorted(by_component.items())
    }
    return summary


def _config_from_request(request: dict[str, Any]) -> TextEvaluationConfig:
    values = request.get("config", {})
    allowed = TextEvaluationConfig.__dataclass_fields__.keys()
    return TextEvaluationConfig(**{key: value for key, value in values.items() if key in allowed})


def _pair_dict(pair: TextPair) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "source_text": pair.source_text,
        "translated_text": pair.translated_text,
        "source_language": pair.source_language,
        "target_language": pair.target_language,
        "pair_confidence": pair.pair_confidence,
        "risk_level": pair.risk_level,
        "risk_type": pair.risk_type,
    }


def _base_result(request: dict[str, Any], pairs: list[TextPair]) -> dict[str, Any]:
    return {
        "task_id": request.get("task_id"),
        "source_language": request.get("source_language"),
        "target_language": request.get("target_language"),
        "text_pairs": [_pair_dict(pair) for pair in pairs],
        "text_critic_policy_version": TEXT_CRITIC_POLICY_VERSION,
        "text_policy_version": POLICY_VERSION,
    }
