# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .agents.faithfulness import FaithfulnessAgent
from .agents.fluency import FluencyAgent
from .agents.text_judge import TextJudge
from .agents.text_critic import TextCritic
from .config import TextEvaluationConfig
from .llm_client import ModelClient, OpenAICompatibleClient
from .models import Severity, TextDimension, TextError, TextPair, ensure_unique_error_ids
from .runtime import ModelRuntimeConfig
from .scoring import score_text_errors
from .skills.accuracy_detail import AccuracyDetailSkill
from .skills.fluency_style import FluencyStyleSkill
from .trigger import plan_text_skill_calls


def evaluate_text_quality(request: dict, model_client: ModelClient | None = None) -> dict:
    config = _config_from_request(request)
    runtime = ModelRuntimeConfig.from_dict(request.get("runtime"))
    client = model_client or OpenAICompatibleClient()
    pairs = [TextPair.from_dict(pair) for pair in request.get("text_pairs", [])]

    faithfulness_agent = FaithfulnessAgent(client, runtime)
    fluency_agent = FluencyAgent(client, runtime)

    candidate_errors: list[TextError] = []
    for pair in pairs:
        candidate_errors.extend(faithfulness_agent.evaluate(pair))
        candidate_errors.extend(fluency_agent.evaluate(pair))

    skill_plan = plan_text_skill_calls(candidate_errors, config)
    refined_errors = _run_skills(pairs, candidate_errors, skill_plan, client, runtime)
    if request.get("context_aware_critic") is True:
        critic_review = TextCritic(client, runtime).review_many(pairs, refined_errors)
        final_errors = _critic_errors(critic_review, pairs)
        final_errors = _merge_deterministic_errors(
            final_errors,
            request.get("deterministic_errors", []),
            pairs,
        )
        caps = critic_review.get("caps", [])
        review_payload = {
            "deduplicated_errors": critic_review.get("suppressed_errors", []),
            "severity_updates": critic_review.get("severity_changes", []),
        }
    else:
        judge_result = TextJudge(client, runtime).review(refined_errors)
        final_errors = judge_result.errors
        caps = judge_result.caps
        critic_review = None
        review_payload = {
            "deduplicated_errors": judge_result.deduplicated_errors,
            "severity_updates": judge_result.severity_updates,
        }
    score = score_text_errors(final_errors, config, caps)

    return {
        "task_id": request.get("task_id"),
        "runtime": runtime.public_dict(),
        "text_score": score,
        "caps": caps,
        "error_ledger": [error.to_dict() for error in final_errors],
        "text_judge": review_payload,
        **({"text_critic": critic_review} if critic_review is not None else {}),
        "triggered_skills": [plan["skill"] for plan in skill_plan],
        "verdict": _verdict(score["text_score_100"], final_errors, bool(caps)),
    }


def _critic_errors(review: dict, pairs: list[TextPair]) -> list[TextError]:
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    errors: list[TextError] = []
    for item in review.get("adjudicated_errors", []):
        pair = pair_by_id.get(str(item.get("pair_id", "")))
        if pair is None:
            continue
        payload = dict(item)
        payload["dimension"] = payload.get(
            "primary_dimension", TextDimension.FAITHFULNESS.value
        )
        errors.append(
            TextError.from_model_dict(
                payload,
                pair,
                TextDimension(payload["dimension"]),
                "text_critic",
            )
        )
    return errors


def _merge_deterministic_errors(
    errors: list[TextError],
    raw_errors: object,
    pairs: list[TextPair],
) -> list[TextError]:
    if not isinstance(raw_errors, list):
        raise ValueError("deterministic_errors must be a list")
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    merged = list(errors)
    severity_rank = {Severity.MINOR: 0, Severity.MAJOR: 1, Severity.CRITICAL: 2}
    for item in raw_errors:
        if not isinstance(item, dict):
            raise ValueError("deterministic error entries must be objects")
        pair = pair_by_id.get(str(item.get("pair_id", "")))
        if pair is None:
            raise ValueError("deterministic error references an unknown pair")
        key = (pair.pair_id, str(item.get("type", "")))
        deterministic = TextError.from_model_dict(
            item,
            pair,
            TextDimension.FAITHFULNESS,
            "annotated_empty_target_gate",
        )
        matching = [error for error in merged if (error.pair_id, error.type) == key]
        selected = deterministic
        for existing_error in matching:
            if severity_rank[existing_error.severity] > severity_rank[selected.severity]:
                selected = existing_error
        merged = [error for error in merged if (error.pair_id, error.type) != key]
        merged.append(selected)
    return ensure_unique_error_ids(merged)


def _config_from_request(request: dict) -> TextEvaluationConfig:
    values = request.get("config", {})
    allowed = TextEvaluationConfig.__dataclass_fields__.keys()
    return TextEvaluationConfig(**{key: value for key, value in values.items() if key in allowed})


def _run_skills(
    pairs: list[TextPair],
    errors: list[TextError],
    skill_plan: list[dict],
    model_client: ModelClient,
    runtime: ModelRuntimeConfig,
) -> list[TextError]:
    current = errors
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    for plan in skill_plan:
        target_ids = set(plan["target_error_ids"])
        target_errors = [error for error in current if error.error_id in target_ids]
        untouched = [error for error in current if error.error_id not in target_ids]
        refined = target_errors
        if plan["skill"] == "accuracy_detail":
            skill = AccuracyDetailSkill(model_client, runtime)
            for pair_id in plan["target_pair_ids"]:
                refined = skill.refine(pair_by_id[pair_id], refined)
        elif plan["skill"] == "fluency_style":
            skill = FluencyStyleSkill(model_client, runtime)
            for pair_id in plan["target_pair_ids"]:
                refined = skill.refine(pair_by_id[pair_id], refined)
        current = untouched + refined
    return current


def _verdict(score_100: float, errors: list[TextError], has_cap: bool) -> str:
    if has_cap or any(error.severity.value == "Critical" for error in errors):
        return "fail" if score_100 < 50 else "needs_review"
    if any(error.severity.value == "Major" for error in errors):
        return "needs_review"
    if score_100 >= 85:
        return "pass"
    if score_100 >= 70:
        return "minor_issues"
    return "needs_review"


