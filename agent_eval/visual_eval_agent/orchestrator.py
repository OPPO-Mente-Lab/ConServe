# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .agents.background import BackgroundAgent
from .agents.layout import LayoutAgent
from .agents.text_rendering import TextRenderingAgent
from .agents.visual_critic import VisualCritic, normalize_visual_critic_response
from .config import VisualEvaluationConfig
from .consistency.background_controller import (
    calibrate_background_errors,
    consolidate_background_errors,
)
from .consistency.layout_controller import (
    consolidate_layout_errors,
    filter_unstable_direction_errors,
)
from .consistency.coverage_guard import (
    CoverageAudit,
    audit_empty_ledger,
    finalize_coverage,
)
from .llm_client import SignedVisionClient, VisionModelClient
from .models import ImagePair
from .runtime import VisionRuntimeConfig
from .scoring import score_visual_errors
from .skills.background_detail import BackgroundDetailSkill
from .skills.layout_detail import LayoutDetailSkill
from .skills.text_rendering_detail import TextRenderingDetailSkill
from .trigger import (
    plan_background_skill_call,
    plan_layout_skill_call,
    plan_text_rendering_skill_call,
)


VISUAL_DIMENSIONS = ("text_rendering", "layout", "background")
DEFAULT_VISUAL_DIMENSIONS = ("text_rendering", "layout")
INTERNAL_CANVAS_LAYOUT_RISK_TYPES = {
    "font_size_inconsistency",
    "size_mismatch",
}


def evaluate_text_rendering(
    request: dict,
    model_client: VisionModelClient | None = None,
) -> dict:
    image_pair = ImagePair.from_request(request)
    runtime = VisionRuntimeConfig.from_dict(request.get("runtime"))
    config = VisualEvaluationConfig.from_dict(request.get("config"))
    client = model_client or SignedVisionClient()

    candidate_errors, final_errors, triggered_skills, coverage_audit = _run_text_rendering_branch(
        image_pair, client, runtime, config
    )
    return _result(
        image_pair,
        runtime,
        candidate_errors,
        final_errors,
        triggered_skills,
        coverage_audit,
    )


def evaluate_layout(
    request: dict,
    model_client: VisionModelClient | None = None,
) -> dict:
    image_pair = ImagePair.from_request(request)
    runtime = VisionRuntimeConfig.from_dict(request.get("runtime"))
    config = VisualEvaluationConfig.from_dict(request.get("config"))
    client = model_client or SignedVisionClient()

    candidate_errors, final_errors, triggered_skills, coverage_audit = _run_layout_branch(
        image_pair, client, runtime, config
    )
    return _result(
        image_pair,
        runtime,
        candidate_errors,
        final_errors,
        triggered_skills,
        coverage_audit,
    )


def evaluate_background(
    request: dict,
    model_client: VisionModelClient | None = None,
) -> dict:
    image_pair = ImagePair.from_request(request)
    runtime = VisionRuntimeConfig.from_dict(request.get("runtime"))
    config = VisualEvaluationConfig.from_dict(request.get("config"))
    client = model_client or SignedVisionClient()

    candidate_errors, final_errors, triggered_skills, detail_risk_regions, coverage_audit = (
        _run_background_branch(image_pair, client, runtime, config)
    )
    result = _result(
        image_pair,
        runtime,
        candidate_errors,
        final_errors,
        triggered_skills,
        coverage_audit,
    )
    result["detail_risk_regions"] = detail_risk_regions
    return result


def evaluate_visual(
    request: dict,
    model_client: VisionModelClient | None = None,
    dimensions: tuple[str, ...] = DEFAULT_VISUAL_DIMENSIONS,
    use_critic: bool | None = None,
) -> dict:
    selected_dimensions = normalize_dimensions(dimensions)
    image_pair = ImagePair.from_request(request)
    runtime = VisionRuntimeConfig.from_dict(request.get("runtime"))
    config = VisualEvaluationConfig.from_dict(request.get("config"))
    client = model_client or SignedVisionClient()

    candidate_errors = []
    final_errors = []
    triggered_skills: list[str] = []
    detail_risk_regions: list[dict] = []
    coverage_audits: list[CoverageAudit] = []
    internal_canvas_layout_risk = False
    for dimension in selected_dimensions:
        if dimension == "text_rendering":
            branch = _run_text_rendering_branch(image_pair, client, runtime, config)
        elif dimension == "layout":
            branch = _run_layout_branch(
                image_pair,
                client,
                runtime,
                config,
                review_internal_canvas=internal_canvas_layout_risk,
            )
        else:
            background_branch = _run_background_branch(
                image_pair, client, runtime, config
            )
            branch = background_branch[:3]
            detail_risk_regions.extend(background_branch[3])
            branch_audit = background_branch[4]
        if dimension != "background":
            branch_candidates, branch_final, branch_skills, branch_audit = branch
        else:
            branch_candidates, branch_final, branch_skills = branch
        candidate_errors.extend(branch_candidates)
        final_errors.extend(branch_final)
        triggered_skills.extend(branch_skills)
        coverage_audits.append(branch_audit)
        if dimension == "text_rendering":
            internal_canvas_layout_risk = any(
                error.type in INTERNAL_CANVAS_LAYOUT_RISK_TYPES
                for error in [*branch_candidates, *branch_final]
            )

    coverage_audit = _merge_coverage_audits(coverage_audits)

    result = _result(
        image_pair,
        runtime,
        candidate_errors,
        final_errors,
        list(dict.fromkeys(triggered_skills)),
        coverage_audit,
    )
    result["dimensions"] = list(selected_dimensions)
    if "background" in selected_dimensions:
        result["detail_risk_regions"] = detail_risk_regions
    critic_enabled = config.enable_visual_critic if use_critic is None else use_critic
    critic_review, adjudicated_errors = _run_visual_critic(
        image_pair,
        final_errors,
        client,
        runtime,
        enabled=critic_enabled,
    )
    adjudicated_errors, review_required = finalize_coverage(
        adjudicated_errors,
        coverage_audit,
        suppressed_errors=critic_review.get("suppressed_errors", []),
    )
    result["critic_review"] = critic_review
    result["final_errors"] = adjudicated_errors
    result["review_required"] = (
        review_required or critic_review.get("review_status") == "failed"
    )
    result["visual_score"] = score_visual_errors(adjudicated_errors)
    return result


def normalize_dimensions(dimensions: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    requested = set(dimensions)
    unknown = requested - set(VISUAL_DIMENSIONS)
    if unknown:
        raise ValueError(f"Unsupported visual dimension: {sorted(unknown)[0]}")
    if not requested:
        raise ValueError("At least one visual dimension is required")
    return tuple(dimension for dimension in VISUAL_DIMENSIONS if dimension in requested)


def _run_text_rendering_branch(
    image_pair: ImagePair,
    client: VisionModelClient,
    runtime: VisionRuntimeConfig,
    config: VisualEvaluationConfig,
):

    candidate_errors = TextRenderingAgent(client, runtime).evaluate(image_pair)
    skill_plan = plan_text_rendering_skill_call(
        candidate_errors,
        config,
        translated_text_regions=image_pair.translated_text_regions,
    )
    final_errors = candidate_errors
    rejection_records = []
    if skill_plan:
        target_ids = set(skill_plan["target_error_ids"])
        targeted = [error for error in candidate_errors if error.error_id in target_ids]
        untouched = [error for error in candidate_errors if error.error_id not in target_ids]
        refined = TextRenderingDetailSkill(client, runtime).refine(
            image_pair,
            targeted,
            target_region_ids=skill_plan["target_region_ids"],
        )
        final_errors = untouched + refined.errors
        rejection_records = refined.rejected_errors

    coverage_audit = audit_empty_ledger(
        candidate_errors, final_errors, rejection_records
    )

    return (
        candidate_errors,
        coverage_audit.critic_errors,
        [skill_plan["skill"]] if skill_plan else [],
        coverage_audit,
    )


def _run_layout_branch(
    image_pair: ImagePair,
    client: VisionModelClient,
    runtime: VisionRuntimeConfig,
    config: VisualEvaluationConfig,
    review_internal_canvas: bool = False,
):
    candidate_errors = LayoutAgent(client, runtime).evaluate(image_pair)
    skill_plan = plan_layout_skill_call(candidate_errors, config)
    if skill_plan is None and review_internal_canvas:
        skill_plan = {
            "skill": "layout_detail",
            "target_error_ids": [],
            "trigger_reasons": ["internal_content_canvas_overflow_risk"],
            "review_focus": "internal_content_canvas_overflow",
        }
    final_errors = candidate_errors
    rejection_records = []
    stability_warnings = []
    if skill_plan:
        target_ids = set(skill_plan["target_error_ids"])
        targeted = [error for error in candidate_errors if error.error_id in target_ids]
        untouched = [error for error in candidate_errors if error.error_id not in target_ids]
        refined = LayoutDetailSkill(client, runtime).refine(
            image_pair,
            targeted,
            review_focus=skill_plan.get("review_focus"),
        )
        final_errors = consolidate_layout_errors(untouched + refined.errors)
        rejection_records = refined.rejected_errors
        final_errors, stability_rejections, stability_warnings = (
            filter_unstable_direction_errors(candidate_errors, final_errors)
        )
        rejection_records = [*rejection_records, *stability_rejections]
    coverage_audit = audit_empty_ledger(
        candidate_errors, final_errors, rejection_records
    )
    if stability_warnings:
        coverage_audit = CoverageAudit(
            critic_errors=coverage_audit.critic_errors,
            fallback_candidates=coverage_audit.fallback_candidates,
            coverage_warnings=[
                *coverage_audit.coverage_warnings,
                *stability_warnings,
            ],
            rejected_errors=coverage_audit.rejected_errors,
            review_required=True,
        )
    return (
        candidate_errors,
        coverage_audit.critic_errors,
        [skill_plan["skill"]] if skill_plan else [],
        coverage_audit,
    )


def _run_background_branch(
    image_pair: ImagePair,
    client: VisionModelClient,
    runtime: VisionRuntimeConfig,
    config: VisualEvaluationConfig,
):
    assessment = BackgroundAgent(client, runtime).evaluate(image_pair)
    candidate_errors = assessment.errors
    detail_risk_regions = assessment.detail_risk_regions
    skill_plan = plan_background_skill_call(
        candidate_errors, config, detail_risk_regions
    )
    final_errors = candidate_errors
    rejection_records = []
    if skill_plan:
        target_ids = set(skill_plan["target_error_ids"])
        targeted = [error for error in candidate_errors if error.error_id in target_ids]
        untouched = [error for error in candidate_errors if error.error_id not in target_ids]
        refined = BackgroundDetailSkill(client, runtime).refine(
            image_pair,
            targeted,
            detail_risk_regions=skill_plan["detail_risk_regions"],
        )
        final_errors = consolidate_background_errors(untouched + refined.errors)
        rejection_records = refined.rejected_errors
    final_errors = calibrate_background_errors(final_errors)
    coverage_audit = audit_empty_ledger(
        candidate_errors, final_errors, rejection_records
    )
    return (
        candidate_errors,
        coverage_audit.critic_errors,
        [skill_plan["skill"]] if skill_plan else [],
        detail_risk_regions,
        coverage_audit,
    )


def _result(
    image_pair,
    runtime,
    candidate_errors,
    final_errors,
    triggered_skills,
    coverage_audit: CoverageAudit,
):
    return {
        "task_id": image_pair.task_id,
        "source_image": str(image_pair.source_image),
        "translated_image": str(image_pair.translated_image),
        "runtime": runtime.public_dict(),
        "candidate_errors": [error.to_dict() for error in candidate_errors],
        "error_ledger": [error.to_dict() for error in final_errors],
        "triggered_skills": triggered_skills,
        "skill_rejections": coverage_audit.rejected_errors,
        "coverage_warnings": coverage_audit.coverage_warnings,
        "review_required": coverage_audit.review_required,
    }


def _merge_coverage_audits(audits: list[CoverageAudit]) -> CoverageAudit:
    return CoverageAudit(
        critic_errors=[error for audit in audits for error in audit.critic_errors],
        fallback_candidates=[
            error for audit in audits for error in audit.fallback_candidates
        ],
        coverage_warnings=[
            warning for audit in audits for warning in audit.coverage_warnings
        ],
        rejected_errors=[
            rejection for audit in audits for rejection in audit.rejected_errors
        ],
        review_required=any(audit.review_required for audit in audits),
    )


def _run_visual_critic(
    image_pair: ImagePair,
    errors,
    client: VisionModelClient,
    runtime: VisionRuntimeConfig,
    enabled: bool,
):
    if not errors:
        return (
            {
                "adjudicated_errors": [],
                "suppressed_errors": [],
                "severity_changes": [],
                "notes": [],
                "review_status": "skipped_no_errors",
            },
            [],
        )

    if not enabled:
        review = normalize_visual_critic_response({}, errors)
        review["review_status"] = "disabled"
        return review, review["adjudicated_errors"]

    try:
        review = VisualCritic(client, runtime).review(image_pair, errors)
        review["review_status"] = "reviewed"
    except Exception as error:
        review = normalize_visual_critic_response({}, errors)
        review["review_status"] = "failed"
        review["error_message"] = f"{type(error).__name__}: {error}"
    return review, review["adjudicated_errors"]
