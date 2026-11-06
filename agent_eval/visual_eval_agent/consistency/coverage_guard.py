# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..models import Severity, VisualError


_SEVERITY_RANK = {
    Severity.MINOR: 0,
    Severity.MAJOR: 1,
    Severity.CRITICAL: 2,
}


@dataclass(frozen=True)
class CoverageAudit:
    critic_errors: list[VisualError]
    fallback_candidates: list[VisualError]
    coverage_warnings: list[dict[str, Any]]
    rejected_errors: list[dict[str, Any]]
    review_required: bool


def audit_empty_ledger(
    candidates: list[VisualError],
    refined: list[VisualError],
    rejected_errors: list[dict[str, Any]],
) -> CoverageAudit:
    """Retain unresolved high-severity candidates for Critic review."""

    refined_ids = {error.error_id for error in refined}
    rejection_by_id = {
        str(item.get("original_error_id")): item
        for item in rejected_errors
        if isinstance(item, dict) and item.get("original_error_id")
    }
    unresolved: list[VisualError] = []
    warnings: list[dict[str, Any]] = []

    for candidate in candidates:
        if candidate.error_id in refined_ids:
            continue
        if candidate.severity not in {Severity.MAJOR, Severity.CRITICAL}:
            continue
        rejection = rejection_by_id.get(candidate.error_id)
        if rejection is not None and _direct_contradiction(rejection):
            continue
        reason = (
            "Major/Critical candidate disappeared without an audited rejection."
            if rejection is None
            else "Major/Critical rejection lacks direct contradiction evidence at confidence >= 0.95."
        )
        unresolved.append(
            replace(
                candidate,
                source="coverage_guard",
                needs_skill=False,
                comparison={
                    **candidate.comparison,
                    "coverage_audit": {
                        "status": "unresolved_skill_rejection",
                        "reason": reason,
                    },
                },
            )
        )
        warnings.append(
            {
                "candidate_error_id": candidate.error_id,
                "dimension": candidate.dimension.value,
                "severity": candidate.severity.value,
                "reason": reason,
                "rejection": rejection,
            }
        )

    critic_errors = _unique_errors([*refined, *unresolved])
    return CoverageAudit(
        critic_errors=critic_errors,
        fallback_candidates=unresolved,
        coverage_warnings=warnings,
        rejected_errors=list(rejected_errors),
        review_required=bool(unresolved),
    )


def finalize_coverage(
    adjudicated_errors: list[dict[str, Any]],
    audit: CoverageAudit,
    suppressed_errors: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Prevent an unresolved high-severity disappearance from scoring as perfect."""

    suppressed_ids = {
        str(item.get("candidate_error_id"))
        for item in (suppressed_errors or [])
        if isinstance(item, dict) and item.get("candidate_error_id")
    }
    unresolved_fallbacks = [
        error
        for error in audit.fallback_candidates
        if error.error_id not in suppressed_ids
    ]
    if adjudicated_errors or not unresolved_fallbacks:
        return adjudicated_errors, bool(unresolved_fallbacks)
    strongest = max(
        unresolved_fallbacks,
        key=lambda error: (_SEVERITY_RANK[error.severity], error.confidence),
    )
    return [_as_final_error(strongest)], True


def _direct_contradiction(rejection: dict[str, Any]) -> bool:
    evidence = rejection.get("evidence")
    try:
        confidence = float(rejection.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    return bool(
        rejection.get("direct_contradiction") is True
        and isinstance(evidence, str)
        and evidence.strip()
        and confidence >= 0.95
    )


def _unique_errors(errors: list[VisualError]) -> list[VisualError]:
    seen: set[str] = set()
    unique: list[VisualError] = []
    for error in errors:
        if error.error_id in seen:
            continue
        seen.add(error.error_id)
        unique.append(error)
    return unique


def _as_final_error(error: VisualError) -> dict[str, Any]:
    return {
        "error_id": error.error_id,
        "primary_dimension": error.dimension.value,
        "type": error.type,
        "severity": error.severity.value,
        "bbox_1000": error.bbox_1000,
        "localization_status": error.localization_status,
        "evidence": error.evidence,
        "impact_dimensions": [],
        "impact_tags": list(error.impact_tags),
        "merged_from": [error.error_id],
        "confidence": error.confidence,
        "suggestion": error.suggestion,
    }
