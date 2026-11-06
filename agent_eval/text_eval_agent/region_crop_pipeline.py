# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from visual_eval_agent.llm_client import VisionModelClient
from visual_eval_agent.runtime import VisionRuntimeConfig

from .agents.region_crop_text_preprocessor import RegionCropTextPreprocessorAgent
from .config import TextEvaluationConfig
from .diagnostic_language import localize_diagnostic_fields
from .llm_client import ModelClient
from .orchestrator import evaluate_text_quality
from .policy import POLICY_VERSION
from .region_crop_coverage import build_region_coverage_pairs
from .retrying import call_with_retries
from .runtime import ModelRuntimeConfig


REGION_CROP_TEXT_POLICY_VERSION = "region-crop-text-policy-v1.0"
SOURCE_CONFIDENCE_THRESHOLD = 0.80
PAIR_CONFIDENCE_THRESHOLD = 0.80


def run_region_crop_text_task(
    task: dict[str, Any],
    vision_runtime: VisionRuntimeConfig,
    text_runtime: ModelRuntimeConfig,
    vision_client: VisionModelClient,
    text_client: ModelClient,
    config: TextEvaluationConfig | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """Evaluate external translations against source text read from one image crop."""
    _validate_task(task, retries)
    started = time.time()
    task_id = str(task["task_id"])
    result = _base_result(task_id, task)

    try:
        extraction = call_with_retries(
            lambda: RegionCropTextPreprocessorAgent(
                vision_client, vision_runtime
            ).extract(
                Path(str(task["region_image"])),
                task_id,
                task["translations"],
                source_language=task.get("source_language"),
                target_language=task.get("target_language"),
            ),
            retries,
        )
        pair_dicts, extraction_issues = _build_pairs(task_id, extraction, task)
        coverage_pairs, coverage_issues = build_region_coverage_pairs(
            task_id,
            extraction,
            task,
            [pair["source_text"] for pair in pair_dicts],
        )
        pair_dicts.extend(coverage_pairs)
        extraction_issues.extend(coverage_issues)
    except Exception as exc:
        return _failed_result(result, "region_extraction", exc, started)

    if not pair_dicts:
        result.update(
            {
                "region_extraction": extraction,
                "pairs": [],
                "extraction_issues": extraction_issues,
                "text_score": None,
                "error_ledger": [],
                "triggered_skills": [],
                "text_judge": {
                    "deduplicated_errors": [],
                    "severity_updates": [],
                },
                "text_critic": {
                    "review_status": "skipped_non_evaluable",
                    "adjudicated_errors": [],
                },
                "caps": [],
                "verdict": "needs_review",
                "failed": False,
                "failed_stage": None,
                "latency_seconds": round(time.time() - started, 3),
            }
        )
        return result

    try:
        evaluation = localize_diagnostic_fields(
            evaluate_text_quality(
                {
                    "task_id": task_id,
                    "runtime": asdict(text_runtime),
                    "config": asdict(config or TextEvaluationConfig()),
                    "text_pairs": pair_dicts,
                    "context_aware_critic": True,
                    "deterministic_errors": _empty_target_omissions(pair_dicts),
                },
                model_client=text_client,
            )
        )
    except Exception as exc:
        result.update(
            {
                "region_extraction": extraction,
                "pairs": pair_dicts,
                "extraction_issues": extraction_issues,
            }
        )
        return _failed_result(result, "text_evaluation", exc, started)

    result.update(
        {
            "region_extraction": extraction,
            "pairs": pair_dicts,
            "extraction_issues": extraction_issues,
            "text_score": evaluation["text_score"],
            "error_ledger": evaluation["error_ledger"],
            "triggered_skills": evaluation["triggered_skills"],
            "text_judge": evaluation["text_judge"],
            "text_critic": evaluation["text_critic"],
            "caps": evaluation["caps"],
            "verdict": evaluation["verdict"],
            "failed": False,
            "failed_stage": None,
            "latency_seconds": round(time.time() - started, 3),
        }
    )
    return result


def _validate_task(task: dict[str, Any], retries: int) -> None:
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    if "source_text" in task:
        raise ValueError("task-level source_text is not accepted for region evaluation")
    if "annotated_image" in task or "image_input_mode" in task:
        raise ValueError("annotated-image input mode is not supported")
    for field in ("task_id", "region_image", "translations"):
        if field not in task:
            raise ValueError(f"task.{field} is required")
    if not isinstance(task["task_id"], str) or not task["task_id"].strip():
        raise ValueError("task.task_id must be a non-empty string")
    if not isinstance(task["region_image"], (str, Path)):
        raise ValueError("task.region_image must be a path")
    if not isinstance(task["translations"], list):
        raise ValueError("task.translations must be a list")
    if not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")


def _build_pairs(
    task_id: str, extraction: dict[str, Any], task: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    image_context = dict(extraction["image_context"])
    for item in extraction["pairs"]:
        source_text = str(item["source_text"])
        if not source_text.strip():
            issues.append(
                {
                    "pid": item["pid"],
                    "type": "source_unreadable",
                    "source_visibility": item["source_visibility"],
                    "warnings": list(item["warnings"]),
                    "scored": False,
                }
            )
            continue
        issue_type = _ineligibility_reason(item)
        if issue_type:
            issues.append(
                {
                    "pid": item["pid"],
                    "type": issue_type,
                    "source_visibility": item["source_visibility"],
                    "source_confidence": item["source_confidence"],
                    "pair_confidence": item["pair_confidence"],
                    "warnings": list(item["warnings"]),
                    "scored": False,
                }
            )
            continue
        pairs.append(
            {
                "pair_id": f"{task_id}:pid_{item['pid']}",
                "source_text": source_text,
                "translated_text": item["translated_text"],
                "source_language": task.get("source_language")
                or item.get("source_language"),
                "target_language": task.get("target_language"),
                "source_bbox": list(item["source_bbox_1000"]),
                "translated_bbox": None,
                "pair_confidence": item["pair_confidence"],
                "source_ocr_confidence": item["source_confidence"],
                "metadata": {"pid": item["pid"], "image_context": image_context},
            }
        )
    for pid in extraction.get("missing_pids", []):
        issues.append({"pid": pid, "type": "source_region_missing", "scored": False})
    return pairs, issues


def _ineligibility_reason(item: dict[str, Any]) -> str | None:
    if item["source_visibility"] != "clear":
        return f"source_{item['source_visibility']}"
    if float(item["source_confidence"]) < SOURCE_CONFIDENCE_THRESHOLD:
        return "source_insufficient_confidence"
    if float(item["pair_confidence"]) < PAIR_CONFIDENCE_THRESHOLD:
        return "pair_insufficient_confidence"
    return None


def _empty_target_omissions(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for pair in pairs:
        if str(pair["translated_text"]).strip():
            continue
        metadata = pair.get("metadata") or {}
        coverage = metadata.get("unit_type") == "coverage_omission"
        errors.append(
            {
                "error_id": f"{pair['pair_id']}:faithfulness:omission:empty_target",
                "dimension": "text.faithfulness",
                "type": "omission",
                "severity": metadata.get("coverage_severity", "Major"),
                "pair_id": pair["pair_id"],
                "evidence": (
                    "原图中存在清晰且需要翻译的文字，但候选译文中没有对应内容，构成漏译。"
                    if coverage
                    else "源文清晰可读，但候选译文为空，构成整段漏译。"
                ),
                "source_fragment": pair["source_text"],
                "target_fragment": "",
                "suggestion": "补充与缺失源文完整对应的译文。",
                "confidence": 1.0,
                "source": metadata.get(
                    "coverage_source", "region_crop_empty_target_gate"
                ),
                "needs_skill": False,
            }
        )
    return errors


def _base_result(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "region_image": str(task["region_image"]),
        "source_language": task.get("source_language"),
        "target_language": task.get("target_language"),
        "region_crop_text_policy_version": REGION_CROP_TEXT_POLICY_VERSION,
        "text_policy_version": POLICY_VERSION,
    }


def _failed_result(
    result: dict[str, Any], stage: str, exc: Exception, started: float
) -> dict[str, Any]:
    result.update(
        {
            "failed": True,
            "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error_message": f"{stage} failed",
            "latency_seconds": round(time.time() - started, 3),
        }
    )
    return result
