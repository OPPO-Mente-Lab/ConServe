# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from typing import Any

from .llm_client import ModelClient, OpenAICompatibleClient
from .orchestrator import evaluate_text_quality


TEXT_PAIR_MODEL = "gemini-3.1-pro-preview"
DEFAULT_BASE_URL = "https://api.example.invalid/v1"


class PacedModelClient:
    """Thread-safe client wrapper that spaces every real request evenly."""

    def __init__(
        self,
        delegate: ModelClient,
        *,
        requests_per_minute: float = 60,
        max_retries: int = 2,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater")
        self.delegate = delegate
        self.interval_seconds = 60.0 / float(requests_per_minute)
        self.max_retries = max_retries
        self.clock = clock
        self.sleeper = sleeper
        self._lock = threading.Lock()
        self._next_start = 0.0

    def complete_json(self, component, messages, runtime):
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            try:
                return self.delegate.complete_json(component, messages, runtime)
            except Exception:
                if attempt >= self.max_retries:
                    raise
                self.sleeper(min(2.0 ** attempt, 10.0))
        raise RuntimeError("unreachable")

    def _wait_for_slot(self) -> None:
        with self._lock:
            now = self.clock()
            request_start = max(now, self._next_start)
            self._next_start = request_start + self.interval_seconds
            wait_seconds = request_start - now
        if wait_seconds > 0:
            self.sleeper(wait_seconds)


def evaluate_text_pair(
    source_text: str,
    translated_text: str,
    *,
    source_language: str | None = None,
    target_language: str | None = None,
    task_id: str | None = None,
    pair_id: str | None = None,
    risk_level: str = "normal",
    risk_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    model_client: ModelClient | None = None,
) -> dict[str, Any]:
    source = str(source_text or "")
    translation = str(translated_text or "")
    if not source.strip():
        raise ValueError("source_text must not be empty")
    if not translation.strip():
        raise ValueError("translated_text must not be empty")

    resolved_task_id = str(task_id or f"text-pair-{uuid.uuid4().hex}")
    resolved_pair_id = str(pair_id or f"{resolved_task_id}:pair_1")
    request = {
        "task_id": resolved_task_id,
        "runtime": _locked_runtime(runtime),
        "config": dict(config or {}),
        "text_pairs": [
            {
                "pair_id": resolved_pair_id,
                "source_text": source,
                "translated_text": translation,
                "source_language": source_language,
                "target_language": target_language,
                "risk_level": risk_level,
                "risk_type": risk_type,
                "metadata": dict(metadata or {}),
            }
        ],
    }
    result = evaluate_text_quality(
        request,
        model_client=model_client or OpenAICompatibleClient(),
    )
    return {
        **result,
        "pair_id": resolved_pair_id,
        "source_text": source,
        "translated_text": translation,
        "source_language": source_language,
        "target_language": target_language,
        "metadata": dict(metadata or {}),
        "failed": False,
    }


def evaluate_text_pairs_batch(
    items: list[dict[str, Any]],
    *,
    max_workers: int = 8,
    requests_per_minute: float = 60,
    max_retries: int = 2,
    runtime: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    model_client: ModelClient | None = None,
    on_result: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be positive")

    paced_client = PacedModelClient(
        model_client or OpenAICompatibleClient(),
        requests_per_minute=requests_per_minute,
        max_retries=max_retries,
    )
    results: list[dict[str, Any] | None] = [None] * len(items)

    def run_one(index: int, item: dict[str, Any]) -> dict[str, Any]:
        source = str(item.get("source_text", item.get("src", "")) or "")
        translation = str(
            item.get("translated_text", item.get("hypo", "")) or ""
        )
        task_id = str(
            item.get("task_id")
            or item.get("identifier")
            or f"text-batch-{index + 1}"
        )
        try:
            result = evaluate_text_pair(
                source,
                translation,
                source_language=item.get("source_language", item.get("src_lang")),
                target_language=item.get("target_language", item.get("tgt_lang")),
                task_id=task_id,
                pair_id=item.get("pair_id"),
                risk_level=str(item.get("risk_level") or "normal"),
                risk_type=item.get("risk_type"),
                metadata=dict(item.get("metadata") or {}),
                runtime=runtime,
                config=config,
                model_client=paced_client,
            )
            result["input_index"] = index
            return result
        except Exception as error:
            return {
                "task_id": task_id,
                "pair_id": item.get("pair_id"),
                "source_text": source,
                "translated_text": translation,
                "source_language": item.get("source_language", item.get("src_lang")),
                "target_language": item.get("target_language", item.get("tgt_lang")),
                "metadata": dict(item.get("metadata") or {}),
                "input_index": index,
                "failed": True,
                "failed_stage": "text_evaluation",
                "error_type": type(error).__name__,
                "error_message": "text evaluation failed",
                "text_score": None,
                "error_ledger": [],
                "triggered_skills": [],
            }

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="text-pair-eval",
    ) as executor:
        future_to_index = {
            executor.submit(run_one, index, dict(item)): index
            for index, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            results[index] = result
            if on_result is not None:
                on_result(index, result)

    return [result for result in results if result is not None]


def _locked_runtime(value: dict[str, Any] | None) -> dict[str, Any]:
    runtime = {
        "provider": "signed_api",
        "base_url": DEFAULT_BASE_URL,
        "ak_env": "MODEL_SERVICE_AK",
        "sk_env": "MODEL_SERVICE_SK",
        "default_model": TEXT_PAIR_MODEL,
        "agent_models": {},
        "timeout_seconds": 240,
        "temperature": 0.0,
    }
    supplied = dict(value or {})
    supplied_model = supplied.get("default_model", supplied.get("model"))
    component_models = dict(supplied.get("agent_models") or {})
    if supplied_model is not None and supplied_model != TEXT_PAIR_MODEL:
        raise ValueError(f"text model must be {TEXT_PAIR_MODEL}")
    if any(model != TEXT_PAIR_MODEL for model in component_models.values()):
        raise ValueError(f"text model must be {TEXT_PAIR_MODEL}")
    supplied.pop("model", None)
    runtime.update(supplied)
    runtime["default_model"] = TEXT_PAIR_MODEL
    runtime["agent_models"] = component_models
    return runtime
