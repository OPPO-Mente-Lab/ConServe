# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from .text_critic_pipeline import batch_result_envelope, evaluate_text_with_critic, pairs_from_request

Evaluator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        tasks = payload
    elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        tasks = payload["tasks"]
    else:
        raise ValueError("manifest must be a JSON array or an object with a tasks array")

    validated: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"task at index {index} must be a JSON object")
        if not task.get("task_id"):
            raise ValueError(f"task at index {index} is missing task_id")
        pairs_from_request(task)
        validated.append(task)
    return validated


def run_batch(
    tasks: Iterable[dict[str, Any]],
    config: dict[str, Any],
    output_jsonl: Path,
    output_json: Path,
    *,
    max_workers: int = 1,
    evaluator: Evaluator = evaluate_text_with_critic,
) -> list[dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    task_list = list(tasks)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    indexed_results: dict[int, dict[str, Any]] = {}
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_task, task, config, evaluator): (index, task)
                for index, task in enumerate(task_list)
            }
            for future in as_completed(futures):
                index, task = futures[future]
                result = future.result()
                indexed_results[index] = result
                stream.write(
                    json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                stream.flush()
                scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
                print(
                    f"[{result.get('task_id')}] status={result.get('status')} "
                    f"text={scores.get('text_score_5')}",
                    flush=True,
                )

    results = [indexed_results[index] for index in range(len(task_list))]
    _write_json_atomic(output_json, results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch text-to-text evaluation with Faithfulness, Fluency, and Text Critic. "
            "Each task must provide text_pairs or source_text/translated_text."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--disable-skills",
        action="store_true",
        help="Skip Accuracy/Fluency detail skills after the base agents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and configuration without calling models.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = load_tasks(args.manifest)
        config = _load_config(args.runtime_config)
        if args.disable_skills:
            config["enable_text_skills"] = False
        if args.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if args.dry_run:
            print(
                f"validated {len(tasks)} task(s); "
                f"max_workers={args.max_workers}"
            )
            return 0
        run_batch(
            tasks,
            config,
            args.output_jsonl,
            args.output_json,
            max_workers=args.max_workers,
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


def _evaluate_task(
    task: dict[str, Any],
    config: dict[str, Any],
    evaluator: Evaluator,
) -> dict[str, Any]:
    try:
        pipeline_result = evaluator(task, config)
        return batch_result_envelope(task, pipeline_result)
    except Exception as error:
        return {
            "task_id": task.get("task_id"),
            "status": "failed",
            "scores": {"text_score_5": None},
            "text_result": None,
            "failures": [
                {
                    "branch": "text",
                    "stage": "task_execution",
                    "error_type": type(error).__name__,
                    "message": "task evaluation failed",
                }
            ],
        }


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"runtime config does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("runtime config must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
