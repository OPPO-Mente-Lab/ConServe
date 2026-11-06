# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import (
    evaluate_background,
    evaluate_layout,
    evaluate_text_rendering,
    evaluate_visual,
    normalize_dimensions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate translated-text rendering from one original/translated image pair."
    )
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--translated-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--source-language")
    parser.add_argument("--target-language")
    parser.add_argument("--output-language", default="zh-CN")
    parser.add_argument(
        "--base-url",
        default="https://api.example.invalid/v1",
    )
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--critic-model",
        help="Optional model override for the Visual Critic.",
    )
    parser.add_argument("--ak-env", default="MODEL_SERVICE_AK")
    parser.add_argument("--sk-env", default="MODEL_SERVICE_SK")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--detail-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--ocr-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--minor-trigger-count", type=int, default=3)
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Disable Visual Critic for combined visual evaluation.",
    )
    parser.add_argument(
        "--dimensions",
        default="text_rendering",
        help="Comma-separated: text_rendering, layout, background, or any combination",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.minor_trigger_count < 1:
        parser.error("--minor-trigger-count must be at least 1")
    for name, value in (
        ("--detail-confidence-threshold", args.detail_confidence_threshold),
        ("--ocr-confidence-threshold", args.ocr_confidence_threshold),
        ("--temperature", args.temperature),
    ):
        if value < 0 or value > 1:
            parser.error(f"{name} must be between 0 and 1")

    request = {
        "task_id": args.task_id or args.source_image.stem,
        "source_image": str(args.source_image),
        "translated_image": str(args.translated_image),
        "source_language": args.source_language,
        "target_language": args.target_language,
        "output_language": args.output_language,
        "runtime": {
            "provider": "signed_api",
            "base_url": args.base_url,
            "ak_env": args.ak_env,
            "sk_env": args.sk_env,
            "default_model": args.model,
            "agent_models": (
                {"visual_critic": args.critic_model} if args.critic_model else {}
            ),
            "timeout_seconds": args.timeout,
            "temperature": args.temperature,
        },
        "config": {
            "detail_skill_confidence_threshold": args.detail_confidence_threshold,
            "ocr_detail_confidence_threshold": args.ocr_confidence_threshold,
            "minor_skill_trigger_count": args.minor_trigger_count,
            "enable_visual_critic": not args.no_critic,
        },
    }
    try:
        dimensions = normalize_dimensions(
            tuple(part.strip() for part in args.dimensions.split(",") if part.strip())
        )
        if dimensions == ("text_rendering",):
            result = evaluate_text_rendering(request)
        elif dimensions == ("layout",):
            result = evaluate_layout(request)
        elif dimensions == ("background",):
            result = evaluate_background(request)
        else:
            result = evaluate_visual(request, dimensions=dimensions)
        _write_json_atomic(args.output, result)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
