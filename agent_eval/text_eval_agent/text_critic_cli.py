# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .text_critic_pipeline import evaluate_text_with_critic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source/translated text pairs with Faithfulness, Fluency, "
            "and Text Critic (no images, no Text Judge)."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Request JSON with text_pairs.")
    parser.add_argument("--output", type=Path, required=True, help="Path for result JSON.")
    parser.add_argument("--runtime-config", type=Path, help="Optional runtime JSON with text_runtime.")
    parser.add_argument(
        "--disable-skills",
        action="store_true",
        help="Skip Accuracy/Fluency detail skills after the base agents.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = _load_json(args.input)
        config = _load_config(args.runtime_config)
        if args.disable_skills:
            config["enable_text_skills"] = False
        result = evaluate_text_with_critic(request, config)
        _write_json_atomic(args.output, result)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"input does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("input must contain a JSON object")
    return value


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"runtime config does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("runtime config must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
