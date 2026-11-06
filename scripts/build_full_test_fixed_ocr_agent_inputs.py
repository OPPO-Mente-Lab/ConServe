#!/usr/bin/env python3
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Build fixed-OCR, end-to-end Agent tasks for full_test_merged outputs."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fixed_source(path: Path) -> OrderedDict[int, str]:
    grouped: OrderedDict[int, list[str]] = OrderedDict()
    for index, item in enumerate(((load(path).get("data") or {}).get("items") or [])):
        number = int(item.get("paragraphNo", index))
        text = str(item.get("text") or item.get("text_suggest") or "").strip()
        if text:
            grouped.setdefault(number, []).append(text)
    return OrderedDict((number, " ".join(parts)) for number, parts in grouped.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--template-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    tasks: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    missing_ocr: list[str] = []
    for tag in args.tags:
        for output_path in sorted((Path(args.result_root) / tag).glob("*__vlm.json")):
            payload = load(output_path)
            raw_stem = str(payload.get("stem") or output_path.stem.removesuffix("__vlm"))
            ocr_path = Path(args.data_root) / tag / "ocr_json" / f"{raw_stem}_ocr.json"
            if not ocr_path.exists():
                missing_ocr.append(str(ocr_path))
                continue
            predictions = {
                int(paragraph.get("paragraph_no", index)): str(paragraph.get("translated_text") or "").strip()
                for index, paragraph in enumerate(payload.get("paragraphs") or [])
            }
            task_id = f"{tag}__{raw_stem}"
            pairs = [{
                "pair_id": f"{task_id}:p{number}",
                "source_text": source_text,
                # Empty output is intentionally retained: it measures end-to-end coverage.
                "translated_text": predictions.get(number, ""),
                "metadata": {"subset": tag, "image_stem": raw_stem, "paragraph_no": number, "source_kind": "fixed_ocr"},
            } for number, source_text in fixed_source(ocr_path).items()]
            if not pairs:
                continue
            tasks.append({"task_id": task_id, "target_language": "zh-CN", "text_pairs": pairs,
                          "metadata": {"subset": tag, "image_stem": raw_stem, "source_kind": "fixed_ocr"}})
            images.append({"task_id": task_id, "source_path": str(output_path), "ocr_path": str(ocr_path), "subset": tag})

    template = Path(args.template_out)
    summary = Path(args.summary_out)
    template.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary.write_text(json.dumps({"images": images, "missing_ocr": missing_ocr}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "missing_ocr": len(missing_ocr), "tags": args.tags}, ensure_ascii=False))


if __name__ == "__main__":
    main()
