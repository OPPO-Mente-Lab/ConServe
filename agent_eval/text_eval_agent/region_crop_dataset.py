# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DIRECTIONS = {
    "en": ("en", "zh", "en_to_zh"),
    "ja": ("ja", "zh", "ja_to_zh"),
    "ko": ("ko", "zh", "ko_to_zh"),
    "zh": ("zh", "en", "zh_to_en"),
}
_RESULT_DIR_NAME = "translation_results"
_LEGACY_RESULT_DIR_NAME = "接口返回JSON"


def load_region_crop_tasks(
    json_root: Path,
    image_root: Path,
    *,
    input_issues: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build image-grounded text tasks without trusting API sourceText."""
    root = Path(json_root)
    source_dir = Path(image_root)
    tasks: list[dict[str, Any]] = []
    for group, (source_language, target_language, direction) in _DIRECTIONS.items():
        group_dir = root / group
        result_dir = group_dir / _RESULT_DIR_NAME
        if not result_dir.is_dir():
            result_dir = group_dir / _LEGACY_RESULT_DIR_NAME
        if not result_dir.is_dir():
            continue
        for json_path in sorted(result_dir.glob("*.json"), key=lambda path: path.name):
            payload = _read_payload(json_path)
            if payload.get("code") != 0:
                if input_issues is not None:
                    input_issues.append(
                        {
                            "result_file": str(json_path),
                            "type": "upstream_result_failed",
                            "code": payload.get("code"),
                            "message": str(payload.get("message") or ""),
                        }
                    )
                continue
            data = payload.get("data")
            if not isinstance(data, dict) or data.get("to") != target_language:
                raise ValueError(f"unexpected target language: {json_path}")
            translations, api_paragraph_nos = _read_translations(
                data.get("translate"), json_path
            )
            image_stem = _strip_language_suffix(json_path.stem, group)
            image_path = source_dir / f"{image_stem}.jpg"
            if not image_path.is_file():
                raise ValueError(f"missing source image: {image_path}")
            tasks.append(
                {
                    "task_id": f"{direction}:{json_path.stem}",
                    "region_image": str(image_path),
                    "translations": translations,
                    "source_language": source_language,
                    "target_language": target_language,
                    "metadata": {
                        "direction": direction,
                        "result_file": str(json_path),
                        "image_id": json_path.stem,
                        "api_paragraph_nos": api_paragraph_nos,
                    },
                }
            )
    if not tasks:
        raise ValueError(f"no region translation results found under: {root}")
    return tasks


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid result JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid result JSON object: {path}")
    return value


def _read_translations(
    value: Any, path: Path
) -> tuple[list[dict[str, Any]], list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"result has no translations: {path}")
    translations: list[dict[str, Any]] = []
    api_paragraph_nos: list[int] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"invalid translation row: {path}")
        paragraph_no = row.get("paragraphNo")
        translated_text = row.get("translateText")
        if (
            not isinstance(paragraph_no, int)
            or isinstance(paragraph_no, bool)
            or paragraph_no < 0
        ):
            raise ValueError(f"invalid paragraphNo: {path}")
        if not isinstance(translated_text, str):
            raise ValueError(f"invalid translateText: {path}")
        api_paragraph_nos.append(paragraph_no)
        translations.append({"pid": index, "translated_text": translated_text})
    return translations, api_paragraph_nos


def _strip_language_suffix(stem: str, group: str) -> str:
    suffix = f"_{group}"
    if not stem.endswith(suffix):
        raise ValueError(f"result filename must end with {suffix}: {stem}")
    return stem[: -len(suffix)]
