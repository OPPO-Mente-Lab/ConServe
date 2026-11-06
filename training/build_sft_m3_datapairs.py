#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Filter fullscreen VLM results and build one-image multi-box mixed SFT data pairs.

Pipeline:
  1. Filter (without modifying the original VLM json / images) and write filtered_images.json
  2. Count the total number of valid boxes T; with the total sample size ~= T, evenly split
     the target counts across K=1..N (default N=5)
  3. Globally choose K dynamically according to the quota gap, randomly sample non-repeating
     box combinations from available images, and persist them

The data-pair format is aligned with ``SFT_M3_optimized.py``.

Usage (with an existing filtered_images.json):
  python build_sft_m3_datapairs.py --out-dir /path/to/output --skip-filter --clean-samples

Usage (to run filtering):
  python build_sft_m3_datapairs.py --vlm-dirs /path/to/vlm --image-dir /path/to/images \\
      --out-dir /path/to/output
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_INFERENCE_DIR = Path(__file__).resolve().parents[1] / "inference"
sys.path.insert(0, str(_INFERENCE_DIR))
from m3_pid_labels import infer_paragraph_orientation, render_pid_framed_image


def _log(msg: str) -> None:
    """Progress print with flush, avoiding long silences under pipes/redirection."""
    print(msg, flush=True)

# ---------------------------------------------------------------------------
#  Default paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_HALLUCINATION_SCRIPT = _THIS_DIR / "hallucination_filter.py"

DEFAULT_CROP_PAD = 6
DEFAULT_MAX_BOXES = 5  # maximum number of boxes N per sample
DEFAULT_EVAL_RATIO = 0.08
DEFAULT_K_RATIOS = {1: 0.30, 2: 0.30, 3: 0.30, 4: 0.05, 5: 0.05}
M123_K_RATIOS = {1: 0.10, 2: 0.20, 3: 0.70}


def parse_k_ratios(text: str) -> dict[int, float]:
    """Parse ``1=0.1,2=0.2,3=0.7`` into {k: ratio}."""
    ratios: dict[int, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"无效 k-ratios 片段: {part!r}")
        k_str, ratio_str = part.split("=", 1)
        k = int(k_str.strip())
        ratio = float(ratio_str.strip())
        if k < 1:
            raise ValueError(f"K 必须 >= 1: {k}")
        if ratio <= 0:
            raise ValueError(f"比例必须 > 0: k={k}, ratio={ratio}")
        ratios[k] = ratio
    if not ratios:
        raise ValueError("k-ratios 不能为空")
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"k-ratios 之和必须为 1.0，当前为 {total}")
    return dict(sorted(ratios.items()))


def _preferred_k(k_ratios: dict[int, float]) -> int:
    """Prefer the K with the highest ratio when the gaps are tied."""
    return max(k_ratios, key=lambda k: k_ratios[k])
DEFAULT_TRAIN_TYPE_RATIOS = {
    "mixed": 0.60,
    "all_positive": 0.30,
    "all_negative": 0.10,
}
# The eval set keeps the original balanced distribution and does not shift with the training sampling prior.
DEFAULT_EVAL_TYPE_RATIOS = {
    "mixed": 0.50,
    "all_positive": 0.30,
    "all_negative": 0.20,
}
# mixed is rounded by k*50%; affected by discrete values such as k=3, the actual share of positive boxes is about 55%.
# Combined with the 60/30/10 sample types, the simulated overall box-level positive ratio of the existing corpus is about 60%.
DEFAULT_TRAIN_MIXED_POSITIVE_RATIO = 0.50
DEFAULT_EVAL_MIXED_POSITIVE_RATIO = 0.50

# ---------------------------------------------------------------------------
#  Prompt (identical to SFT_M3_optimized.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_MULTI = """图片多区域文字翻译任务。输入是完整截图，图中有 {m} 个橙色粗框；每个框外紧邻一个红色 #PID 标签，框与标签共同标识一个独立处理单元，框外内容仅供理解语境。
按用户给出的 #PID 顺序逐框识别文字并判断是否翻译。清晰可读、非{target_lang}的自然语言必须翻译；已是{target_lang}、纯数字符号、URL或邮箱、代码标识符、单字符、商标Logo、人名或模糊残缺文字不翻译。
只输出严格单行JSON数组，长度必须为 {m}，数组元素与用户给出的 #PID 顺序一一对应；不得遗漏、合并、拆分或交换。
每个数组元素固定且只能包含以下三个字段，字段顺序固定:
需要翻译:{{"should_translate":true,"original_text":"框内原文","translated_text":"忠实完整译文"}}
不需翻译:{{"should_translate":false,"original_text":"框内可辨文字","translated_text":""}}
original_text只填写对应橙框内的可见文字，不得混入标签、相邻框或框外内容；无法辨认时为空字符串。
translated_text仅在should_translate=true时填写忠实、完整、简洁的{target_lang}译文，不得省略、扩写或夹带框外内容；false时必须为空字符串。"""


def _build_user_prompt_multi(target_lang: str, pids: list[int] | None = None) -> str:
    """Build the three-field prompt, explicitly specifying the OCR pids in the image and the output order."""
    pids = pids or []
    m = len(pids)
    labels = "、".join(f"#{pid}" for pid in pids)
    order = " → ".join(f"#{pid}" for pid in pids)
    return (
        f"目标语言：{target_lang}。本图有 {m} 个橙框，编号为 {labels}；"
        f"严格按 {order} 输出长度为 {m} 的三字段JSON数组。"
    )


# ---------------------------------------------------------------------------
#  Hallucination detection: dynamically load the reference script
# ---------------------------------------------------------------------------
def _load_hallucination_module(script_path: Path) -> Any:
    if not script_path.is_file():
        raise FileNotFoundError(f"幻觉过滤脚本不存在: {script_path}")
    spec = importlib.util.spec_from_file_location("_hallucination_filter", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
#  JSON / stem utilities
# ---------------------------------------------------------------------------
def _load_json_resilient(path: Path) -> dict[str, Any]:
    """Load JSON; fall back to raw_decode for trailing extra data (some corrupted __vlm.json files)."""
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(obj, dict):
        raise ValueError(f"顶层必须是 object: {path}")
    return obj


def _stem_of_path(path_like: Any) -> str:
    return Path(str(path_like)).stem


def _extract_stems_from_filter_json(path: Path) -> set[str]:
    """Extract stems from filter lists such as unknown / zh_zh / benchmark."""
    if not path.is_file():
        print(f"[warn] 过滤列表不存在, 跳过: {path}")
        return set()
    data = _load_json_resilient(path)
    stems: set[str] = set()

    if isinstance(data.get("items"), list):
        for item in data["items"]:
            if not isinstance(item, dict):
                continue
            for key in ("image_path", "path", "filename", "stem"):
                if item.get(key):
                    stems.add(_stem_of_path(item[key]))
                    break

    if isinstance(data.get("benchmark_image_paths"), list):
        for p in data["benchmark_image_paths"]:
            stems.add(_stem_of_path(p))

    if isinstance(data.get("annotations"), list):
        for ann in data["annotations"]:
            if not isinstance(ann, dict):
                continue
            # Only decision=yes entries in the benchmark file are official benchmark items;
            # the path list already covers them. Include annotations.stem / paths here as
            # well; taking the union with the path list is more robust.
            if ann.get("stem"):
                stems.add(str(ann["stem"]))
            for key in ("original_image_path", "resolved_image_path", "image_path"):
                if ann.get(key):
                    stems.add(_stem_of_path(ann[key]))

    # Plain list[str] / list[dict]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                stems.add(_stem_of_path(item))
            elif isinstance(item, dict):
                for key in ("image_path", "path", "filename", "stem"):
                    if item.get(key):
                        stems.add(_stem_of_path(item[key]))
                        break

    return {s for s in stems if s}


# ---------------------------------------------------------------------------
#  Image localization / box rendering / visual ordering (aligned with SFT_M3_optimized)
# ---------------------------------------------------------------------------
def resolve_image_path(
    stem: str,
    doc_image_path: str | None,
    image_dir: Path,
) -> Path | None:
    """Prefer image_dir/{stem}.*; otherwise fall back to image_path inside the VLM json."""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"):
        cand = image_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    if doc_image_path:
        p = Path(doc_image_path)
        if p.is_file():
            return p
    return None


def _padded_bbox(
    img_h: int, img_w: int, bbox: list[int], pad: int = DEFAULT_CROP_PAD,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + pad)
    return x1, y1, x2, y2


def _render_framed_full_image_multi(
    image_bgr: np.ndarray,
    pids_in_order: list[int],
    padded_by_pid: dict[int, tuple[int, int, int, int]],
    orient_by_pid: dict[int, str] | None = None,
) -> np.ndarray:
    """Full image + thick orange boxes + red ``#pid`` labels placed just outside each box."""
    return render_pid_framed_image(
        image_bgr,
        pids_in_order,
        padded_by_pid,
        orient_by_pid,
    )


def _sort_pids_by_box_position(
    pids: list[int],
    boxes: dict[int, tuple[int, int, int, int]],
) -> list[int]:
    """Sort top to bottom, and within the same row left to right."""
    rows: list[list[int]] = []
    for pid in sorted(pids, key=lambda p: (boxes[p][1], boxes[p][0], p)):
        _, y1, _, y2 = boxes[pid]
        height = max(1, y2 - y1)
        matched_row: list[int] | None = None
        for row in rows:
            for other in row:
                _, oy1, _, oy2 = boxes[other]
                other_height = max(1, oy2 - oy1)
                overlap = max(0, min(y2, oy2) - max(y1, oy1))
                if overlap >= 0.5 * min(height, other_height):
                    matched_row = row
                    break
            if matched_row is not None:
                break
        if matched_row is None:
            rows.append([pid])
        else:
            matched_row.append(pid)

    rows.sort(key=lambda row: min(boxes[p][1] for p in row))
    ordered: list[int] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda p: (boxes[p][0], boxes[p][1], p)))
    return ordered


def build_compact_assistant_item(para: dict[str, Any]) -> dict[str, Any]:
    """Build the fixed three-field SFT target."""
    should_translate = para.get("should_translate") is True
    original = para.get("original_text") or ""
    translated = para.get("translated_text") or ""
    if not isinstance(original, str):
        original = str(original)
    if not isinstance(translated, str):
        translated = str(translated)
    # If the top-level fields are empty, try backfilling from the vlm sub-object.
    if not original.strip() or (should_translate and not translated.strip()):
        vlm = para.get("vlm") or {}
        if isinstance(vlm, dict):
            if not original.strip():
                original = str(vlm.get("original_text") or "")
            if should_translate and not translated.strip():
                translated = str(vlm.get("translated_text") or "")
    # Reliable negatives usually have an empty top-level original_text; use the OCR text as the in-box source text.
    if not original.strip() and not should_translate:
        original = str(para.get("ocr_text") or "")
    if not should_translate:
        translated = ""
    return {
        "should_translate": should_translate,
        "original_text": original,
        "translated_text": translated,
    }


# ---------------------------------------------------------------------------
#  Filtering
# ---------------------------------------------------------------------------
_UNRELIABLE_NEGATIVE_REASONS = {
    "",
    "missing",
    "other",
    "parse_error",
    "vlm_parse_error",
    "unsure",
}
_PURE_DIGIT_EXTRA_CHARS = frozenset(".,:%+-_/，。％：＋－＿／")


def _normalize_skip_reason(para: dict[str, Any]) -> str:
    return str(para.get("skip_reason") or "").strip().lower()


def _is_pure_digits_or_numeric_symbols(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and any(ch.isdigit() for ch in stripped) and all(
        ch.isdigit() or ch.isspace() or ch in _PURE_DIGIT_EXTRA_CHARS
        for ch in stripped
    )


def _suspicious_positive_reason(item: dict[str, Any]) -> str | None:
    """Return a reason for which the item should definitely be removed from positives.

    Note: original_equals_translated is no longer discarded here; the filtering pipeline
    relabels it as a negative instead. Short Chinese UI text does not hit any rule when
    translated_text differs from original_text, and is kept per the final agreement.
    """
    original = str(item.get("original_text") or "").strip()
    if len(original) == 1:
        return "single_char"
    if _is_pure_digits_or_numeric_symbols(original):
        return "pure_digits"
    return None


def _should_convert_positive_to_negative(item: dict[str, Any]) -> bool:
    """A positive should be collapsed into a negative when original ~= translated (including multi-line), serving as supervision that the text is already in the target language."""
    original = str(item.get("original_text") or "").strip()
    translated = str(item.get("translated_text") or "").strip()
    return bool(original) and original == translated


def _is_valid_translate_para(para: dict[str, Any]) -> bool:
    if para.get("should_translate") is not True:
        return False
    if para.get("vlm_ok") is False:
        return False
    if para.get("prefiltered") is True:
        return False
    bbox = para.get("padded_bbox") or para.get("paragraph_bbox")
    if not bbox or len(bbox) != 4:
        return False
    item = build_compact_assistant_item(para)
    if not item["original_text"].strip() or not item["translated_text"].strip():
        return False
    return True


def _is_valid_negative_para(para: dict[str, Any]) -> bool:
    if para.get("should_translate") is not False:
        return False
    if _normalize_skip_reason(para) in _UNRELIABLE_NEGATIVE_REASONS:
        return False
    bbox = para.get("padded_bbox") or para.get("paragraph_bbox")
    return bool(bbox and len(bbox) == 4)


def _empty_filter_stats(
    unknown_stems: set[str],
    zh_zh_stems: set[str],
    benchmark_stems: set[str],
) -> dict[str, Any]:
    # zh_zh is no longer excluded at image level; only unknown + benchmark count toward the exclude list
    exclude_stems = unknown_stems | benchmark_stems
    return {
        "vlm_files_total": 0,
        "vlm_files_parse_failed": 0,
        "images_excluded_unknown": 0,
        "images_excluded_zh_zh": 0,
        "images_excluded_benchmark": 0,
        "images_excluded_duplicate_stem": 0,
        "images_excluded_no_image_file": 0,
        "images_excluded_no_valid_boxes": 0,
        "paragraphs_total_seen": 0,
        "paragraphs_seen_positive": 0,
        "paragraphs_seen_negative": 0,
        "paragraphs_dropped_invalid": 0,
        "paragraphs_dropped_unreliable_negative": 0,
        "paragraphs_dropped_negative_other": 0,
        "paragraphs_dropped_negative_parse_error": 0,
        "paragraphs_dropped_negative_unsure": 0,
        "paragraphs_dropped_suspicious_positive": 0,
        "paragraphs_dropped_positive_original_equals_translated": 0,
        "paragraphs_converted_positive_original_equals_translated_to_negative": 0,
        "paragraphs_converted_zh_zh_to_negative": 0,
        "paragraphs_dropped_positive_single_char": 0,
        "paragraphs_dropped_positive_pure_digits": 0,
        "paragraphs_dropped_hallucination": 0,
        "paragraphs_kept_positive": 0,
        "paragraphs_kept_negative": 0,
        "paragraphs_kept": 0,
        "images_kept": 0,
        "images_kept_from_zh_zh": 0,
        "boxes_per_image_hist": {},
        "per_source": {},
        "filter_list_sizes": {
            "unknown": len(unknown_stems),
            "zh_zh": len(zh_zh_stems),
            "benchmark": len(benchmark_stems),
            "exclude_union": len(exclude_stems),
        },
    }


def _merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if key in ("filter_list_sizes", "per_source"):
            continue
        if key == "boxes_per_image_hist":
            hist = dst.setdefault("boxes_per_image_hist", {})
            for hk, hv in (value or {}).items():
                hist[hk] = hist.get(hk, 0) + hv
            continue
        if isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0) + value


def filter_one_vlm_dir(
    *,
    vlm_dir: Path,
    image_dir: Path,
    unknown_stems: set[str],
    zh_zh_stems: set[str],
    benchmark_stems: set[str],
    evaluate_paragraph: Any,
    lcs_threshold: float,
    length_ratio_threshold: float,
    ocr_confidence_threshold: float,
    seen_stems: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter a single VLM directory."""
    stats = _empty_filter_stats(unknown_stems, zh_zh_stems, benchmark_stems)
    kept: list[dict[str, Any]] = []
    vlm_paths = sorted(vlm_dir.glob("*__vlm.json"))
    stats["vlm_files_total"] = len(vlm_paths)

    for vlm_path in vlm_paths:
        stem = vlm_path.name.replace("__vlm.json", "")
        if stem in unknown_stems:
            stats["images_excluded_unknown"] += 1
            continue
        # zh_zh: no longer excluded at image level; collected as negatives instead (see force_negative below)
        is_zh_zh = stem in zh_zh_stems
        if stem in benchmark_stems:
            stats["images_excluded_benchmark"] += 1
            continue
        if stem in seen_stems:
            stats["images_excluded_duplicate_stem"] += 1
            continue

        try:
            doc = _load_json_resilient(vlm_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            stats["vlm_files_parse_failed"] += 1
            print(f"[warn] 解析失败 {vlm_path.name}: {exc}")
            continue

        stem = str(doc.get("stem") or stem)
        is_zh_zh = is_zh_zh or (stem in zh_zh_stems)
        if stem in seen_stems:
            stats["images_excluded_duplicate_stem"] += 1
            continue

        image_path = resolve_image_path(stem, doc.get("image_path"), image_dir)
        if image_path is None:
            stats["images_excluded_no_image_file"] += 1
            continue

        target_lang = doc.get("target_lang") or "中文"
        # Normalize the language name: "中文" -> "简体中文" (Traditional Chinese is kept unchanged)
        if target_lang == "中文":
            target_lang = "简体中文"
        tgt_lang_code = doc.get("tgt_lang_code")

        valid_paras: list[dict[str, Any]] = []
        hallu_dropped: list[dict[str, Any]] = []

        for para in doc.get("paragraphs") or []:
            if not isinstance(para, dict):
                continue
            stats["paragraphs_total_seen"] += 1

            should_translate = para.get("should_translate")
            compact: dict[str, Any]
            skip_reason: str | None
            if should_translate is True:
                stats["paragraphs_seen_positive"] += 1
                if not _is_valid_translate_para(para):
                    stats["paragraphs_dropped_invalid"] += 1
                    continue
                compact = build_compact_assistant_item(para)
                # original ~= translated -> collect as a negative (multi-line preserved)
                if _should_convert_positive_to_negative(compact) or is_zh_zh:
                    compact = {
                        "should_translate": False,
                        "original_text": compact["original_text"],
                        "translated_text": "",
                    }
                    skip_reason = (
                        "converted_from_zh_zh"
                        if is_zh_zh
                        else "converted_from_original_equals_translated"
                    )
                    if is_zh_zh:
                        stats["paragraphs_converted_zh_zh_to_negative"] += 1
                    else:
                        stats[
                            "paragraphs_converted_positive_original_equals_translated_to_negative"
                        ] += 1
                    stats["paragraphs_kept_negative"] += 1
                    bbox = para.get("padded_bbox") or para.get("paragraph_bbox")
                    valid_paras.append(
                        {
                            "paragraph_no": int(para["paragraph_no"]),
                            "paragraph_bbox": list(para.get("paragraph_bbox") or bbox),
                            "padded_bbox": list(bbox),
                            "should_translate": False,
                            "original_text": compact["original_text"],
                            "translated_text": "",
                            "skip_reason": skip_reason,
                            "ocr_text": para.get("ocr_text"),
                            "ocr_mean_confidence": para.get("ocr_mean_confidence"),
                            "item_bboxes": para.get("item_bboxes") or [],
                            "angles": para.get("angles") or [],
                        }
                    )
                    continue

                suspicious_reason = _suspicious_positive_reason(compact)
                if suspicious_reason is not None:
                    stats["paragraphs_dropped_suspicious_positive"] += 1
                    stats[
                        f"paragraphs_dropped_positive_{suspicious_reason}"
                    ] += 1
                    continue

                should_filter, reasons, metrics = evaluate_paragraph(
                    para,
                    lcs_threshold,
                    length_ratio_threshold,
                    ocr_confidence_threshold,
                )
                if should_filter:
                    stats["paragraphs_dropped_hallucination"] += 1
                    hallu_dropped.append(
                        {
                            "paragraph_no": para.get("paragraph_no"),
                            "reasons": reasons,
                            "metrics": {
                                k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in (metrics or {}).items()
                            },
                            "ocr_text": para.get("ocr_text"),
                            "original_text": para.get("original_text"),
                            "translated_text": para.get("translated_text"),
                        }
                    )
                    continue
                skip_reason = None
                stats["paragraphs_kept_positive"] += 1
            elif should_translate is False:
                stats["paragraphs_seen_negative"] += 1
                reason = _normalize_skip_reason(para)
                if not _is_valid_negative_para(para):
                    if reason in _UNRELIABLE_NEGATIVE_REASONS:
                        stats["paragraphs_dropped_unreliable_negative"] += 1
                        if reason == "other":
                            stats["paragraphs_dropped_negative_other"] += 1
                        elif reason in {"parse_error", "vlm_parse_error"}:
                            stats["paragraphs_dropped_negative_parse_error"] += 1
                        elif reason == "unsure":
                            stats["paragraphs_dropped_negative_unsure"] += 1
                    else:
                        stats["paragraphs_dropped_invalid"] += 1
                    continue
                compact = build_compact_assistant_item(para)
                skip_reason = reason
                stats["paragraphs_kept_negative"] += 1
            else:
                stats["paragraphs_dropped_invalid"] += 1
                continue

            bbox = para.get("padded_bbox") or para.get("paragraph_bbox")
            valid_paras.append(
                {
                    "paragraph_no": int(para["paragraph_no"]),
                    "paragraph_bbox": list(para.get("paragraph_bbox") or bbox),
                    "padded_bbox": list(bbox),
                    "should_translate": compact["should_translate"],
                    "original_text": compact["original_text"],
                    "translated_text": compact["translated_text"],
                    "skip_reason": skip_reason,
                    "ocr_text": para.get("ocr_text"),
                    "ocr_mean_confidence": para.get("ocr_mean_confidence"),
                    "item_bboxes": para.get("item_bboxes") or [],
                    "angles": para.get("angles") or [],
                }
            )

        if not valid_paras:
            stats["images_excluded_no_valid_boxes"] += 1
            continue

        boxes = {
            p["paragraph_no"]: tuple(int(x) for x in p["padded_bbox"])  # type: ignore[misc]
            for p in valid_paras
        }
        ordered_pids = _sort_pids_by_box_position(list(boxes.keys()), boxes)
        para_by_pid = {p["paragraph_no"]: p for p in valid_paras}
        ordered_paras = [para_by_pid[pid] for pid in ordered_pids]

        n_boxes = len(ordered_paras)
        hist_key = str(min(n_boxes, 50))
        hist = stats["boxes_per_image_hist"]
        hist[hist_key] = hist.get(hist_key, 0) + 1

        kept.append(
            {
                "stem": stem,
                "source_dataset": vlm_dir.name,
                "target_lang": target_lang,
                "tgt_lang_code": tgt_lang_code,
                "image_path": str(image_path),
                "source_vlm_json": str(vlm_path),
                "image_size": doc.get("image_size"),
                "num_valid_boxes": n_boxes,
                "paragraphs": ordered_paras,
                "hallucination_dropped": hallu_dropped,
                "from_zh_zh": is_zh_zh,
            }
        )
        seen_stems.add(stem)
        stats["paragraphs_kept"] += n_boxes
        stats["images_kept"] += 1
        if is_zh_zh:
            stats["images_kept_from_zh_zh"] += 1

    return kept, stats


def filter_corpus(
    *,
    vlm_dirs: list[Path],
    image_dir: Path,
    unknown_json: Path | None,
    zh_zh_json: Path | None,
    benchmark_json: Path | None,
    hallucination_script: Path | None,
    lcs_threshold: float,
    length_ratio_threshold: float,
    ocr_confidence_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan __vlm.json across all VLM directories; return (kept_image_records, stats). Does not rewrite original files."""
    hallu = _load_hallucination_module(hallucination_script or DEFAULT_HALLUCINATION_SCRIPT)
    evaluate_paragraph = hallu.evaluate_paragraph

    unknown_stems = _extract_stems_from_filter_json(unknown_json) if unknown_json else set()
    zh_zh_stems = _extract_stems_from_filter_json(zh_zh_json) if zh_zh_json else set()
    benchmark_stems: set[str] = set()
    if benchmark_json is not None and benchmark_json.is_file():
        bm = _load_json_resilient(benchmark_json)
        for p in bm.get("benchmark_image_paths") or []:
            benchmark_stems.add(_stem_of_path(p))
        for ann in bm.get("annotations") or []:
            if not isinstance(ann, dict):
                continue
            if ann.get("decision") == "yes":
                if ann.get("stem"):
                    benchmark_stems.add(str(ann["stem"]))
                for key in ("original_image_path", "resolved_image_path", "image_path"):
                    if ann.get(key):
                        benchmark_stems.add(_stem_of_path(ann[key]))

    total_stats = _empty_filter_stats(unknown_stems, zh_zh_stems, benchmark_stems)
    kept_all: list[dict[str, Any]] = []
    seen_stems: set[str] = set()

    for vlm_dir in vlm_dirs:
        if not vlm_dir.is_dir():
            print(f"[warn] VLM 目录不存在, 跳过: {vlm_dir}")
            continue
        print(f"[filter] 扫描 {vlm_dir} ...")
        kept, stats = filter_one_vlm_dir(
            vlm_dir=vlm_dir,
            image_dir=image_dir,
            unknown_stems=unknown_stems,
            zh_zh_stems=zh_zh_stems,
            benchmark_stems=benchmark_stems,
            evaluate_paragraph=evaluate_paragraph,
            lcs_threshold=lcs_threshold,
            length_ratio_threshold=length_ratio_threshold,
            ocr_confidence_threshold=ocr_confidence_threshold,
            seen_stems=seen_stems,
        )
        kept_all.extend(kept)
        _merge_stats(total_stats, stats)
        total_stats["per_source"][vlm_dir.name] = {
            "vlm_files_total": stats["vlm_files_total"],
            "images_kept": stats["images_kept"],
            "paragraphs_kept": stats["paragraphs_kept"],
            "images_excluded_unknown": stats["images_excluded_unknown"],
            "images_excluded_zh_zh": stats["images_excluded_zh_zh"],
            "images_excluded_benchmark": stats["images_excluded_benchmark"],
            "images_excluded_no_valid_boxes": stats["images_excluded_no_valid_boxes"],
            "paragraphs_kept_positive": stats["paragraphs_kept_positive"],
            "paragraphs_kept_negative": stats["paragraphs_kept_negative"],
            "paragraphs_dropped_suspicious_positive": stats[
                "paragraphs_dropped_suspicious_positive"
            ],
            "paragraphs_dropped_unreliable_negative": stats[
                "paragraphs_dropped_unreliable_negative"
            ],
            "paragraphs_dropped_hallucination": stats["paragraphs_dropped_hallucination"],
        }
        print(
            f"[filter] {vlm_dir.name}: kept_images={stats['images_kept']} "
            f"kept_boxes={stats['paragraphs_kept']}"
        )

    return kept_all, total_stats


# ---------------------------------------------------------------------------
#  Box-count distribution statistics
# ---------------------------------------------------------------------------
def compute_box_count_distribution(kept: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the distribution of valid translatable box counts across all filtered images."""
    counts = [int(r.get("num_valid_boxes") or len(r.get("paragraphs") or [])) for r in kept]
    if not counts:
        return {
            "num_images": 0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0,
            "p90": 0,
            "max": 0,
            "min": 0,
            "total_valid_boxes": 0,
            "histogram": {},
        }

    arr = np.asarray(counts, dtype=np.float64)
    hist: dict[str, int] = {}
    for c in counts:
        key = str(min(int(c), 50))
        hist[key] = hist.get(key, 0) + 1

    return {
        "num_images": len(counts),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p75": int(math.ceil(float(np.percentile(arr, 75)))),
        "p90": int(math.ceil(float(np.percentile(arr, 90)))),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "total_valid_boxes": int(arr.sum()),
        "histogram_capped_at_50": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
    }


def _largest_remainder_targets(
    total: int,
    ratios: dict[Any, float],
) -> dict[Any, int]:
    """Convert ratios to integer targets whose sum strictly equals total, using the largest remainder method."""
    raw = {key: total * float(ratio) for key, ratio in ratios.items()}
    out = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(out.values())
    order = sorted(raw, key=lambda key: (raw[key] - out[key], str(key)), reverse=True)
    for key in order[:remaining]:
        out[key] += 1
    return out


def _box_count_bucket(n_boxes: int) -> str:
    if n_boxes <= 5:
        return "1-5"
    if n_boxes <= 10:
        return "6-10"
    if n_boxes <= 20:
        return "11-20"
    if n_boxes <= 50:
        return "21-50"
    return "51+"


def _positive_ratio_bucket(paragraphs: list[dict[str, Any]]) -> int:
    if not paragraphs:
        return 0
    positive = sum(p.get("should_translate") is True for p in paragraphs)
    return min(4, int(5 * positive / len(paragraphs)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source_level_split(
    images: list[dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Group by source_image_id/SHA256 and perform a stratified split, preventing the same
    original image from crossing train/eval.

    Duplicate records with the same content hash and target language keep only the one with
    the most boxes; different-language versions are retained but forced into the same split
    via a shared source_group_key.
    """
    enriched: list[dict[str, Any]] = []
    hash_cache: dict[str, str] = {}
    missing_images: list[str] = []
    for rec in images:
        image_path = Path(rec["image_path"])
        cache_key = str(image_path.resolve())
        if not image_path.is_file():
            missing_images.append(cache_key)
            continue
        image_sha256 = hash_cache.get(cache_key)
        if image_sha256 is None:
            image_sha256 = _sha256_file(image_path)
            hash_cache[cache_key] = image_sha256
        source_image_id = rec.get("source_image_id")
        group_key = (
            f"source_id:{source_image_id}"
            if source_image_id not in (None, "")
            else f"sha256:{image_sha256}"
        )
        item = dict(rec)
        item["image_sha256"] = image_sha256
        item["source_group_key"] = group_key
        enriched.append(item)

    # Keep only the copy with more supervised boxes for identical content and language to avoid duplicate training; cross-language versions stay in the same group.
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_records: list[dict[str, Any]] = []
    for rec in enriched:
        key = (rec["source_group_key"], str(rec.get("target_lang") or ""))
        old = dedup.get(key)
        if old is None or len(rec.get("paragraphs") or []) > len(old.get("paragraphs") or []):
            if old is not None:
                duplicate_records.append(old)
            dedup[key] = rec
        else:
            duplicate_records.append(rec)
    records = list(dedup.values())

    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(rec["source_group_key"], []).append(rec)

    strata: dict[tuple[Any, ...], list[str]] = {}
    for group_key, variants in groups.items():
        representative = max(variants, key=lambda r: len(r.get("paragraphs") or []))
        paragraphs = representative.get("paragraphs") or []
        langs = tuple(sorted({str(v.get("target_lang") or "") for v in variants}))
        stratum = (
            representative.get("source_dataset"),
            langs,
            _box_count_bucket(len(paragraphs)),
            _positive_ratio_bucket(paragraphs),
        )
        strata.setdefault(stratum, []).append(group_key)

    target_eval_groups = int(round(len(groups) * eval_ratio))
    per_stratum_raw = {
        stratum: len(keys) * eval_ratio for stratum, keys in strata.items()
    }
    per_stratum = {
        stratum: int(math.floor(value)) for stratum, value in per_stratum_raw.items()
    }
    remaining = target_eval_groups - sum(per_stratum.values())
    remainder_order = sorted(
        strata,
        key=lambda s: (
            per_stratum_raw[s] - per_stratum[s],
            len(strata[s]),
            str(s),
        ),
        reverse=True,
    )
    for stratum in remainder_order[:remaining]:
        per_stratum[stratum] += 1

    eval_group_keys: set[str] = set()
    for stratum, keys in strata.items():
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(),
        )
        eval_group_keys.update(ordered[: per_stratum[stratum]])

    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for rec in sorted(records, key=lambda r: (r["source_group_key"], r["stem"])):
        split = "eval" if rec["source_group_key"] in eval_group_keys else "train"
        rec["split"] = split
        (eval_ if split == "eval" else train).append(rec)
        assignments.append(
            {
                "stem": rec["stem"],
                "target_lang": rec.get("target_lang"),
                "source_group_key": rec["source_group_key"],
                "image_sha256": rec["image_sha256"],
                "split": split,
            }
        )

    stats = {
        "eval_ratio": eval_ratio,
        "num_input_records": len(images),
        "num_records_after_exact_dedup": len(records),
        "num_source_groups": len(groups),
        "num_train_source_groups": len(groups) - len(eval_group_keys),
        "num_eval_source_groups": len(eval_group_keys),
        "num_train_records": len(train),
        "num_eval_records": len(eval_),
        "num_dropped_exact_duplicate_records": len(duplicate_records),
        "missing_images": missing_images,
        "assignments": assignments,
        "dropped_exact_duplicates": [
            {
                "stem": rec.get("stem"),
                "target_lang": rec.get("target_lang"),
                "image_path": rec.get("image_path"),
            }
            for rec in duplicate_records
        ],
    }
    return train, eval_, stats


def _max_samples_for_image(n_boxes: int) -> int:
    if n_boxes <= 20:
        return 10
    if n_boxes <= 50:
        return 15
    return 20


def _sample_type_from_labels(labels: list[bool]) -> str:
    if labels and all(labels):
        return "all_positive"
    if labels and not any(labels):
        return "all_negative"
    return "mixed"


def _take_group(
    positive_pids: list[int],
    negative_pids: list[int],
    *,
    k: int,
    sample_type: str,
    rng: random.Random,
    mixed_positive_ratio: float,
) -> tuple[int, ...] | None:
    if sample_type == "all_positive":
        if len(positive_pids) < k:
            return None
        chosen = positive_pids[:k]
        del positive_pids[:k]
        return tuple(chosen)
    if sample_type == "all_negative":
        if len(negative_pids) < k:
            return None
        chosen = negative_pids[:k]
        del negative_pids[:k]
        return tuple(chosen)
    if k < 2 or not positive_pids or not negative_pids:
        return None
    n_positive = max(1, min(k - 1, int(round(k * mixed_positive_ratio))))
    n_negative = k - n_positive
    if n_positive > len(positive_pids):
        n_positive = len(positive_pids)
        n_negative = k - n_positive
    if n_negative > len(negative_pids):
        n_negative = len(negative_pids)
        n_positive = k - n_negative
    if n_positive <= 0 or n_negative <= 0 or n_positive > len(positive_pids):
        return None
    chosen = positive_pids[:n_positive] + negative_pids[:n_negative]
    del positive_pids[:n_positive]
    del negative_pids[:n_negative]
    rng.shuffle(chosen)
    return tuple(chosen)


def plan_train_disjoint_samples(
    images: list[dict[str, Any]],
    *,
    seed: int,
    target_samples: int = 0,
    k_ratios: dict[int, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Plan the training set: boxes are disjoint; K ratios and sample-type ratios are configurable."""
    k_ratios = k_ratios or DEFAULT_K_RATIOS
    preferred_k = _preferred_k(k_ratios)
    rng = random.Random(seed)
    total_boxes = sum(len(rec.get("paragraphs") or []) for rec in images)
    max_by_image = sum(
        min(_max_samples_for_image(len(rec.get("paragraphs") or [])),
            len(rec.get("paragraphs") or []))
        for rec in images
    )
    if target_samples <= 0:
        # Empirical feasible upper bound for the current ratios; the actual result is still constrained by box labels and the per-image cap.
        target_samples = min(max_by_image, int(total_boxes / 3.1))
    k_targets = _largest_remainder_targets(target_samples, k_ratios)
    type_targets = _largest_remainder_targets(
        target_samples, DEFAULT_TRAIN_TYPE_RATIOS,
    )
    k_counts: Counter[int] = Counter()
    type_counts: Counter[str] = Counter()

    states: list[dict[str, Any]] = []
    for image_index, rec in enumerate(images):
        positive = [
            int(p["paragraph_no"])
            for p in rec.get("paragraphs") or []
            if p.get("should_translate") is True
        ]
        negative = [
            int(p["paragraph_no"])
            for p in rec.get("paragraphs") or []
            if p.get("should_translate") is False
        ]
        rng.shuffle(positive)
        rng.shuffle(negative)
        states.append(
            {
                "image_index": image_index,
                "positive": positive,
                "negative": negative,
                "total_boxes": len(positive) + len(negative),
                "max_samples": _max_samples_for_image(len(positive) + len(negative)),
                "num_samples": 0,
            }
        )
    rng.shuffle(states)

    plans: list[dict[str, Any]] = []
    made_progress = True
    while len(plans) < target_samples and made_progress:
        made_progress = False
        for state in states:
            if len(plans) >= target_samples:
                break
            if state["num_samples"] >= state["max_samples"]:
                continue
            n_remaining = len(state["positive"]) + len(state["negative"])
            if n_remaining <= 0:
                continue

            candidates: list[tuple[float, int, str]] = []
            for k, k_target in k_targets.items():
                if k_counts[k] >= k_target or k > n_remaining:
                    continue
                for sample_type, type_target in type_targets.items():
                    if type_counts[sample_type] >= type_target:
                        continue
                    feasible = (
                        sample_type == "all_positive" and len(state["positive"]) >= k
                    ) or (
                        sample_type == "all_negative" and len(state["negative"]) >= k
                    ) or (
                        sample_type == "mixed"
                        and k >= 2
                        and state["positive"]
                        and state["negative"]
                        and n_remaining >= k
                    )
                    if not feasible:
                        continue
                    k_gap = (k_target - k_counts[k]) / max(1, k_target)
                    type_gap = (
                        type_target - type_counts[sample_type]
                    ) / max(1, type_target)
                    density_bonus = 0.0
                    total_in_image = int(state["total_boxes"])
                    if total_in_image <= 20 and k <= 3:
                        density_bonus = 0.08
                    elif total_in_image > 20 and k >= 4:
                        density_bonus = 0.08
                    candidates.append((k_gap + type_gap + density_bonus, k, sample_type))
            if not candidates:
                continue
            _, k, sample_type = max(
                candidates,
                key=lambda x: (x[0], -abs(x[1] - preferred_k), x[2]),
            )
            chosen = _take_group(
                state["positive"],
                state["negative"],
                k=k,
                sample_type=sample_type,
                rng=rng,
                mixed_positive_ratio=DEFAULT_TRAIN_MIXED_POSITIVE_RATIO,
            )
            if chosen is None:
                continue
            image_index = int(state["image_index"])
            rec = images[image_index]
            plans.append(
                {
                    "image_index": image_index,
                    "stem": rec["stem"],
                    "split": "train",
                    "source_group_key": rec["source_group_key"],
                    "image_sha256": rec["image_sha256"],
                    "k": k,
                    "sample_type": sample_type,
                    "sample_idx": state["num_samples"],
                    "pids": chosen,
                }
            )
            state["num_samples"] += 1
            k_counts[k] += 1
            type_counts[sample_type] += 1
            made_progress = True

    overflow: list[dict[str, Any]] = []
    for state in states:
        image_index = int(state["image_index"])
        rec = images[image_index]
        remaining_pids = list(state["positive"]) + list(state["negative"])
        if remaining_pids:
            overflow.append(
                {
                    "stem": rec["stem"],
                    "source_group_key": rec["source_group_key"],
                    "split": "train",
                    "remaining_pids": remaining_pids,
                    "num_remaining": len(remaining_pids),
                }
            )

    used_boxes = sum(int(p["k"]) for p in plans)
    selected_positive = 0
    selected_mixed_positive = 0
    selected_mixed_boxes = 0
    for plan in plans:
        para_by_pid = {
            int(para["paragraph_no"]): para
            for para in images[int(plan["image_index"])].get("paragraphs") or []
        }
        labels = [
            para_by_pid[int(pid)].get("should_translate") is True
            for pid in plan["pids"]
        ]
        selected_positive += sum(labels)
        if plan["sample_type"] == "mixed":
            selected_mixed_positive += sum(labels)
            selected_mixed_boxes += len(labels)
    stats = {
        "split": "train",
        "target_samples": target_samples,
        "num_planned_samples": len(plans),
        "total_available_boxes": total_boxes,
        "num_used_boxes": used_boxes,
        "num_overflow_boxes": total_boxes - used_boxes,
        "box_utilization": round(used_boxes / total_boxes, 6) if total_boxes else 0.0,
        "positive_boxes": selected_positive,
        "positive_box_ratio": (
            round(selected_positive / used_boxes, 6) if used_boxes else 0.0
        ),
        "mixed_positive_boxes": selected_mixed_positive,
        "mixed_total_boxes": selected_mixed_boxes,
        "mixed_positive_box_ratio": (
            round(selected_mixed_positive / selected_mixed_boxes, 6)
            if selected_mixed_boxes else 0.0
        ),
        "target_by_k": {str(k): value for k, value in k_targets.items()},
        "planned_by_k": {
            str(k): k_counts[k] for k in sorted(k_ratios)
        },
        "target_by_sample_type": type_targets,
        "planned_by_sample_type": {
            key: type_counts[key] for key in DEFAULT_TRAIN_TYPE_RATIOS
        },
        "target_mixed_positive_ratio": DEFAULT_TRAIN_MIXED_POSITIVE_RATIO,
        "k_ratios": {str(k): v for k, v in k_ratios.items()},
        "max_samples_per_image": {"le20": 10, "21to50": 15, "gt50": 20},
        "box_overlap_count": 0,
    }
    return plans, overflow, stats


def plan_eval_one_sample_per_image(
    images: list[dict[str, Any]],
    *,
    seed: int,
    k_ratios: dict[int, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The eval set takes exactly one sample per record; K and sample_type are allocated by the configured ratios."""
    k_ratios = k_ratios or DEFAULT_K_RATIOS
    preferred_k = _preferred_k(k_ratios)
    rng = random.Random(seed + 10_000)
    n_images = len(images)
    k_targets = _largest_remainder_targets(n_images, k_ratios)
    type_targets = _largest_remainder_targets(
        n_images, DEFAULT_EVAL_TYPE_RATIOS,
    )
    k_counts: Counter[int] = Counter()
    type_counts: Counter[str] = Counter()

    states: list[dict[str, Any]] = []
    for image_index, rec in enumerate(images):
        positive = [
            int(p["paragraph_no"])
            for p in rec.get("paragraphs") or []
            if p.get("should_translate") is True
        ]
        negative = [
            int(p["paragraph_no"])
            for p in rec.get("paragraphs") or []
            if p.get("should_translate") is False
        ]
        rng.shuffle(positive)
        rng.shuffle(negative)
        states.append(
            {
                "image_index": image_index,
                "positive": positive,
                "negative": negative,
                "total_boxes": len(positive) + len(negative),
            }
        )

    order = list(range(n_images))
    rng.shuffle(order)
    pending = set(order)
    plans_by_index: dict[int, dict[str, Any]] = {}

    def _feasible(state: dict[str, Any], k: int, sample_type: str) -> bool:
        n_remaining = int(state["total_boxes"])
        if k > n_remaining or k < 1:
            return False
        if sample_type == "all_positive":
            return len(state["positive"]) >= k
        if sample_type == "all_negative":
            return len(state["negative"]) >= k
        return (
            k >= 2
            and bool(state["positive"])
            and bool(state["negative"])
            and n_remaining >= k
        )

    def _assign(image_index: int, k: int, sample_type: str) -> bool:
        state = states[image_index]
        # Sample only once per image: copy the pools before _take_group to avoid polluting the candidate pools
        positive = list(state["positive"])
        negative = list(state["negative"])
        chosen = _take_group(
            positive,
            negative,
            k=k,
            sample_type=sample_type,
            rng=rng,
            mixed_positive_ratio=DEFAULT_EVAL_MIXED_POSITIVE_RATIO,
        )
        if chosen is None:
            return False
        rec = images[image_index]
        plans_by_index[image_index] = {
            "image_index": image_index,
            "stem": rec["stem"],
            "split": "eval",
            "source_group_key": rec.get("source_group_key"),
            "image_sha256": rec.get("image_sha256"),
            "k": k,
            "sample_type": sample_type,
            "sample_idx": 0,
            "pids": chosen,
        }
        k_counts[k] += 1
        type_counts[sample_type] += 1
        pending.discard(image_index)
        return True

    # Pass 1: chase the K / type gaps simultaneously
    made_progress = True
    while pending and made_progress:
        made_progress = False
        for image_index in list(pending):
            state = states[image_index]
            candidates: list[tuple[float, int, str]] = []
            for k, k_target in k_targets.items():
                if k_counts[k] >= k_target:
                    continue
                for sample_type, type_target in type_targets.items():
                    if type_counts[sample_type] >= type_target:
                        continue
                    if not _feasible(state, k, sample_type):
                        continue
                    k_gap = (k_target - k_counts[k]) / max(1, k_target)
                    type_gap = (
                        type_target - type_counts[sample_type]
                    ) / max(1, type_target)
                    candidates.append((k_gap + type_gap, k, sample_type))
            if not candidates:
                continue
            _, k, sample_type = max(
                candidates,
                key=lambda x: (x[0], -abs(x[1] - preferred_k), x[2]),
            )
            if _assign(image_index, k, sample_type):
                made_progress = True

    # Pass 2: relax the type quota and prioritize filling K
    made_progress = True
    while pending and made_progress:
        made_progress = False
        for image_index in list(pending):
            state = states[image_index]
            candidates = []
            for k, k_target in k_targets.items():
                for sample_type in DEFAULT_EVAL_TYPE_RATIOS:
                    if not _feasible(state, k, sample_type):
                        continue
                    k_gap = (k_target - k_counts[k]) / max(1, k_target)
                    type_gap = (
                        type_targets[sample_type] - type_counts[sample_type]
                    ) / max(1, type_targets[sample_type])
                    candidates.append((k_gap + 0.5 * type_gap, k, sample_type))
            if not candidates:
                continue
            _, k, sample_type = max(
                candidates,
                key=lambda x: (x[0], -abs(x[1] - preferred_k), x[2]),
            )
            if _assign(image_index, k, sample_type):
                made_progress = True

    # Pass 3: any feasible (k, type), guaranteeing one sample per image
    for image_index in list(pending):
        state = states[image_index]
        assigned = False
        for k in sorted(k_ratios, reverse=True):
            for sample_type in ("mixed", "all_positive", "all_negative"):
                if _feasible(state, k, sample_type) and _assign(
                    image_index, k, sample_type
                ):
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            raise RuntimeError(
                f"eval 无法为 stem={images[image_index].get('stem')} 分配样本 "
                f"(boxes={state['total_boxes']})"
            )

    plans = [plans_by_index[i] for i in range(n_images)]
    stats = {
        "split": "eval",
        "num_planned_samples": len(plans),
        "one_sample_per_image": True,
        "type_balanced": True,
        "target_by_k": {str(k): value for k, value in k_targets.items()},
        "planned_by_k": {
            str(k): k_counts[k] for k in sorted(k_ratios)
        },
        "target_by_sample_type": {
            key: type_targets[key] for key in DEFAULT_EVAL_TYPE_RATIOS
        },
        "planned_by_sample_type": {
            key: type_counts[key] for key in DEFAULT_EVAL_TYPE_RATIOS
        },
        "target_mixed_positive_ratio": DEFAULT_EVAL_MIXED_POSITIVE_RATIO,
        "k_ratios": {str(k): v for k, v in k_ratios.items()},
    }
    return plans, stats


def _materialize_one_sample(
    record: dict[str, Any],
    plan: dict[str, Any],
    out_dir: Path,
    *,
    image_bgr: np.ndarray,
    padded_by_pid: dict[int, tuple[int, int, int, int]],
    max_boxes_n: int,
) -> dict[str, Any] | None:
    """Draw boxes on a copy of the original image according to plan and write one SFT sample (does not modify the original image file)."""
    stem = record["stem"]
    target_lang = record["target_lang"]
    paras = record["paragraphs"]
    para_by_pid = {int(p["paragraph_no"]): p for p in paras}
    k = int(plan["k"])
    s_idx = int(plan["sample_idx"])
    # PIDs are explicit identity anchors; keep the sampling order and make the prompt, target array, and on-image labels consistent.
    ordered = [int(pid) for pid in plan["pids"] if int(pid) in padded_by_pid]
    if not ordered:
        return None

    split = str(plan.get("split") or record.get("split") or "train")
    image_index = int(plan["image_index"])
    sample_id = f"{split}__{stem}__i{image_index:05d}__k{k}_s{s_idx:04d}"
    sample_dir = out_dir / "samples" / split / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    orient_by_pid = {
        pid: infer_paragraph_orientation(para_by_pid[pid])
        for pid in ordered
    }
    framed = _render_framed_full_image_multi(
        image_bgr,
        ordered,
        padded_by_pid,
        orient_by_pid,
    )
    framed_path = sample_dir / "framed_multi.jpg"
    cv2.imwrite(str(framed_path), framed)

    target_array = [build_compact_assistant_item(para_by_pid[pid]) for pid in ordered]
    target_str = json.dumps(target_array, ensure_ascii=False, separators=(",", ":"))
    (sample_dir / "target.json").write_text(target_str, encoding="utf-8")

    system_prompt = SYSTEM_PROMPT_MULTI.format(
        target_lang=target_lang, m=len(ordered),
    )
    user_prompt = _build_user_prompt_multi(target_lang, ordered)
    meta = {
        "id": sample_id,
        "pair_type": "multi_box_disjoint_source_split",
        "split": split,
        "sample_type": plan["sample_type"],
        "stem": stem,
        "source_dataset": record.get("source_dataset"),
        "source_group_key": record.get("source_group_key"),
        "image_sha256": record.get("image_sha256"),
        "m": k,
        "k": k,
        "sample_idx": s_idx,
        "num_valid_boxes_in_image": len(paras),
        "max_boxes_n": max_boxes_n,
        "batch_pids": ordered,
        "target_lang": target_lang,
        "tgt_lang_code": record.get("tgt_lang_code"),
        "source_vlm_json": record["source_vlm_json"],
        "source_image": record["image_path"],
        "images": [str(framed_path.relative_to(out_dir))],
        "system": system_prompt,
        "user": user_prompt,
        "assistant": target_str,
        "boxes": [
            {
                "paragraph_no": pid,
                "padded_bbox": list(padded_by_pid[pid]),
                "should_translate": para_by_pid[pid]["should_translate"],
                "original_text": para_by_pid[pid]["original_text"],
                "translated_text": para_by_pid[pid]["translated_text"],
            }
            for pid in ordered
        ],
    }
    (sample_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return meta


def build_samples_from_plans(
    images: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_boxes_n: int,
    crop_pad: int = DEFAULT_CROP_PAD,
    log_every: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Persist samples according to the global plan; group by image_index to reuse image reads."""
    from collections import defaultdict

    by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for plan in plans:
        by_img[int(plan["image_index"])].append(plan)

    records: list[dict[str, Any]] = []
    k_counter: Counter[int] = Counter()
    type_counter: Counter[str] = Counter()
    positive_boxes = 0
    total_boxes = 0
    stems_by_k: dict[int, set[str]] = {k: set() for k in range(1, max_boxes_n + 1)}
    per_image: list[dict[str, Any]] = []
    done = 0
    total_plans = len(plans)
    t0 = time.perf_counter()
    _log(f"[build] 开始落盘: 共 {total_plans} 条, 涉及 {len(by_img)} 张原图")

    for img_idx in sorted(by_img.keys()):
        record = images[img_idx]
        image_path = Path(record["image_path"])
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            _log(f"[warn] 无法读图, 跳过该图全部 plan: {image_path}")
            continue
        img_h, img_w = image_bgr.shape[:2]
        padded_by_pid: dict[int, tuple[int, int, int, int]] = {}
        for para in record["paragraphs"]:
            pid = int(para["paragraph_no"])
            bbox = para.get("padded_bbox") or para.get("paragraph_bbox")
            padded_by_pid[pid] = _padded_bbox(img_h, img_w, bbox, crop_pad)

        img_k_counter: Counter[int] = Counter()
        for plan in by_img[img_idx]:
            meta = _materialize_one_sample(
                record, plan, out_dir,
                image_bgr=image_bgr,
                padded_by_pid=padded_by_pid,
                max_boxes_n=max_boxes_n,
            )
            if meta is None:
                continue
            records.append(meta)
            k = int(meta["k"])
            k_counter[k] += 1
            type_counter[str(meta["sample_type"])] += 1
            positive_boxes += sum(
                int(box["should_translate"] is True) for box in meta["boxes"]
            )
            total_boxes += len(meta["boxes"])
            stems_by_k[k].add(meta["stem"])
            img_k_counter[k] += 1
            done += 1
            if log_every > 0 and (done % log_every == 0 or done == total_plans):
                elapsed = time.perf_counter() - t0
                speed = done / elapsed if elapsed > 0 else 0.0
                eta = (total_plans - done) / speed if speed > 0 else 0.0
                pct = 100.0 * done / total_plans if total_plans else 100.0
                _log(
                    f"[build] {done}/{total_plans} ({pct:.1f}%) "
                    f"speed={speed:.1f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s "
                    f"latest={record['stem']}__k{k}"
                )

        per_image.append(
            {
                "stem": record["stem"],
                "source_dataset": record.get("source_dataset"),
                "source_group_key": record.get("source_group_key"),
                "split": record.get("split"),
                "num_valid_boxes_m": record.get("num_valid_boxes"),
                "num_samples": sum(img_k_counter.values()),
                "samples_by_k": {str(k): v for k, v in sorted(img_k_counter.items())},
            }
        )

    build_detail = {
        "actual_by_k": {str(k): k_counter.get(k, 0) for k in range(1, max_boxes_n + 1)},
        "unique_source_images_by_k": {
            str(k): len(stems_by_k[k]) for k in range(1, max_boxes_n + 1)
        },
        "num_samples": len(records),
        "actual_by_sample_type": dict(type_counter),
        "positive_boxes": positive_boxes,
        "total_boxes": total_boxes,
        "positive_box_ratio": (
            round(positive_boxes / total_boxes, 6) if total_boxes else 0.0
        ),
        "per_image": per_image,
    }
    return records, build_detail


# ---------------------------------------------------------------------------
#  IO
# ---------------------------------------------------------------------------
def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="过滤 VLM 结果并构建原图隔离、框互斥的一图多框 SFT 数据",
    )
    p.add_argument(
        "--vlm-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="一个或多个 VLM __vlm.json 目录 (过滤时必填)",
    )
    p.add_argument("--image-dir", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--unknown-json", type=Path)
    p.add_argument("--zh-zh-json", type=Path)
    p.add_argument("--benchmark-json", type=Path)
    p.add_argument(
        "--hallucination-script", type=Path,
        help="覆盖仓库内置的 training/hallucination_filter.py",
    )
    p.add_argument("--lcs-threshold", type=float, default=0.40)
    p.add_argument("--length-ratio-threshold", type=float, default=2.0)
    p.add_argument("--ocr-confidence-threshold", type=float, default=0.9)
    p.add_argument(
        "--limit-images", type=int, default=0,
        help="构建阶段最多使用多少张过滤后图片; 0=全量。默认 0",
    )
    p.add_argument(
        "--max-boxes", type=int, default=DEFAULT_MAX_BOXES,
        help=f"单条样本最大框数 N (默认 {DEFAULT_MAX_BOXES})",
    )
    p.add_argument(
        "--k-ratios",
        type=str,
        default="",
        help=(
            "训练/验证 K 分层比例，如 1=0.1,2=0.2,3=0.7；"
            f"默认 {DEFAULT_K_RATIOS}"
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="组合采样种子")
    p.add_argument(
        "--eval-ratio",
        type=float,
        default=DEFAULT_EVAL_RATIO,
        help=f"原图级验证集比例 (默认 {DEFAULT_EVAL_RATIO})",
    )
    p.add_argument(
        "--target-train-samples",
        type=int,
        default=0,
        help="训练目标条数；0=按框量自动估算",
    )
    p.add_argument("--crop-pad", type=int, default=DEFAULT_CROP_PAD)
    p.add_argument(
        "--log-every", type=int, default=200, help="落盘进度打印间隔 (条数, 默认 200)",
    )
    p.add_argument(
        "--skip-filter", action="store_true",
        help="若 out-dir/filtered_images.json 已存在则复用, 跳过过滤",
    )
    p.add_argument(
        "--filter-only", action="store_true",
        help="只做过滤, 不构建 SFT 样本",
    )
    p.add_argument(
        "--clean-samples", action="store_true",
        help="构建前清空 out-dir/samples 及旧 manifest/datapair 输出",
    )
    p.add_argument(
        "--plan-only", action="store_true",
        help="只做全局规划与统计, 不落盘图片/样本",
    )
    return p.parse_args()


def main() -> None:
    # Ensure progress is flushed promptly (even without python -u)
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = out_dir / "filtered_images.json"
    stats_path = out_dir / "stats.json"
    box_dist_path = out_dir / "box_count_distribution.json"
    plan_stats_path = out_dir / "plan_stats.json"
    split_stats_path = out_dir / "split_assignments.json"
    overflow_path = out_dir / "overflow_boxes.jsonl"

    if args.skip_filter and filtered_path.is_file():
        _log(f"[filter] 复用已有列表: {filtered_path}")
        payload = _load_json_resilient(filtered_path)
        kept = payload.get("images") or []
        filter_stats = payload.get("stats") or {}
    else:
        if not args.vlm_dirs or args.image_dir is None:
            raise SystemExit("Filtering requires --vlm-dirs and --image-dir")
        _log("[filter] 开始过滤 (不修改原始 VLM/图片) ...")
        kept, filter_stats = filter_corpus(
            vlm_dirs=list(args.vlm_dirs),
            image_dir=args.image_dir,
            unknown_json=args.unknown_json,
            zh_zh_json=args.zh_zh_json,
            benchmark_json=args.benchmark_json,
            hallucination_script=args.hallucination_script,
            lcs_threshold=args.lcs_threshold,
            length_ratio_threshold=args.length_ratio_threshold,
            ocr_confidence_threshold=args.ocr_confidence_threshold,
        )
        filter_payload = {
            "description": (
                "过滤后待构造 SFT 的图片列表; 原始 VLM json 与图片未被修改"
            ),
            "vlm_dirs": [str(p) for p in args.vlm_dirs],
            "image_dir": str(args.image_dir),
            "num_images": len(kept),
            "stats": filter_stats,
            "images": kept,
        }
        _write_json(filtered_path, filter_payload)
        _log(
            f"[filter] 完成: kept_images={len(kept)} "
            f"kept_boxes={filter_stats.get('paragraphs_kept')} "
            f"-> {filtered_path}"
        )

    box_dist = compute_box_count_distribution(kept)
    max_boxes_n = max(1, int(args.max_boxes))
    box_dist["resolved_max_boxes_n"] = max_boxes_n
    box_dist["resolved_max_boxes_source"] = f"cli:--max-boxes={max_boxes_n}"
    _write_json(box_dist_path, box_dist)
    _log(
        f"[stats] images={box_dist['num_images']} boxes={box_dist['total_valid_boxes']} "
        f"mean={box_dist['mean']} median={box_dist['median']} "
        f"P75={box_dist['p75']} P90={box_dist['p90']} max={box_dist['max']} "
        f"→ N={max_boxes_n}"
    )

    if args.filter_only:
        _write_json(
            stats_path,
            {
                "filter": filter_stats,
                "box_count_distribution": box_dist,
                "build": None,
            },
        )
        _log("[done] --filter-only, 跳过构建")
        return

    if not 0.0 < args.eval_ratio < 1.0:
        raise SystemExit("--eval-ratio 必须在 (0, 1) 内")
    try:
        k_ratios = (
            parse_k_ratios(args.k_ratios) if args.k_ratios.strip() else DEFAULT_K_RATIOS
        )
    except ValueError as exc:
        raise SystemExit(f"--k-ratios 无效: {exc}") from exc
    max_k = max(k_ratios)
    if max_boxes_n < max_k:
        raise SystemExit(
            f"--max-boxes={max_boxes_n} 小于 --k-ratios 中的最大 K={max_k}"
        )
    k_ratio_desc = ",".join(f"{k}={v:g}" for k, v in k_ratios.items())

    build_list = kept[: args.limit_images] if args.limit_images > 0 else kept
    _log(
        f"[split] 对 {len(build_list)}/{len(kept)} 条过滤记录计算 SHA256 并按原图切分 ..."
    )
    train_images, eval_images, split_stats = prepare_source_level_split(
        build_list,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    _write_json(split_stats_path, split_stats)
    _log(
        f"[split] train={len(train_images)} eval={len(eval_images)} "
        f"source_groups={split_stats['num_source_groups']} "
        f"exact_duplicates_dropped={split_stats['num_dropped_exact_duplicate_records']}"
    )

    _log(
        f"[plan] 训练集：框互斥，K={k_ratio_desc}，类型=60/30/10 ..."
    )
    train_plans, overflow_rows, train_plan_stats = plan_train_disjoint_samples(
        train_images,
        seed=args.seed,
        target_samples=args.target_train_samples,
        k_ratios=k_ratios,
    )
    _log("[plan] 验证集：每图一条，K 分层，类型保持自然分布 ...")
    eval_plans, eval_plan_stats = plan_eval_one_sample_per_image(
        eval_images,
        seed=args.seed,
        k_ratios=k_ratios,
    )
    plan_stats = {
        "rules": {
            "source_split": "source_image_id_else_sha256",
            "eval_ratio": args.eval_ratio,
            "train_k_ratios": {str(k): v for k, v in k_ratios.items()},
            "train_sample_type_ratios": DEFAULT_TRAIN_TYPE_RATIOS,
            "train_mixed_positive_ratio": DEFAULT_TRAIN_MIXED_POSITIVE_RATIO,
            "target_train_positive_box_ratio": 0.60,
            "eval_sample_type_ratios": DEFAULT_EVAL_TYPE_RATIOS,
            "eval_mixed_positive_ratio": DEFAULT_EVAL_MIXED_POSITIVE_RATIO,
            "train_boxes_disjoint": True,
            "eval_one_sample_per_image_record": True,
        },
        "split": {
            key: value for key, value in split_stats.items()
            if key not in {"assignments", "dropped_exact_duplicates"}
        },
        "train": train_plan_stats,
        "eval": eval_plan_stats,
    }
    _write_json(plan_stats_path, plan_stats)
    _write_jsonl(overflow_path, overflow_rows)
    _log(
        f"[plan] train={len(train_plans)} "
        f"K={train_plan_stats['planned_by_k']} "
        f"types={train_plan_stats['planned_by_sample_type']} "
        f"positive_boxes={train_plan_stats['positive_box_ratio']:.1%} "
        f"overflow_boxes={train_plan_stats['num_overflow_boxes']}"
    )
    _log(
        f"[plan] eval={len(eval_plans)} "
        f"K={eval_plan_stats['planned_by_k']} "
        f"types={eval_plan_stats['planned_by_sample_type']}"
    )

    if args.plan_only:
        _write_json(
            stats_path,
            {
                "filter": filter_stats,
                "box_count_distribution": box_dist,
                "plan": plan_stats,
                "build": None,
            },
        )
        _log("[done] --plan-only，仅写 split/plan/overflow 统计")
        return

    if args.clean_samples:
        import subprocess
        samples_dir = out_dir / "samples"
        if samples_dir.is_dir():
            _log(f"[build] 清空旧样本: {samples_dir}")
            subprocess.run(["rm", "-rf", str(samples_dir)], check=True)
        for old_name in (
            "manifest.jsonl",
            "manifest_train.jsonl",
            "manifest_eval.jsonl",
            "sft_datapairs.json",
            "sft_datapairs_train.json",
            "sft_datapairs_eval.json",
            "per_image_sample_stats.json",
        ):
            old_path = out_dir / old_name
            if old_path.is_file():
                old_path.unlink()
    (out_dir / "samples" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples" / "eval").mkdir(parents=True, exist_ok=True)

    _log(f"[build] 落盘 train={len(train_plans)} ...")
    train_records, train_build = build_samples_from_plans(
        train_images,
        train_plans,
        out_dir,
        max_boxes_n=max_boxes_n,
        crop_pad=args.crop_pad,
        log_every=args.log_every,
    )
    _log(f"[build] 落盘 eval={len(eval_plans)} ...")
    eval_records, eval_build = build_samples_from_plans(
        eval_images,
        eval_plans,
        out_dir,
        max_boxes_n=max_boxes_n,
        crop_pad=args.crop_pad,
        log_every=args.log_every,
    )

    manifest_train = out_dir / "manifest_train.jsonl"
    manifest_eval = out_dir / "manifest_eval.jsonl"
    datapair_train = out_dir / "sft_datapairs_train.json"
    datapair_eval = out_dir / "sft_datapairs_eval.json"
    per_image_stats_path = out_dir / "per_image_sample_stats.json"
    _write_jsonl(manifest_train, train_records)
    _write_jsonl(manifest_eval, eval_records)

    common_description = {
        "prompt_source": "SFT_M3_optimized.py",
        "schema": ["should_translate", "original_text", "translated_text"],
        "construction_rule": (
            "source-level 92/8 split; disjoint selected boxes within each train "
            f"image; K ratios {k_ratio_desc}; train sample types "
            "mixed/all_positive/all_negative=60/30/10; mixed positive-box "
            "selection target=50% (discrete actual about 55%); explicit OCR #pid labels"
        ),
    }
    _write_json(
        datapair_train,
        {
            **common_description,
            "split": "train",
            "num_samples": len(train_records),
            "plan": train_plan_stats,
            "samples": train_records,
        },
    )
    _write_json(
        datapair_eval,
        {
            **common_description,
            "split": "eval",
            "num_samples": len(eval_records),
            "plan": eval_plan_stats,
            "samples": eval_records,
        },
    )
    _write_json(
        per_image_stats_path,
        {
            "train": train_build.get("per_image") or [],
            "eval": eval_build.get("per_image") or [],
        },
    )

    build_stats = {
        "construction_rule": "source_split_disjoint_boxes_v2",
        "seed": args.seed,
        "train": train_build,
        "eval": eval_build,
        "outputs": {
            "filtered_images": str(filtered_path),
            "split_assignments": str(split_stats_path),
            "plan_stats": str(plan_stats_path),
            "overflow_boxes": str(overflow_path),
            "manifest_train": str(manifest_train),
            "manifest_eval": str(manifest_eval),
            "sft_datapairs_train": str(datapair_train),
            "sft_datapairs_eval": str(datapair_eval),
            "samples_dir": str(out_dir / "samples"),
        },
    }
    _write_json(
        stats_path,
        {
            "filter": filter_stats,
            "box_count_distribution": box_dist,
            "plan": plan_stats,
            "build": build_stats,
        },
    )
    _log(
        f"[done] train={len(train_records)} eval={len(eval_records)} out={out_dir}"
    )


if __name__ == "__main__":
    main()
