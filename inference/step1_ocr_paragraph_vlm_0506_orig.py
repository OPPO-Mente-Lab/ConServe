#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""
Step1 - OCR paragraph cropping + Qwen3-VL batch translation (vLLM continuous batching concurrency)

=========================================================================
  What this script does
=========================================================================

Inputs
  1. JSON returned by the online OCR + layout analysis (containing items[],
     paragraphNo, polygon, paragraph_polygon, etc.)
  2. Original images (.jpg/.png)

Pipeline
  Step A: Parse the OCR JSON, aggregate by ``paragraphNo``, and build the
          outer bbox of each paragraph
  Step B: Draw boxes on the original image with OpenCV
          - thin blue line = item polygon (text box, text_bbox)
          - thick green line = paragraph polygon (paragraph/layout box, layout_bbox)
          - red text = paragraphNo
          Output: ``<output_dir>/<stem>__bbox.jpg``
  Step C: Crop a thumbnail for each paragraph by its paragraph bbox (optional pad in pixels)
          Output: ``<output_dir>/crops/<stem>__p<pid>.jpg``
  Step D: Feed all cropped thumbnails to Qwen3-VL-8B in a single batch (vLLM's
          continuous batching is itself the "concurrency"; with one instance
          this is the most efficient concurrency mode)
          Each paragraph gets its own prompt, with the recognized OCR text attached as a hint
  Step E: Collect per-paragraph outputs + _meta, merge and save
          Output: ``<output_dir>/<stem>__vlm.json``

The VLM output for each paragraph is strict JSON:
  {
    "should_translate": bool,
    "skip_reason":      str | null,
    "original_text":    str,
    "translated_text":  str,
    "alignment":        "left" | "center" | "right"
  }

=========================================================================
  Differences from step1_vlm_translate_open_vllm.py
=========================================================================

- This script **does not let the VLM output coordinates**; coordinates are
  provided entirely by the online OCR -> accuracy is guaranteed
- The VLM only does "semantic" work: should_translate / translation / alignment
- Paragraph-level crops + OCR-text hint -> cleaner VLM context, fewer tokens,
  lower latency

=========================================================================
  Typical usage
=========================================================================

Single image:
  python step1_ocr_paragraph_vlm.py \\
      --ocr-json /path/to/test1/fullscreen/ocr_json/\\
2b0_1773946279648523_0-1773946280_ocr.json \\
      --image    /path/to/test1/fullscreen/\\
2b0_1773946279648523_0-1773946280.jpg \\
      --model-path /path/to/models/Qwen3-VL-8B-Instruct \\
      --output-dir ./out/ocr_paragraph_vlm \\
      --target-lang English

Batch (one OCR directory + one image directory):
  python step1_ocr_paragraph_vlm.py \\
      --ocr-dir   /path/to/test1/fullscreen/ocr_json \\
      --image-dir /path/to/test1/fullscreen \\
      --model-path /path/to/models/Qwen3-VL-8B-Instruct \\
      --output-dir ./out/ocr_paragraph_vlm \\
      --target-lang English \\
      --vllm-max-model-len 32768 \\
      --vllm-max-pixels 1003520
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# # Reuse the vLLM loading / prompt / environment from batch_photo_translate_vlm
# _REPO_ROOT = Path(__file__).resolve().parent
# _VLM_TRANSLATE_CANDIDATES = [
#     _REPO_ROOT / "VLM-translate",
#     Path(__file__).resolve().parent,
# ]
# for _cand in _VLM_TRANSLATE_CANDIDATES:
#     if _cand.is_dir() and str(_cand) not in sys.path:
#         sys.path.insert(0, str(_cand))

try:
    from batch_photo_translate_vlm import (  # noqa: E402
        _apply_chat_template_no_think,
        _cleanup,
        _load_vllm_vl_engine,
        _strip_thinking,
        _warn_if_vllm_without_cuda,
        detect_model_kind,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "无法导入 batch_photo_translate_vlm —— 请保持 image-translation/ 与 "
        f"VLM-translate/ 目录结构一致。原始错误: {e}"
    ) from e


# =============================================================================
#  Prompt
# =============================================================================

# Key design (aligned with file 2, step1_ocr_paragraph_vlm.py; only 6 new fields added):
#   (1) The VLM only sees the image and **receives no OCR text at all** (inaccurate
#       OCR would bias the VLM into hallucinations)
#   (2) The VLM outputs both orig_text and translated_text; sharing one source
#       guarantees the two are aligned
#   (3) **Conservatism first**: when the VLM cannot read clearly, is blurry or
#       unsure -> always false; prefer missing a translation over wild guessing
#   (4) **Anti in-context anchoring** (for easily contaminated continuous-value fields):
#       - In the schema examples, hex colors / pixel font sizes use <#RRGGBB> / <INT>
#         placeholders instead of concrete values, so the model has no anchor to copy
#       - Categorical fields (block_type/bg_type/font_family/alignment, etc.) keep the
#         concrete example values from file 2 (these are enums and do not pollute
#         continuous values)
#       - Pixel-level fields may be null when unmeasurable; the hard requirements
#         forbid defaulting to #FFFFFF/#000000
# Two valid outputs:
#   (A) Translation needed: compact 16-field JSON, strict field order
#   (B) No translation needed: 3 fields {"block_type":"...","should_translate":false,"skip_reason":"..."}
SYSTEM_PROMPT = """你是图像翻译助手。输入是一张**已经裁剪好的段落小图**(仅含一个文本段落), 识别图中文字翻译为 {target_lang}, 输出一行紧凑 JSON。
注意 **只输出 VLM 视觉判断需要的字段**; 

## 输出格式 (二选一, 严格单行, 无解释, 无 markdown)

(A) 需要翻译 (16 字段, 顺序严格; 其中 <#RRGGBB> 与 <INT> 是占位符, 必须替换为从当前图实际测量出的真值):
{{"block_type":"body","should_translate":true,"skip_reason":null,"original_text":"Line 1\\nLine 2","translated_text":"第一行\\n第二行","text_color_hex":"<#RRGGBB>","bg_color_hex":"<#RRGGBB>","bg_type":"solid","font_family":"sans-serif","font_size_px":<INT>,"recommended_font_size_px":<INT>,"is_bold":false,"alignment":"left","block_angle":0,"overflow_strategy":"wrap","line_break_hints":[]}}

(B) 不需翻译 (3 字段):
{{"block_type":"body","should_translate":false,"skip_reason":"already_target_lang"}}

## 字段说明

block_type        : "title" / "header" / "body" / "button" / "label" / "caption" / "logo" / "footer" / "list_item" / "link" / "other"
should_translate  : true=需翻译 / false=不需
skip_reason       : false 时必填 (枚举见下); true 时写 null
original_text     : 图中原文, 逐字精确; 图中有换行用 "\\n" (JSON 转义, 解析后即 0x0A)
translated_text   : 译文; 可含 "\\n"; 长度 ≤ 原文字符数 × 1.3
text_color_hex    : 大写 hex "#RRGGBB"; 字芯主色 (描边/阴影忽略); 多色字取占比最大者; UI 中黑/白/深灰文字最常见, 实际就是黑就报 #000000, 实际就是白就报 #FFFFFF, 不要为了"看起来不一样"猜成 #FF00FF / #FF1493 等艳色; 看不清写 null
bg_color_hex      : 大写 hex "#RRGGBB"; 文字底色; 渐变/纹理取主色; 透明叠图取裁剪图边缘的承载页底色; UI 中白/灰/黑底最常见, 实际就是白/黑就如实报; 看不清写 null
bg_type           : "solid" / "gradient" / "texture" / "transparent"
font_family       : "sans-serif" (大多数 UI) / "serif" / "monospace" / "display" / "handwriting" / "other"
font_size_px      : 正整数像素; 大写字母/汉字字身高度 (cap-height, 不含行距); 多行取行高中位
                    典型范围: 正文/小字 12-20; 标签/按钮 14-28; 副标题 20-32; 主标题 28-48; 巨型横幅 50-80; **罕见超过 80**
                    估算法: 单行段落 font_size_px ≈ 段落小图的可见文字行高 × 0.8 (注意是图中实际像素行高, 不是 CSS 倍数)
                    切忌套用 24 / 48 / 72 / 120 这种"网页常用整数"; 看不清写 null
recommended_font_size_px : 正整数像素; 译文不长于原文则 = font_size_px; 译文更长按字符比缩小, 下限 max(8, font_size_px × 0.6); font_size_px 为 null 则也写 null
is_bold           : true / false
alignment         : "left" / "center" / "right"
block_angle       : 整数度数 (-180~180); 水平向右 = 0, 顺时针为正; 90/-90 = 竖排; UI 段落绝大多数 = 0
overflow_strategy : "shrink" / "wrap" / "truncate" / "keep"
line_break_hints  : 译文里强制折行的字符下标数组 (整数, 从 0 起算, 该下标处之前断开); 单行段落 / 不需折行 → []

## skip_reason 枚举 (should_translate=false 时)
  already_target_lang  : 图中文字已是 {target_lang}
  pure_digits          : 纯数字/符号/标点/时间戳
  url_or_email         : URL / email / 代码标识符
  single_char          : 单字符
  brand_or_logo        : 商标 / Logo (Nike / Apple / 可口可乐 等)
  person_name          : 人名全名
  blurred_or_cropped   : 模糊 / 残缺 / 读不清
  unsure               : 没信心
  other                : 其他

## 判断规则 (保守优先: 宁可漏翻不要幻觉)
true 必要条件: 文字清晰可 100% 识别, 是自然语言, 不是 {target_lang} 本身, 不属于 skip_reason。

## 硬性要求
- 严格单行紧凑 JSON; 禁止换行 / 解释 / markdown 围栏
- 看不清 → (B) + 对应 skip_reason, 严禁编造
- (A) 字段顺序严格, 一个不多一个不少
- (B) 仅输出 {{"block_type":"...","should_translate":false,"skip_reason":"..."}}, 不得多生成字符
- text_color_hex / bg_color_hex / font_size_px / recommended_font_size_px 必须从当前图独立测量, 严禁原样输出占位符 <#RRGGBB> / <INT>
- font_size_px 必须落在视觉合理范围 (绝大多数 UI 段落 12-48, 巨型横幅最多 80); 严禁套用 24 / 48 / 72 / 120 等"网页常用倍数"作为偷懒填充; 看不清写 null
- 颜色字段如实报告, 黑/白/灰是 UI 最常见色, 真就是黑/白时直接写 #000000 / #FFFFFF; 严禁照搬 #F8ECDE / #B73D3D 这类历史污染值, 也不要反向猜成 #FF00FF / #FF1493 这类艳色; 看不清写 null
"""

# Minimal user prompt: **no OCR is passed**; the VLM only sees the image and the target language.
# The user prompt is identical for all paragraphs, so prefix caching hits 100%.
USER_PROMPT_TEMPLATE = (
    "目标语言: {target_lang}。只输出符合 schema 的单行 JSON。"
)


# =============================================================================
#  Data structures
# =============================================================================

def _bbox_from_polygon(polygon: list[dict]) -> list[int] | None:
    if not polygon:
        return None
    xs = [int(p["x"]) for p in polygon]
    ys = [int(p["y"]) for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _merge_bboxes(bboxes: list[list[int]]) -> list[int]:
    xs1 = [b[0] for b in bboxes]
    ys1 = [b[1] for b in bboxes]
    xs2 = [b[2] for b in bboxes]
    ys2 = [b[3] for b in bboxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


# -----------------------------------------------------------------------------
#  Pre-filter: paragraphs that "definitely do not need translation" are decided by
#  rules directly and are never sent to the VLM
# -----------------------------------------------------------------------------
_SKIP_PURE_DIGIT_RE = re.compile(r"^[\d\s.,/:\-–+%¥$€£@()#*×²³]+$")
_SKIP_URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
_SKIP_EMAIL_RE = re.compile(r"^\S+@\S+\.\S+$")
_SKIP_ASCII_PUNCT_RE = re.compile(r"^[\x20-\x7E]*[\W\d_]+$")  # all ASCII with no letters


def classify_no_translate(text: str, min_alpha_chars: int = 2) -> tuple[bool, str | None]:
    """Look only at the OCR text and decide whether the VLM can be skipped on the Python side.

    Returns (should_skip, reason). should_skip=True means the paragraph is not
    sent to the VLM and directly produces should_translate=False.
    """
    t = (text or "").strip()
    if not t:
        return True, "empty"
    if len(t) <= 1:
        # A single letter/character has almost no translation value
        return True, "single_char"
    if _SKIP_PURE_DIGIT_RE.match(t):
        return True, "pure_digits_or_symbols"
    if _SKIP_URL_RE.match(t):
        return True, "url"
    if _SKIP_EMAIL_RE.match(t):
        return True, "email"
    # Short ASCII snippets containing letters (e.g. "OK", "HD", "4G", "USB"):
    # usually symbols / abbreviations
    if t.isascii() and len(t) <= 4 and sum(c.isalpha() for c in t) < min_alpha_chars:
        return True, "short_ascii_code"
    return False, None


def group_by_paragraph(ocr_items: list[dict]) -> dict[int, dict[str, Any]]:
    """Aggregate OCR items by paragraphNo, producing the outer bbox and joined original text of each paragraph."""
    groups: dict[int, dict[str, Any]] = {}
    for it in ocr_items:
        pid = int(it.get("paragraphNo", -1))
        if pid not in groups:
            groups[pid] = {
                "paragraph_no": pid,
                "items": [],
                "item_bboxes": [],
                "paragraph_bbox": None,
                "texts": [],
                "text_colors": [],
                "bg_colors": [],
                "angles": [],
                "mean_confidence": 0.0,
            }
        g = groups[pid]
        bbox = _bbox_from_polygon(it.get("polygon", []))
        if bbox is not None:
            g["item_bboxes"].append(bbox)
        text = (it.get("text_suggest") or it.get("text") or "").strip()
        if text:
            g["texts"].append(text)
        g["items"].append(it)
        if it.get("text_color"):
            g["text_colors"].append(list(it["text_color"]))
        if it.get("background_color"):
            g["bg_colors"].append(list(it["background_color"]))
        if "angle" in it:
            try:
                g["angles"].append(int(it["angle"]))
            except (TypeError, ValueError):
                pass
        pp_bbox = _bbox_from_polygon(it.get("paragraph_polygon", []))
        if pp_bbox is not None:
            if g["paragraph_bbox"] is None:
                g["paragraph_bbox"] = pp_bbox
            else:
                g["paragraph_bbox"] = _merge_bboxes([g["paragraph_bbox"], pp_bbox])

    # Fall back to the outer bbox of all item bboxes when paragraph_polygon is missing
    for g in groups.values():
        if g["paragraph_bbox"] is None and g["item_bboxes"]:
            g["paragraph_bbox"] = _merge_bboxes(g["item_bboxes"])
        confs = [float(it.get("confidence", 0)) for it in g["items"]]
        g["mean_confidence"] = round(sum(confs) / max(1, len(confs)), 4)

    return groups


# =============================================================================
#  Visualization & cropping
# =============================================================================

def draw_bbox_visualization(
    image_bgr: np.ndarray,
    groups: dict[int, dict[str, Any]],
    output_path: Path,
    show_item_bbox: bool = True,
) -> None:
    """Draw paragraph_bbox (green) + item bbox (blue) + pid labels (red) on the original image"""
    vis = image_bgr.copy()

    if show_item_bbox:
        for g in groups.values():
            for b in g["item_bboxes"]:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 120, 0), 1)

    for pid, g in groups.items():
        if g["paragraph_bbox"] is None:
            continue
        x1, y1, x2, y2 = g["paragraph_bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"P{pid}"
        cv2.putText(
            vis, label, (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def crop_paragraphs(
    image_bgr: np.ndarray,
    groups: dict[int, dict[str, Any]],
    crop_dir: Path,
    stem: str,
    pad: int = 6,
    save_to_disk: bool = True,
) -> dict[int, dict[str, Any]]:
    """Crop each paragraph by its paragraph_bbox.

    save_to_disk=True  -> write JPGs to crop_dir and return crop_path
    save_to_disk=False -> do not write to disk; keep only the PIL.Image in memory
                          (crop_pil), with crop_path=None

    Returns {pid: {crop_path, crop_pil, crop_w, crop_h, padded_bbox}}
    """
    from PIL import Image as PILImage

    if save_to_disk:
        crop_dir.mkdir(parents=True, exist_ok=True)
    h, w = image_bgr.shape[:2]
    out: dict[int, dict[str, Any]] = {}
    for pid, g in groups.items():
        bbox = g["paragraph_bbox"]
        if bbox is None:
            continue
        x1 = max(0, bbox[0] - pad)
        y1 = max(0, bbox[1] - pad)
        x2 = min(w, bbox[2] + pad)
        y2 = min(h, bbox[3] + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_bgr[y1:y2, x1:x2]
        crop_path: str | None = None
        crop_pil = None
        if save_to_disk:
            crop_path = str(crop_dir / f"{stem}__p{pid:03d}.jpg")
            cv2.imwrite(crop_path, crop)
        else:
            crop_pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        out[pid] = {
            "crop_path": crop_path,
            "crop_pil": crop_pil,
            "crop_w": crop.shape[1],
            "crop_h": crop.shape[0],
            "padded_bbox": [x1, y1, x2, y2],
        }
    return out


# =============================================================================
#  VLM batch calls (vLLM continuous batching = concurrency)
# =============================================================================

def _safe_parse_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {"_parse_error": True, "_raw": ""}
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Truncation repair: when generation is cut off by max_new_tokens, extract the
    # already generated fields and assemble a minimally usable JSON
    recovered = _recover_truncated_json(cleaned)
    if recovered is not None:
        recovered["_recovered_from_truncation"] = True
        return recovered
    return {"_parse_error": True, "_raw": cleaned[:2000]}


# Match fully closed key-value pairs (the value must have its closing quote / be a
# number / bool / null)
_KV_PAIR_RE = re.compile(
    r'"(?P<k>block_type|should_translate|skip_reason|original_text|orig_text|translated_text|'
    r'text_color_hex|bg_color_hex|bg_type|font_family|font_size_px|recommended_font_size_px|'
    r'is_bold|alignment|block_angle|overflow_strategy)"\s*:\s*'
    r'(?P<v>"(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?)',
    re.DOTALL,
)

# Match array-typed values (line_break_hints): only integer arrays / empty arrays accepted
_KV_ARRAY_RE = re.compile(
    r'"(?P<k>line_break_hints)"\s*:\s*(?P<v>\[\s*(?:-?\d+(?:\s*,\s*-?\d+)*\s*)?\])',
    re.DOTALL,
)

# Match a truncated trailing value (no closing quote; generation was cut off by max_tokens)
_DANGLING_STR_RE = re.compile(
    r'"(?P<k>original_text|orig_text|translated_text|text_color_hex|bg_color_hex)"\s*:\s*'
    r'"(?P<v>(?:\\.|[^"\\])*)$',
    re.DOTALL,
)


def _recover_truncated_json(text: str) -> dict[str, Any] | None:
    """When generation is cut off mid-field, extract completed key-value pairs key by
    key with regexes; a truncated trailing orig_text / translated_text is also
    salvaged as best as possible (the content after removing the broken quote is
    used as the value). It counts as rescued when should_translate=false, or when
    should_translate=true and translated_text is present.
    """
    pairs: dict[str, Any] = {}
    for m in _KV_PAIR_RE.finditer(text):
        k = m.group("k")
        v_raw = m.group("v")
        try:
            pairs[k] = json.loads(v_raw)
        except json.JSONDecodeError:
            continue

    for m in _KV_ARRAY_RE.finditer(text):
        k = m.group("k")
        v_raw = m.group("v")
        try:
            pairs[k] = json.loads(v_raw)
        except json.JSONDecodeError:
            continue

    # Try to salvage the last unclosed string value
    tail = _DANGLING_STR_RE.search(text)
    if tail:
        k = tail.group("k")
        if k not in pairs:
            v_raw = tail.group("v")
            # JSON unescape (\" \\ \n, etc.)
            try:
                pairs[k] = json.loads('"' + v_raw + '"')
            except json.JSONDecodeError:
                pairs[k] = v_raw  # fallback: keep as-is

    if not pairs:
        return None
    # Compatibility: the VLM occasionally outputs orig_text (renamed to original_text)
    if "orig_text" in pairs and "original_text" not in pairs:
        pairs["original_text"] = pairs.pop("orig_text")
    if pairs.get("should_translate") is False:
        return {
            "block_type": pairs.get("block_type", "body"),
            "should_translate": False,
            "skip_reason": pairs.get("skip_reason") or "other",
        }
    if pairs.get("should_translate") is True and pairs.get("translated_text"):
        return pairs
    return None


_VALID_ALIGNMENTS = {"left", "center", "right"}
_VALID_BG_TYPES = {"solid", "gradient", "texture", "transparent"}
_VALID_OVERFLOW = {"shrink", "wrap", "truncate", "keep"}
_VALID_BLOCK_TYPES = {
    "title", "header", "body", "button", "label", "caption",
    "logo", "footer", "list_item", "link", "other",
}
_VALID_SKIP_REASONS = {
    "already_target_lang", "pure_digits", "url_or_email", "single_char",
    "brand_or_logo", "person_name", "blurred_or_cropped", "unsure", "other",
}
_VALID_FONT_FAMILIES = {
    "sans-serif", "serif", "monospace", "display", "handwriting", "other",
}


_HEX_COLOR_RE = re.compile(r"^#?([0-9A-Fa-f]{6})$")


def _normalize_hex_color(s: Any) -> str | None:
    """Normalize a VLM-produced color string to uppercase '#RRGGBB'; return None if invalid.

    Accepts 'F8ECDE' / '#f8ecde' / '#F8ECDE'; rejects the '#RGB' shorthand / RGBA / 'rgb(...)'.
    """
    if not isinstance(s, str):
        return None
    m = _HEX_COLOR_RE.match(s.strip())
    if not m:
        return None
    return "#" + m.group(1).upper()


def _rgb_to_hex(rgb: list[int] | tuple[int, ...] | None) -> str | None:
    """[R, G, B] (0-255) -> '#RRGGBB'. Returns None when empty/invalid."""
    if not rgb or len(rgb) < 3:
        return None
    try:
        r, g, b = (max(0, min(255, int(c))) for c in rgb[:3])
    except (TypeError, ValueError):
        return None
    return f"#{r:02X}{g:02X}{b:02X}"


def _avg_rgb(rgbs: list[list[int]]) -> list[int] | None:
    """Average multiple RGB groups (kept for compatibility)."""
    if not rgbs:
        return None
    n = len(rgbs)
    r = sum(c[0] for c in rgbs) // n
    g = sum(c[1] for c in rgbs) // n
    b = sum(c[2] for c in rgbs) // n
    return [r, g, b]


def _median_rgb(rgbs: list[list[int]]) -> list[int] | None:
    """Per-channel median, robust to outliers."""
    if not rgbs:
        return None
    arr = np.asarray(rgbs, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return [int(np.median(arr[:, 0])), int(np.median(arr[:, 1])), int(np.median(arr[:, 2]))]


def _dominant_angle(angles: list[int]) -> int:
    if not angles:
        return 0
    from collections import Counter as _Counter
    return _Counter(angles).most_common(1)[0][0]


def _estimate_font_size_px(item_bboxes: list[list[int]]) -> int | None:
    """Roughly estimate the font size as median item bbox height x 0.85."""
    if not item_bboxes:
        return None
    heights = sorted(max(1, b[3] - b[1]) for b in item_bboxes)
    return max(8, int(round(heights[len(heights) // 2] * 0.85)))


def _recommend_font_size_px(
    font_size_px: int | None, orig_text: str, translated_text: str,
    min_floor: int = 8, min_ratio: float = 0.6,
) -> int | None:
    """When the translation is longer, shrink the font size in proportion to character count; floor = max(8, orig size x 0.6)."""
    if not font_size_px or font_size_px <= 0:
        return None
    lo = len(orig_text) if orig_text else 0
    lt = len(translated_text) if translated_text else 0
    if lt == 0 or lo == 0 or lt <= lo:
        return int(font_size_px)
    scaled = font_size_px * (lo / lt)
    floor = max(min_floor, int(round(font_size_px * min_ratio)))
    return max(floor, int(round(scaled)))


def _map_rule_to_skip_reason(rule: str) -> str:
    """Map the internal reason returned by classify_no_translate to the public skip_reason enum."""
    mapping = {
        "empty": "other",
        "single_char": "single_char",
        "pure_digits_or_symbols": "pure_digits",
        "url": "url_or_email",
        "email": "url_or_email",
        "short_ascii_code": "other",
    }
    return mapping.get(rule, "other")


def _infer_overflow_strategy(orig_text: str, translated_text: str) -> str:
    lo = len(orig_text) if orig_text else 0
    lt = len(translated_text) if translated_text else 0
    if lo == 0 or lt <= lo:
        return "keep"
    ratio = lt / lo
    if ratio > 1.6:
        return "wrap"
    return "shrink"


def _finalize_schema(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize the VLM output. Branch A fills in missing fields; branch B keeps block_type + skip_reason."""
    if not isinstance(d, dict) or "_parse_error" in d:
        return d
    out = dict(d)

    # Compatibility with the old field name
    if "orig_text" in out and "original_text" not in out:
        out["original_text"] = out.pop("orig_text")

    block_type = str(out.get("block_type", "") or "").strip().lower()
    out["block_type"] = block_type if block_type in _VALID_BLOCK_TYPES else "body"

    if "should_translate" in out:
        out["should_translate"] = bool(out["should_translate"])

    if out.get("should_translate") is False:
        skip = str(out.get("skip_reason", "") or "").strip().lower()
        return {
            "block_type": out["block_type"],
            "should_translate": False,
            "skip_reason": skip if skip in _VALID_SKIP_REASONS else "other",
        }

    align = str(out.get("alignment", "") or "").strip().lower()
    out["alignment"] = align if align in _VALID_ALIGNMENTS else "left"

    bg = str(out.get("bg_type", "") or "").strip().lower()
    # Migrate the legacy value "image" to "texture"
    if bg == "image":
        bg = "texture"
    out["bg_type"] = bg if bg in _VALID_BG_TYPES else "solid"

    overflow = str(out.get("overflow_strategy", "") or "").strip().lower()
    out["overflow_strategy"] = overflow if overflow in _VALID_OVERFLOW else ""

    out["is_bold"] = bool(out.get("is_bold"))
    out["original_text"] = str(out.get("original_text", "") or "")
    out["translated_text"] = str(out.get("translated_text", "") or "")
    out["skip_reason"] = None

    # Normalize font_family; fold common aliases
    ff = str(out.get("font_family", "") or "").strip().lower()
    ff_aliases = {
        "sans": "sans-serif", "sansserif": "sans-serif",
        "sans_serif": "sans-serif", "sans serif": "sans-serif",
        "mono": "monospace", "cursive": "handwriting",
        "fantasy": "display",
    }
    ff = ff_aliases.get(ff, ff)
    out["font_family"] = ff if ff in _VALID_FONT_FAMILIES else "sans-serif"

    out["text_color_hex"] = _normalize_hex_color(out.get("text_color_hex"))
    out["bg_color_hex"] = _normalize_hex_color(out.get("bg_color_hex"))

    def _coerce_pos_int(v: Any) -> int | None:
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        return iv if iv > 0 else None

    out["font_size_px"] = _coerce_pos_int(out.get("font_size_px"))
    out["recommended_font_size_px"] = _coerce_pos_int(out.get("recommended_font_size_px"))

    try:
        out["block_angle"] = int(round(float(out.get("block_angle", 0) or 0)))
    except (TypeError, ValueError):
        out["block_angle"] = 0

    hints_raw = out.get("line_break_hints")
    valid_hints: list[int] = []
    if isinstance(hints_raw, list):
        for h in hints_raw:
            try:
                iv = int(h)
            except (TypeError, ValueError):
                continue
            if iv >= 0:
                valid_hints.append(iv)
    out["line_break_hints"] = valid_hints

    return out


# Notes on Qwen3-VL-Instruct ===============================================
# Qwen3-VL-8B-Instruct is the non-thinking variant; it was not trained on
# <think>/</think> data, so it naturally never enters the thinking stage and
# needs no prefill trick. With enable_thinking=False on the chat_template
# (going through _apply_chat_template_no_think), the standard Instruct path
# is sufficient. The previous `<think>\n\n</think>\n\n` prefill is an
# out-of-distribution prefix for this model and has been removed.


def build_vlm_inputs(
    processor: Any,
    groups: dict[int, dict[str, Any]],
    crops_info: dict[int, dict[str, Any]],
    target_lang: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Build vLLM batch inputs; returns (inputs_list, pid_order).

    The VLM only sees the image and **receives no OCR text**. The user prompt is
    identical for all paragraphs, so prefix caching hits 100%; the prompt string
    is built only once and reused across paragraphs.
    """
    from PIL import Image as PILImage

    inputs: list[dict[str, Any]] = []
    pid_order: list[int] = []

    system_prompt = SYSTEM_PROMPT.format(target_lang=target_lang)
    user_prompt = USER_PROMPT_TEMPLATE.format(target_lang=target_lang)
    messages_template = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "__PLACEHOLDER__"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    prompt_str = _apply_chat_template_no_think(
        processor, messages_template,
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    for pid, g in groups.items():
        if pid not in crops_info:
            continue
        info = crops_info[pid]
        pil = info.get("crop_pil")
        if pil is None:
            pil = PILImage.open(info["crop_path"]).convert("RGB")
        inputs.append({"prompt": prompt_str, "multi_modal_data": {"image": pil}})
        pid_order.append(pid)

    return inputs, pid_order


def run_vlm_batch(
    llm: Any,
    inputs: list[dict[str, Any]],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    """Submit to vLLM in one shot; continuous batching decodes concurrently.

    Returns (results, wall). results[i] contains parsed / raw_text / n_prompt /
    n_completion / early_exit.
    """
    from vllm import SamplingParams

    # Early-exit stop sequences:
    #   (A) The last field of the translation branch is the line_break_hints
    #       array, which ends with ']}'
    #       e.g. ...,"line_break_hints":[]}  /  ...,"line_break_hints":[3,7]}
    #   (B) The no-translation branch ends with the skip_reason string "xxx"}
    #   '"}' / ']}' are fallbacks so decoding can still exit early when the model
    #   lazily omits fields
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        stop=[
            ']}',                                              # branch A
            '"already_target_lang"}', '"pure_digits"}',        # common skip_reason values of branch B
            '"url_or_email"}', '"single_char"}',
            '"brand_or_logo"}', '"person_name"}',
            '"blurred_or_cropped"}', '"unsure"}', '"other"}',
            '"}',                                              # fallback
        ],
        include_stop_str_in_output=True,
    )

    t0 = time.perf_counter()
    outs = llm.generate(inputs, sampling_params=sp, use_tqdm=False) if inputs else []
    wall = time.perf_counter() - t0

    results: list[dict[str, Any]] = []
    for ro in outs:
        raw = (ro.outputs[0].text or "").strip()
        text = _strip_thinking(raw)
        parsed = _safe_parse_json(text)
        try:
            n_prompt = len(ro.prompt_token_ids or [])
        except AttributeError:
            n_prompt = 0
        try:
            n_completion = len(ro.outputs[0].token_ids or [])
        except AttributeError:
            n_completion = 0
        results.append({
            "parsed": parsed,
            "raw_text": text,
            "n_prompt": n_prompt,
            "n_completion": n_completion,
            "early_exit": text.rstrip().endswith("false}"),
        })
    return results, wall


# =============================================================================
#  Single-image processing
# =============================================================================

def process_one(
    *,
    ocr_json_path: Path,
    image_path: Path,
    output_dir: Path,
    llm: Any,
    processor: Any,
    target_lang: str,
    max_new_tokens: int,
    crop_pad: int,
    save_visualize: bool = False,
    save_crops: bool = True,
) -> dict[str, Any]:
    """Parse OCR -> crop -> pre-filter -> VLM batch -> assemble the rich-format JSON.

    The output is aligned with the reference directory out_ocr-vlm-json_batch/*__vlm.json:
      top level {stem, image_path, ocr_json_path, image_size, ocr_size, target_lang,
            num_paragraphs, vlm_batch_wall_sec, vlm_avg_sec_per_paragraph,
            bbox_visualization, paragraphs[]}
      paragraphs[i] key = paragraph_no, containing vlm / vlm_ok / vlm_tokens / vlm_raw_preview
      and exposing 16 flattened fields, all produced directly by the VLM (including
      color / size / angle / line_break_hints):
        block_type / should_translate / skip_reason / original_text /
        translated_text / text_color_hex / bg_color_hex / bg_type /
        font_family / font_size_px / recommended_font_size_px / is_bold /
        alignment / block_angle / overflow_strategy / line_break_hints
    """
    stem = image_path.stem
    t_start = time.perf_counter()

    with open(ocr_json_path, encoding="utf-8") as f:
        ocr_raw = json.load(f)
    data = ocr_raw.get("data", ocr_raw)
    ocr_items = data.get("items", [])
    ocr_size = [int(data.get("width", 0)), int(data.get("height", 0))]

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_h, img_w = image_bgr.shape[:2]

    groups = group_by_paragraph(ocr_items)
    t_after_prep = time.perf_counter()
    if not groups:
        empty = {
            "stem": stem,
            "image_path": str(image_path),
            "ocr_json_path": str(ocr_json_path),
            "image_size": [img_w, img_h],
            "ocr_size": ocr_size,
            "target_lang": target_lang,
            "num_paragraphs": 0,
            "paragraphs": [],
            "_warning": "no_ocr_items",
        }
        (output_dir / f"{stem}__vlm.json").write_text(
            json.dumps(empty, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return empty

    # Visualization (off by default; enable with --visualize when debugging)
    vis_path_str: str | None = None
    if save_visualize:
        vis_path = output_dir / f"{stem}__bbox.jpg"
        draw_bbox_visualization(image_bgr, groups, vis_path, show_item_bbox=True)
        vis_path_str = str(vis_path)

    crop_dir = output_dir / "crops"
    crops_info = crop_paragraphs(
        image_bgr, groups, crop_dir, stem=stem, pad=crop_pad,
        save_to_disk=save_crops,
    )
    t_after_crop = time.perf_counter()

    # Rule-based pre-filter (pure digits / URL / email / single character /
    # short ASCII abbreviations skip the VLM directly)
    prefilter_reasons: dict[int, str] = {}
    vlm_groups: dict[int, dict[str, Any]] = {}
    for pid, g in groups.items():
        if pid not in crops_info:
            continue
        joined = " ".join(g["texts"]).strip()
        skip, reason = classify_no_translate(joined)
        if skip:
            prefilter_reasons[pid] = reason or "rule"
        else:
            vlm_groups[pid] = g

    inputs, pid_order = build_vlm_inputs(
        processor=processor,
        groups=vlm_groups,
        crops_info=crops_info,
        target_lang=target_lang,
    )
    vlm_results, decode_wall = run_vlm_batch(
        llm, inputs, max_new_tokens=max_new_tokens,
    )
    per_pid_vlm: dict[int, dict[str, Any]] = dict(zip(pid_order, vlm_results))

    t_after_decode = time.perf_counter()

    # Diagnostics: which sequences did not hit a stop string and slowed down decoding
    # of the whole batch -- expose their pids directly
    comp_lens = [(pid, per_pid_vlm[pid]["n_completion"]) for pid in pid_order]
    maxed = [(pid, n) for pid, n in comp_lens if n >= max_new_tokens * 0.95]
    max_comp = max((n for _, n in comp_lens), default=0)
    median_comp = sorted(n for _, n in comp_lens)[len(comp_lens) // 2] if comp_lens else 0

    prep_wall = t_after_prep - t_start            # OCR parsing + image reading + grouping
    crop_wall = t_after_crop - t_after_prep       # visualization + cropping
    # decode_wall is vLLM internal time (already returned by run_vlm_batch)
    # remainder = total - (prep+crop+decode), i.e. output assembly & disk writes
    # total_wall is computed at the end

    # Assemble paragraphs_out (rich format + 12 flattened fields)
    paragraphs_out: list[dict[str, Any]] = []
    per_paragraph_profile: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for pid in sorted(groups.keys()):
        g = groups[pid]
        info = crops_info.get(pid, {})
        item_bboxes = g["item_bboxes"]
        ocr_text_joined = " ".join(g["texts"]).strip()

        vlm_block: dict[str, Any] = {"should_translate": False, "skip_reason": None}
        prefiltered = pid in prefilter_reasons
        early_exit = False
        vlm_ok = True
        n_prompt = 0
        n_completion = 0
        raw_preview: str | None = None

        # Possible values after VLM parsing (new schema: the VLM directly emits the
        # 16 fields, including color / size / angle / hints)
        vlm_block_type = "body"
        vlm_skip_reason: str | None = None
        vlm_original_text = ""
        vlm_translated_text = ""
        vlm_bg_type: str | None = None
        vlm_font_family: str | None = None
        vlm_is_bold = False
        vlm_alignment: str | None = None
        vlm_overflow: str = ""
        vlm_text_color_hex: str | None = None
        vlm_bg_color_hex: str | None = None
        vlm_font_size_px: int | None = None
        vlm_recommended_font_size_px: int | None = None
        vlm_block_angle: int = 0
        vlm_line_break_hints: list[int] = []

        if prefiltered:
            vlm_skip_reason = _map_rule_to_skip_reason(prefilter_reasons[pid])
            vlm_block = {
                "block_type": "body",
                "should_translate": False,
                "skip_reason": vlm_skip_reason,
            }
        elif pid in per_pid_vlm:
            r = per_pid_vlm[pid]
            n_prompt = r["n_prompt"]
            n_completion = r["n_completion"]
            early_exit = bool(r["early_exit"])
            final = _finalize_schema(r["parsed"])
            recovered = bool(r["parsed"].get("_recovered_from_truncation")) \
                if isinstance(r["parsed"], dict) else False
            if isinstance(final, dict) and final.get("_parse_error"):
                vlm_ok = False
                raw_preview = (final.get("_raw") or r["raw_text"])[:500] or None
                vlm_block = {
                    "block_type": "body",
                    "should_translate": False,
                    "skip_reason": "vlm_parse_error",
                }
            elif not final.get("should_translate"):
                vlm_block_type = final.get("block_type", "body")
                vlm_skip_reason = final.get("skip_reason") or "other"
                vlm_block = {
                    "block_type": vlm_block_type,
                    "should_translate": False,
                    "skip_reason": vlm_skip_reason,
                }
                if recovered:
                    raw_preview = r["raw_text"][:500]
            else:
                vlm_block_type = final.get("block_type", "body")
                vlm_original_text = str(final.get("original_text", ""))
                vlm_translated_text = str(final.get("translated_text", ""))
                vlm_bg_type = final.get("bg_type") or "solid"
                vlm_font_family = final.get("font_family") or "sans-serif"
                vlm_is_bold = bool(final.get("is_bold", False))
                vlm_alignment = final.get("alignment") or "left"
                vlm_overflow = final.get("overflow_strategy") or ""
                vlm_text_color_hex = final.get("text_color_hex")
                vlm_bg_color_hex = final.get("bg_color_hex")
                vlm_font_size_px = final.get("font_size_px")
                vlm_recommended_font_size_px = final.get("recommended_font_size_px")
                vlm_block_angle = int(final.get("block_angle") or 0)
                hints_raw = final.get("line_break_hints")
                if isinstance(hints_raw, list):
                    vlm_line_break_hints = [int(h) for h in hints_raw if isinstance(h, int)]
                vlm_block = {
                    "block_type": vlm_block_type,
                    "should_translate": True,
                    "skip_reason": None,
                    "original_text": vlm_original_text,
                    "translated_text": vlm_translated_text,
                }
                if recovered:
                    vlm_block["skip_reason"] = "recovered_from_truncation"
                    raw_preview = r["raw_text"][:500]

        # --- Flattened-field merge ---
        #   All 16 fields come from the VLM output (including color / size /
        #   angle / line_break_hints)
        #   When should_translate=False there are no rendering fields; all are
        #   set to None / default values
        should_translate = bool(vlm_block.get("should_translate"))
        original_text = vlm_original_text if should_translate else ""
        translated_text = vlm_translated_text if should_translate else ""

        if should_translate:
            text_color_hex = vlm_text_color_hex
            bg_color_hex = vlm_bg_color_hex
            font_size_px = vlm_font_size_px
            recommended_font_size_px = vlm_recommended_font_size_px
            block_angle = vlm_block_angle
            valid_hints: list[int] = list(vlm_line_break_hints)
            bg_type = vlm_bg_type or "solid"
            font_family = vlm_font_family or "sans-serif"
            is_bold = vlm_is_bold
            alignment = vlm_alignment or "left"
            overflow = vlm_overflow or _infer_overflow_strategy(original_text, translated_text)
        else:
            text_color_hex = None
            bg_color_hex = None
            font_size_px = None
            recommended_font_size_px = None
            block_angle = 0
            valid_hints = []
            bg_type = None
            font_family = "sans-serif"
            is_bold = False
            alignment = "left"
            overflow = "keep"

        block_type = vlm_block.get("block_type", "body")
        skip_reason = vlm_block.get("skip_reason")

        entry: dict[str, Any] = {
            "paragraph_no": pid,
            "paragraph_bbox": g["paragraph_bbox"],
            "item_count": len(item_bboxes),
            "item_bboxes": item_bboxes,
            "ocr_text": ocr_text_joined,
            "ocr_mean_confidence": g.get("mean_confidence", 0.0),
            "crop_path": info.get("crop_path"),
            "padded_bbox": info.get("padded_bbox"),
            "prefiltered": prefiltered,
            "vlm_early_exit": early_exit,
            "vlm": vlm_block,
            "vlm_ok": vlm_ok,
            "vlm_tokens": {"prompt": n_prompt, "completion": n_completion},
            "vlm_raw_preview": raw_preview,

            # --- Flattened fields (aligned with the reference schema) ---
            "block_type": block_type,
            "should_translate": should_translate,
            "skip_reason": skip_reason,
            "original_text": original_text,
            "translated_text": translated_text,
            "text_color_hex": text_color_hex,
            "bg_color_hex": bg_color_hex,
            "bg_type": bg_type,
            "font_family": font_family,
            "font_size_px": font_size_px,
            "recommended_font_size_px": recommended_font_size_px,
            "is_bold": is_bold,
            "alignment": alignment,
            "block_angle": block_angle,
            "overflow_strategy": overflow,
            "line_break_hints": valid_hints,
        }
        paragraphs_out.append(entry)

        total_prompt_tokens += n_prompt
        total_completion_tokens += n_completion
        if pid in per_pid_vlm:
            per_paragraph_profile.append({
                "pid": pid,
                "crop_wh": [info.get("crop_w", 0), info.get("crop_h", 0)],
                "crop_pixels": int(info.get("crop_w", 0)) * int(info.get("crop_h", 0)),
                "prompt_tokens": n_prompt,
                "completion_tokens": n_completion,
            })

    num_paragraphs = len(paragraphs_out)
    num_vlm = len(pid_order)
    avg_sec = round(decode_wall / num_vlm, 3) if num_vlm else 0.0

    # Top-level global_info: use the mode of block_angle across all paragraphs as
    # the dominant angle of the whole image
    all_block_angles = [p["block_angle"] for p in paragraphs_out if p.get("block_angle") is not None]
    global_angle = _dominant_angle(all_block_angles) if all_block_angles else 0

    result: dict[str, Any] = {
        "stem": stem,
        "image_path": str(image_path),
        "ocr_json_path": str(ocr_json_path),
        "image_size": [img_w, img_h],
        "ocr_size": ocr_size,
        "target_lang": target_lang,
        "global_info": {"global_angle": global_angle},
        "num_paragraphs": num_paragraphs,
        "vlm_batch_wall_sec": round(decode_wall, 3),
        "vlm_avg_sec_per_paragraph": avg_sec,
        "bbox_visualization": vis_path_str,
        "paragraphs": paragraphs_out,
    }

    out_json = output_dir / f"{stem}__vlm.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    profile = {
        "image_stem": stem,
        "image_size": [img_w, img_h],
        "num_paragraphs": num_vlm,
        "batch_wall_sec": round(decode_wall, 3),
        "tokens": {
            "total_prompt": total_prompt_tokens,
            "total_completion": total_completion_tokens,
            "avg_prompt_per_seq": round(total_prompt_tokens / num_vlm, 1) if num_vlm else 0.0,
            "avg_completion_per_seq": round(total_completion_tokens / num_vlm, 1) if num_vlm else 0.0,
        },
        "latency": {
            "decode_wall_sec": round(decode_wall, 3),
            "avg_wall_per_paragraph_sec": avg_sec,
            "decode_throughput_tok_s": (
                int(total_completion_tokens / decode_wall) if decode_wall > 0.001 else 0
            ),
        },
        "per_paragraph": per_paragraph_profile,
    }
    (output_dir / f"{stem}__profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    result["profile"] = profile

    total_wall = time.perf_counter() - t_start
    io_wall = total_wall - prep_wall - crop_wall - decode_wall
    maxed_tag = (
        f" MAXED_OUT_PIDS={[p for p, _ in maxed]}" if maxed else ""
    )
    print(
        f"[{stem}] total={total_wall:5.2f}s | decode={decode_wall:5.2f}s "
        f"(median/max comp={median_comp}/{max_comp}) | "
        f"prep={prep_wall*1000:4.0f}ms crop={crop_wall*1000:4.0f}ms io={io_wall*1000:4.0f}ms | "
        f"paragraphs total={len(groups)} prefilter={len(prefilter_reasons)} "
        f"vlm={len(pid_order)}{maxed_tag}"
    )
    return result


# =============================================================================
#  Batch mode: OCR dir + image dir
# =============================================================================

def _find_image_for_ocr(
    ocr_json: Path, image_dir: Path,
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Path | None:
    """The stem of an OCR JSON is usually xxx_ocr or xxx; strip _ocr and look up the image in the image directory."""
    raw_stem = ocr_json.stem
    candidates = {raw_stem}
    if raw_stem.endswith("_ocr"):
        candidates.add(raw_stem[:-4])
    for stem in candidates:
        for ext in image_exts:
            p = image_dir / f"{stem}{ext}"
            if p.is_file():
                return p
    return None


def list_ocr_jsons(ocr_dir: Path, limit: int | None) -> list[Path]:
    files = sorted([p for p in ocr_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    return files[:limit] if limit else files


# =============================================================================
#  Main entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OCR 段落切图 + Qwen3-VL 批量翻译 (vLLM continuous batching)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Single image or batch (choose one)
    p.add_argument("--ocr-json", type=str, default=None, help="单张: OCR JSON 路径")
    p.add_argument("--image", type=str, default=None, help="单张: 原图路径")
    p.add_argument("--ocr-dir", type=str, default=None, help="批量: OCR JSON 目录")
    p.add_argument("--image-dir", type=str, default=None, help="批量: 原图目录")

    p.add_argument("--output-dir", type=str, required=True, help="输出目录 (可视化 / crops / JSON)")
    p.add_argument("--model-path",
                   default="/path/to/models/Qwen3-VL-8B-Instruct",
                   help="Qwen3-VL-8B-Instruct 权重目录 (non-thinking 变体)")
    p.add_argument("--target-lang", default="中文", help="目标语言")
    p.add_argument("--crop-pad", type=int, default=6, help="段落裁剪向外扩展的像素")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="每段落生成 token 上限。默认 512。"
                        "注意 VLM 同时吐 orig_text + translated_text + 其余字段, "
                        "越南/泰/阿拉伯等长 token 语种, 192 会被截断 -> vlm_parse_error")
    p.add_argument("--limit", type=int, default=None, help="批量模式: 最多处理张数")

    # Visualization / cropping switches
    p.add_argument("--visualize", action="store_true",
                   help="写 <stem>__bbox.jpg (红框标 paragraph + item bbox)。默认关闭以加速")
    p.add_argument("--no-save-crops", action="store_true",
                   help="不落盘裁剪图, 直接在内存传 PIL 给 VLM (加速 I/O; 但丢失 crop_path 调试能力)")

    # vLLM performance knobs
    p.add_argument("--dtype", default="fp16", choices=["auto", "bf16", "fp16", "fp32"],
                   help="默认 fp16 (速度最快, Qwen3-VL 表现稳定)")
    p.add_argument("--vllm-gpu-memory-utilization", type=float,
                   default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95")))
    p.add_argument("--vllm-tensor-parallel-size", type=int,
                   default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--vllm-max-model-len", type=int, default=16384)
    p.add_argument("--vllm-max-pixels", type=int, default=256 * 28 * 28,
                   help="vLLM 端图像像素上限 (影响 vision tokens 数)。默认 200704 = 256·28·28")
    p.add_argument("--vllm-min-pixels", type=int, default=None)
    p.add_argument("--vllm-max-num-seqs", type=int, default=256,
                   help="vLLM continuous batching 并发上限。默认 256")
    p.add_argument("--vllm-enforce-eager", action="store_true")
    p.add_argument("--vllm-disable-prefix-caching", action="store_true")
    p.add_argument("--no-warmup", action="store_true",
                   help="跳过首张预热 (默认预热, 避免第 1 张巨慢)")
    return p.parse_args()


def _build_vllm(args: argparse.Namespace):
    _warn_if_vllm_without_cuda()

    kind, arch, model_type = detect_model_kind(args.model_path)
    if kind != "vl":
        raise SystemExit(
            f"--model-path 非视觉语言模型 (kind={kind} arch={arch}); "
            "请提供 Qwen3-VL / Qwen3.5-VL 等 VLM 目录"
        )

    mm_kw: dict[str, Any] = {}
    if args.vllm_max_pixels is not None:
        mm_kw["max_pixels"] = int(args.vllm_max_pixels)
    if args.vllm_min_pixels is not None:
        mm_kw["min_pixels"] = int(args.vllm_min_pixels)

    extra: dict[str, Any] = {"enable_prefix_caching": not args.vllm_disable_prefix_caching}
    if mm_kw:
        extra["mm_processor_kwargs"] = mm_kw
    if args.vllm_max_num_seqs is not None:
        extra["max_num_seqs"] = int(args.vllm_max_num_seqs)

    llm, processor, _ = _load_vllm_vl_engine(
        args.model_path,
        dtype=args.dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=args.vllm_enforce_eager,
        extra_engine_kwargs=extra,
    )
    return llm, processor


def _warmup_vllm(llm: Any, processor: Any, target_lang: str,
                 max_new_tokens: int = 16) -> None:
    """Send one dummy crop to trigger CUDA graph capture / weight loading.

    Otherwise the first real image would pay an extra 3-8s of first-call overhead.
    """
    try:
        from PIL import Image as PILImage
        from vllm import SamplingParams
    except ImportError:
        return
    dummy = PILImage.new("RGB", (224, 224), color=(250, 250, 250))
    system_prompt = SYSTEM_PROMPT.format(target_lang=target_lang)
    user_prompt = USER_PROMPT_TEMPLATE.format(target_lang=target_lang)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": "__PLACEHOLDER__"},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    prompt_str = _apply_chat_template_no_think(
        processor, messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    t0 = time.perf_counter()
    llm.generate(
        [{"prompt": prompt_str, "multi_modal_data": {"image": dummy}}],
        sampling_params=sp, use_tqdm=False,
    )
    print(f"[warmup] vLLM first-call warmup: {time.perf_counter() - t0:.2f}s")


def main() -> None:
    args = _parse_args()

    is_single = args.ocr_json and args.image
    is_batch = args.ocr_dir and args.image_dir
    if not is_single and not is_batch:
        raise SystemExit(
            "请二选一：\n"
            "  (A) --ocr-json + --image  单张模式\n"
            "  (B) --ocr-dir  + --image-dir  批量模式"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path]] = []
    if is_single:
        ocr_json = Path(args.ocr_json).resolve()
        image = Path(args.image).resolve()
        if not ocr_json.is_file():
            raise SystemExit(f"OCR JSON 不存在: {ocr_json}")
        if not image.is_file():
            raise SystemExit(f"原图不存在: {image}")
        jobs.append((ocr_json, image))
    else:
        ocr_dir = Path(args.ocr_dir).resolve()
        image_dir = Path(args.image_dir).resolve()
        for oj in list_ocr_jsons(ocr_dir, args.limit):
            img = _find_image_for_ocr(oj, image_dir)
            if img is None:
                continue
            jobs.append((oj, img))
        if not jobs:
            raise SystemExit(f"{ocr_dir} 下没有找到可处理的 OCR JSON (或对应图片缺失)")

    llm, processor = _build_vllm(args)

    if not args.no_warmup:
        _warmup_vllm(llm, processor, target_lang=args.target_lang)

    try:
        overall_t0 = time.perf_counter()
        for idx, (ocr_json, image) in enumerate(jobs, 1):
            print(f"\n=== [{idx}/{len(jobs)}] {image.name} ===")
            try:
                process_one(
                    ocr_json_path=ocr_json,
                    image_path=image,
                    output_dir=output_dir,
                    llm=llm,
                    processor=processor,
                    target_lang=args.target_lang,
                    max_new_tokens=args.max_new_tokens,
                    crop_pad=args.crop_pad,
                    save_visualize=args.visualize,
                    save_crops=not args.no_save_crops,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] {image.name}: {e!r}")
        total = time.perf_counter() - overall_t0
        print(f"\n[DONE] processed {len(jobs)} image(s) in {total:.2f}s "
              f"(avg {total / max(1, len(jobs)):.2f}s/image)")
    finally:
        _cleanup(llm, processor)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
