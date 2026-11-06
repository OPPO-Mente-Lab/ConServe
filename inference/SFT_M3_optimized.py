#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""
SFT_M3_optimized.py
===================
One-image multi-box SFT data generation version with OCR paragraph numbering
(vLLM continuous batching).

Compared to the two-stage version in the same directory,
``step1_ocr_paragraph_vlm_ocr+vlm+bbx_2steps_multi-box.py``, this script:

  1. Sorts the to-translate regions by the visual position of their boxes and
     splits them into batches of size m
  2. Renders "thick orange frame + red #pid label outside the frame" for each
     batch, and writes the pid order into the prompt
  3. Outputs a JSON array in **one** vLLM forward pass; each box always contains
     should_translate / original_text / translated_text
  4. vLLM continuous batching: the prompts of all batches are submitted in a
     single ``llm.generate()`` call

Usage:
  python step1_ocr_paragraph_vlm_ocr+vlm+bbx_multi-box.py \\
      --ocr-dir   <dir> \\
      --image-dir <dir> \\
      --output-dir <dir> \\
      --m 3 \\
      [--target-lang Chinese] [--per-box-max-new-tokens 512] [--max-new-tokens 4096]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from m3_pid_labels import infer_paragraph_orientation, render_pid_framed_image

_THIS_DIR = Path(__file__).resolve().parent
_DEPENDENCY_DIR = _THIS_DIR
_CANDIDATE_DEPENDENCY_DIRS = [
    _THIS_DIR,
    Path(os.environ.get("CONSERVE_HELPER_DIR", _THIS_DIR)),
]
for _cand in _CANDIDATE_DEPENDENCY_DIRS:
    if (_cand / "step1_ocr_paragraph_vlm_0506_orig.py").is_file():
        _DEPENDENCY_DIR = _cand
        break
for _path in (_THIS_DIR, _DEPENDENCY_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from step1_ocr_paragraph_vlm_0506_orig import (  # noqa: E402
    _dominant_angle,
    _infer_overflow_strategy,
    _map_rule_to_skip_reason,
    classify_no_translate,
    draw_bbox_visualization,
    group_by_paragraph,
)
from vlm_sp_up_record import (  # noqa: E402
    build_token_rollups,
    build_vlm_token_cost,
    vlm_prompts_record,
)

# ---------------------------------------------------------------------------
#  Load the base bbx script via importlib (LIGHT_IMPORT=1 avoids pulling in vLLM)
# ---------------------------------------------------------------------------
_BBX_MOD: Any | None = None


def _load_bbx_mod() -> Any:
    global _BBX_MOD
    if _BBX_MOD is not None:
        return _BBX_MOD
    bbx_path = _DEPENDENCY_DIR / "step1_ocr_paragraph_vlm_ocr+vlm+bbx_ocr-vllm-generate.py"
    if not bbx_path.is_file():
        raise FileNotFoundError(f"缺少基础 bbx 脚本: {bbx_path}")
    spec = importlib.util.spec_from_file_location("_bbx_stub", bbx_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法解析模块: {bbx_path}")
    mod = importlib.util.module_from_spec(spec)
    prev = os.environ.get("STEP1_OCR_PARAGRAPH_LIGHT_IMPORT")
    os.environ["STEP1_OCR_PARAGRAPH_LIGHT_IMPORT"] = "1"
    try:
        spec.loader.exec_module(mod)
    finally:
        if prev is None:
            os.environ.pop("STEP1_OCR_PARAGRAPH_LIGHT_IMPORT", None)
        else:
            os.environ["STEP1_OCR_PARAGRAPH_LIGHT_IMPORT"] = prev
    _BBX_MOD = mod
    return mod


# vLLM / processor
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VLM_TRANSLATE = _REPO_ROOT / "VLM-translate"
if _VLM_TRANSLATE.is_dir():
    sys.path.insert(0, str(_VLM_TRANSLATE))

from batch_photo_translate_vlm import (  # noqa: E402
    _apply_chat_template_no_think,
    _cleanup,
    _load_vllm_vl_engine,
    _strip_thinking,
    _warn_if_vllm_without_cuda,
    detect_model_kind,
)

# Keep the shared Qwen chat-template path untouched. InternVL3's local
# template expects message content to be a string, while the Qwen path uses
# structured image/text content lists.
_SHARED_APPLY_CHAT_TEMPLATE = _apply_chat_template_no_think


def _is_internvl_processor(proc_or_tok: Any) -> bool:
    candidates = [
        getattr(proc_or_tok, "name_or_path", ""),
        getattr(getattr(proc_or_tok, "tokenizer", None), "name_or_path", ""),
        type(proc_or_tok).__name__,
    ]
    return any("internvl" in str(value).lower() for value in candidates)


def _apply_chat_template_no_think(proc_or_tok: Any, messages: Any, **kwargs: Any) -> Any:
    # Callers in this runner sometimes pass enable_thinking explicitly, while
    # the shared helper also supplies it. Remove the duplicate keyword first.
    kwargs.pop("enable_thinking", None)
    if not _is_internvl_processor(proc_or_tok):
        return _SHARED_APPLY_CHAT_TEMPLATE(proc_or_tok, messages, **kwargs)

    normalized = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image":
                    parts.append("<image>")
                elif part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            content = "\n".join(parts)
        normalized.append({"role": message.get("role", "user"), "content": content})
    return proc_or_tok.apply_chat_template(normalized, **kwargs)


# ---------------------------------------------------------------------------
#  Multi-box rendering (full image + multiple orange highlighted paragraphs +
#  #pid labels outside the frames)
# ---------------------------------------------------------------------------
def _render_framed_full_image_multi(
    image_bgr: np.ndarray,
    pids_in_order: list[int],
    padded_by_pid: dict[int, tuple[int, int, int, int]],
    orient_by_pid: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw the orange main frames and red ``#pid`` labels placed just outside the frames so they never occlude the text inside."""
    return render_pid_framed_image(
        image_bgr,
        pids_in_order,
        padded_by_pid,
        orient_by_pid,
    )


def frame_multi_paragraphs_on_full_image(
    image_bgr: np.ndarray,
    groups: dict[int, dict[str, Any]],
    crop_dir: Path,
    stem: str,
    batches: list[list[int]],
    pad: int = 6,
    save_to_disk: bool = True,
) -> dict[int, dict[str, Any]]:
    """For each batch, draw the orange to-translate frames and the ``#pid`` labels outside them on the full image.

    Returns: {batch_idx: {
              crop_path, crop_pil, pids_in_order: [pid, ...],
              padded_bbox_by_pid: {pid: [x1,y1,x2,y2]},
              crop_w, crop_h,
          }}
    """
    bbx = _load_bbx_mod()
    if save_to_disk:
        crop_dir.mkdir(parents=True, exist_ok=True)
    h, w = image_bgr.shape[:2]

    padded_by_pid: dict[int, tuple[int, int, int, int]] = {}
    orient_by_pid: dict[int, str] = {}
    for pid, g in groups.items():
        bbox = g.get("paragraph_bbox")
        if bbox is None:
            continue
        pb = bbx._padded_paragraph_bbox(image_bgr, bbox, pad)
        if pb is not None:
            padded_by_pid[pid] = pb
            orient_by_pid[pid] = infer_paragraph_orientation(g)

    out: dict[int, dict[str, Any]] = {}
    for b_idx, batch_pids in enumerate(batches):
        valid_pids = [pid for pid in batch_pids if pid in padded_by_pid]
        if not valid_pids:
            continue
        framed = _render_framed_full_image_multi(
            image_bgr,
            valid_pids,
            padded_by_pid,
            orient_by_pid,
        )
        crop_path: str | None = None
        crop_pil = None
        if save_to_disk:
            crop_path = str(crop_dir / f"{stem}__batch{b_idx:03d}__framed.jpg")
            cv2.imwrite(crop_path, framed)
        else:
            crop_pil = PILImage.fromarray(cv2.cvtColor(framed, cv2.COLOR_BGR2RGB))
        out[b_idx] = {
            "crop_path": crop_path,
            "crop_pil": crop_pil,
            "pids_in_order": valid_pids,
            "padded_bbox_by_pid": {p: list(padded_by_pid[p]) for p in valid_pids},
            "crop_w": w,
            "crop_h": h,
        }
    return out


# ---------------------------------------------------------------------------
# Three-field prompt shared by SFT / inference (kept consistent with build_sft_m3_datapairs.py)
SYSTEM_PROMPT_MULTI = """图片多区域文字翻译任务。输入是完整截图，图中有 {m} 个橙色粗框；每个框外紧邻一个红色 #PID 标签，框与标签共同标识一个独立处理单元，框外内容仅供理解语境。
按用户给出的 #PID 顺序逐框识别文字并判断是否翻译。清晰可读、非{target_lang}的自然语言必须翻译；已是{target_lang}、纯数字符号、URL或邮箱、代码标识符、单字符、商标Logo、人名或模糊残缺文字不翻译。
只输出严格单行JSON数组，长度必须为 {m}，数组元素与用户给出的 #PID 顺序一一对应；不得遗漏、合并、拆分或交换。
每个数组元素固定且只能包含以下三个字段，字段顺序固定:
需要翻译:{{"should_translate":true,"original_text":"框内原文","translated_text":"忠实完整译文"}}
不需翻译:{{"should_translate":false,"original_text":"框内可辨文字","translated_text":""}}
original_text只填写对应橙框内的可见文字，不得混入标签、相邻框或框外内容；无法辨认时为空字符串。
translated_text仅在should_translate=true时填写忠实、完整、简洁的{target_lang}译文，不得省略、扩写或夹带框外内容；false时必须为空字符串。"""

# Teacher condition (OCR context injection): on top of the student SP, reinforce
# "context must be used; literal translation is forbidden"
SYSTEM_PROMPT_MULTI_TEACHER_CTX = """图片多区域文字翻译任务。输入是完整截图，图中有 {m} 个橙色粗框；每个框外紧邻一个红色 #PID 标签，框与标签共同标识一个独立处理单元。
用户消息中会额外给出【语境参考】（图中橙框以外的段落）。你必须把这些语境与整图画面一起用于理解每个目标框的领域、场景、指代与省略，再决定译法。
核心翻译原则（必须遵守）：
1. 必须结合语境翻译：一词多义、缩写、残缺短语、跨框拆分、专业术语，一律优先采用与同屏语境一致的含义；不得脱离语境按字典默认义硬翻。
2. 禁止直译/字面死译：不要逐词对译、不要保留源语语序硬套{target_lang}、不要在已能由语境确定专业义项时仍输出通用义项。
3. 语境只用于理解与消歧，禁止把语境段落本身写入 original_text 或 translated_text；不得翻译、复述或拼接语境文字到答案里。
4. 译文应自然、完整、符合该界面/文档的用语习惯，在 should_translate=true 时输出简洁准确的{target_lang}。
按用户给出的 #PID 顺序逐框识别文字并判断是否翻译。清晰可读、非{target_lang}的自然语言必须翻译；已是{target_lang}、纯数字符号、URL或邮箱、代码标识符、单字符、商标Logo、人名或模糊残缺文字不翻译。
只输出严格单行JSON数组，长度必须为 {m}，数组元素与用户给出的 #PID 顺序一一对应；不得遗漏、合并、拆分或交换。
每个数组元素固定且只能包含以下三个字段，字段顺序固定:
需要翻译:{{"should_translate":true,"original_text":"框内原文","translated_text":"结合语境的{target_lang}译文"}}
不需翻译:{{"should_translate":false,"original_text":"框内可辨文字","translated_text":""}}
original_text只填写对应橙框内的可见文字，不得混入标签、相邻框、语境参考或框外内容；无法辨认时为空字符串。
translated_text仅在should_translate=true时填写；false时必须为空字符串。"""

# Teacher condition (scene-summary injection, not OCR): the SP only states the
# rules; the concrete summary goes into the UP
SYSTEM_PROMPT_MULTI_TEACHER_SCENE = """图片多区域文字翻译任务。输入是完整截图，图中有 {m} 个橙色粗框；每个框外紧邻一个红色 #PID 标签，框与标签共同标识一个独立处理单元。
用户消息中会额外给出【场景语境】：对本图界面/文档类型、功能与主题的文字概括。你必须结合该概括与整图画面理解每个目标框的领域与词义，再决定译法。
核心翻译原则（必须遵守）：
1. 必须结合【场景语境】消歧：一词多义、缩写、残缺短语、专业术语，优先采用与场景概括一致的含义；不得脱离场景按字典默认义硬翻。
2. 禁止直译/字面死译：不要逐词对译、不要在场景已能确定专业/功能义项时仍输出通用义项。
3. 【场景语境】只供理解，禁止把概括原文写入 original_text 或 translated_text；不得复述或拼接概括内容到答案里。
4. 译文应自然、完整、符合该界面/文档用语习惯。
按用户给出的 #PID 顺序逐框识别文字并判断是否翻译。清晰可读、非{target_lang}的自然语言必须翻译；已是{target_lang}、纯数字符号、URL或邮箱、代码标识符、单字符、商标Logo、人名或模糊残缺文字不翻译。
只输出严格单行JSON数组，长度必须为 {m}，数组元素与用户给出的 #PID 顺序一一对应；不得遗漏、合并、拆分或交换。
每个数组元素固定且只能包含以下三个字段，字段顺序固定:
需要翻译:{{"should_translate":true,"original_text":"框内原文","translated_text":"结合场景的{target_lang}译文"}}
不需翻译:{{"should_translate":false,"original_text":"框内可辨文字","translated_text":""}}
original_text只填写对应橙框内的可见文字，不得混入标签、相邻框、场景概括或框外内容；无法辨认时为空字符串。
translated_text仅在should_translate=true时填写；false时必须为空字符串。"""


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


def _build_user_prompt_multi_teacher_ctx(target_lang: str, pids: list[int] | None = None) -> str:
    """Teacher UP: on top of the student UP, require using context and forbid literal translation."""
    base = _build_user_prompt_multi(target_lang, pids)
    return (
        f"{base}"
        f"翻译前必须先阅读后文【语境参考】与整图，结合同屏领域/场景/相邻表述确定每个目标框的含义；"
        f"禁止脱离语境的直译、字面死译或通用义项硬翻；"
        f"语境文字本身不要输出、不要翻译。"
    )


def _build_user_prompt_multi_teacher_scene(target_lang: str, pids: list[int] | None = None) -> str:
    """Teacher UP (scene summary): require reading the 【scene context】 before translating."""
    base = _build_user_prompt_multi(target_lang, pids)
    return (
        f"{base}"
        f"翻译前必须先阅读后文【场景语境】与整图，按界面/文档类型消歧；"
        f"禁止脱离场景的直译或通用义项硬翻；场景概括本身不要输出、不要翻译。"
    )


def _append_scene_summary_block(
    base_user: str, scene_summary: str, target_pids: list[int],
) -> str:
    summary = " ".join((scene_summary or "").split()).strip()
    if not summary:
        return base_user
    order = " → ".join(f"#{pid}" for pid in target_pids)
    return (
        f"{base_user.rstrip()}\n\n"
        f"【场景语境｜必须结合理解，禁止输出】\n"
        f"{summary}\n"
        f"要求：结合上述场景概括翻译目标橙框；禁止直译；不要把概括内容写入答案；"
        f"输出顺序仍为 {order}。"
    )


def _load_scene_summary_map(path: Path | None) -> dict[str, str]:
    """stem -> summary. Accepts {stem: {summary: ...}} or {stem: "..."}."""
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"--scene-summary-json 必须是 dict，收到 {type(raw)}")
    out: dict[str, str] = {}
    for k, v in raw.items():
        stem = Path(str(k)).stem
        if isinstance(v, dict):
            text = str(v.get("summary") or v.get("caption") or v.get("text") or "")
        else:
            text = str(v or "")
        text = " ".join(text.split()).strip()
        if text:
            out[stem] = text
    return out


def _flatten_ocr_text(text: str) -> str:
    return " ".join((text or "").split())


def _select_global_context_paras(
    ocr_by_pid: dict[int, str],
    target_pids: list[int],
    ctx_n: int,
    *,
    max_para_chars: int | None = None,
    max_total_chars: int | None = None,
) -> list[tuple[int, str]]:
    """Select global context paragraphs (excluding target PIDs).

    ``ctx_n==0`` selects none; ``>0`` takes the n paragraphs nearest to the
    targets; ``<0`` takes all non-target paragraphs.
    Returns ``[(pid, text), ...]`` ordered by reading order.

    Default budgets: 200 per paragraph / 1500 total for the nearest-n mode;
    500 per paragraph / 8000 total for the full-text mode (ctx_n<0).
    """
    if ctx_n == 0 or not ocr_by_pid:
        return []
    if max_para_chars is None:
        max_para_chars = 500 if ctx_n < 0 else 200
    if max_total_chars is None:
        max_total_chars = 8000 if ctx_n < 0 else 1500
    tset = {int(x) for x in target_pids}
    cands: list[tuple[int, int, str]] = []  # dist, pid, text
    for pid, text in ocr_by_pid.items():
        if int(pid) in tset:
            continue
        flat = _flatten_ocr_text(text)
        if not flat:
            continue
        dist = min(abs(int(pid) - t) for t in tset) if tset else int(pid)
        cands.append((dist, int(pid), flat))
    if ctx_n > 0:
        cands.sort(key=lambda x: (x[0], x[1]))
        picked = cands[:ctx_n]
    else:
        picked = cands
    picked = sorted(picked, key=lambda x: x[1])
    out: list[tuple[int, str]] = []
    total = 0
    for _, pid, text in picked:
        if len(text) > max_para_chars:
            text = text[:max_para_chars] + "…"
        if total + len(text) > max_total_chars:
            break
        out.append((pid, text))
        total += len(text)
    return out


def _append_context_block(base_user: str, ctx_paras: list[tuple[int, str]], target_pids: list[int]) -> str:
    if not ctx_paras:
        return base_user
    order = " → ".join(f"#{pid}" for pid in target_pids)
    lines = [
        base_user.rstrip(),
        "",
        "【语境参考｜必须结合理解，禁止输出】",
        "下面是本图中橙框以外段落的原文（原始 OCR），按阅读顺序给出。",
        "你必须结合这些语境翻译目标框：优先采用与同屏语境一致的专业/场景义项；禁止直译、字面死译、脱离语境的通用义项。",
        "语境文字本身不要输出、不要翻译，也不得写入 original_text / translated_text：",
    ]
    for pid, text in ctx_paras:
        lines.append(f"#{pid} {text}")
    lines.append(
        f"再次强调：结合语境翻译目标橙框；禁止直译；不要输出语境内容；"
        f"输出顺序仍为 {order}。"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  JSON array parsing (with truncation recovery)
# ---------------------------------------------------------------------------
def _extract_top_level_objects(text: str) -> list[dict]:
    """From a (usually truncated) JSON array body, balance braces character by character and extract the already closed objects."""
    out: list[dict] = []
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    obj_str = text[start:i + 1]
                    try:
                        obj = json.loads(obj_str)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
            else:
                break
    return out


def _safe_parse_json_array(
    raw: str, expected_n: int,
) -> tuple[list[dict], bool]:
    """Parse a JSON array; returns (list of fixed length = expected_n, truncation_recovered_flag).

    - When the element count does not match the box count, the whole batch is
      reported as errors to avoid misaligned supervision
    - Tolerates markdown fences and surrounding junk text
    - On truncation: after the opening ``[``, extract objects by balancing braces
      per object; closed objects count as recovered
    - When N=1, a single object ``{...}`` is accepted as a fallback
    """
    if not raw:
        return [
            {"_parse_error": True, "_raw": ""} for _ in range(expected_n)
        ], False

    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    arr: list[Any] | None = None
    truncated = False

    try:
        v = json.loads(s)
        if isinstance(v, list):
            arr = v
        elif isinstance(v, dict) and expected_n == 1:
            arr = [v]
    except json.JSONDecodeError:
        pass

    if arr is None:
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if m:
            try:
                v = json.loads(m.group(0))
                if isinstance(v, list):
                    arr = v
            except json.JSONDecodeError:
                pass

    if arr is None:
        idx = s.find("[")
        if idx >= 0:
            objs = _extract_top_level_objects(s[idx + 1:])
            if objs:
                arr = objs
                truncated = True

    if arr is None:
        try:
            bbx = _load_bbx_mod()
            obj = bbx._safe_parse_json(s)
            if isinstance(obj, dict) and "_parse_error" not in obj:
                arr = [obj]
                if expected_n > 1:
                    truncated = True
        except Exception:
            pass

    if arr is None:
        return [
            {"_parse_error": True, "_raw": s[:500]} for _ in range(expected_n)
        ], False

    if len(arr) != expected_n:
        error = f"box_count_mismatch:{len(arr)}!={expected_n}"
        return [
            {"_parse_error": True, "_raw": error} for _ in range(expected_n)
        ], truncated

    out: list[dict] = []
    for i in range(expected_n):
        if i < len(arr) and isinstance(arr[i], dict):
            out.append(arr[i])
        else:
            out.append({"_parse_error": True, "_raw": "missing_in_array"})
    return out, truncated


def _finalize_translation_object(obj: Any) -> dict[str, Any]:
    """Validate the fixed three-field SFT output of a single box; the legacy vision schema is no longer applied."""
    if not isinstance(obj, dict):
        return {"_parse_error": True, "_raw": "not_an_object"}
    if obj.get("_parse_error"):
        return obj

    expected_keys = {"should_translate", "original_text", "translated_text"}
    if set(obj) != expected_keys:
        missing = sorted(expected_keys - set(obj))
        extra = sorted(set(obj) - expected_keys)
        return {
            "_parse_error": True,
            "_raw": f"schema_fields:missing={missing},extra={extra}",
        }

    should_translate = obj.get("should_translate")
    original = obj.get("original_text")
    translated = obj.get("translated_text")
    if not isinstance(should_translate, bool):
        return {
            "_parse_error": True,
            "_raw": "should_translate_must_be_boolean",
        }
    if not isinstance(original, str) or not isinstance(translated, str):
        return {
            "_parse_error": True,
            "_raw": "missing_or_non_string_translation_field",
        }
    if should_translate and (not original.strip() or not translated.strip()):
        return {"_parse_error": True, "_raw": "empty_positive_translation_field"}
    if not should_translate and translated != "":
        return {"_parse_error": True, "_raw": "false_translation_must_be_empty"}

    return {
        "should_translate": should_translate,
        "original_text": original,
        "translated_text": translated,
    }


# ---------------------------------------------------------------------------
#  vLLM batch construction and inference
# ---------------------------------------------------------------------------
def build_vlm_inputs_multi(
    processor: Any,
    batch_images: list[tuple[int, PILImage.Image, list[int]]],
    target_lang: str,
    *,
    ocr_by_pid: dict[int, str] | None = None,
    ctx_n: int = 0,
    scene_summary: str | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, list[int]]], list[dict[str, Any]]]:
    """Build vLLM inference inputs (prompt string + PIL image) for each batch.

    batch_images: [(batch_idx, pil_image, pids_in_order), ...]
    Returns: (inputs, order, prompt_meta)

    Priority: ``scene_summary`` (scene-summary teacher) > ``ctx_n!=0`` (OCR teacher) > student.
    """
    inputs: list[dict[str, Any]] = []
    order: list[tuple[int, list[int]]] = []
    prompt_meta: list[dict[str, Any]] = []
    scene_summary = (scene_summary or "").strip() or None
    for b_idx, pil, pids in batch_images:
        ctx_paras: list[tuple[int, str]] = []
        teacher_mode = "none"
        if scene_summary:
            teacher_mode = "scene"
            sp = SYSTEM_PROMPT_MULTI_TEACHER_SCENE.format(
                target_lang=target_lang, m=len(pids),
            )
            up = _build_user_prompt_multi_teacher_scene(target_lang, pids)
            up = _append_scene_summary_block(up, scene_summary, pids)
        elif ctx_n != 0 and ocr_by_pid:
            teacher_mode = "ocr"
            sp = SYSTEM_PROMPT_MULTI_TEACHER_CTX.format(
                target_lang=target_lang, m=len(pids),
            )
            up = _build_user_prompt_multi_teacher_ctx(target_lang, pids)
            ctx_paras = _select_global_context_paras(ocr_by_pid or {}, pids, ctx_n)
            up = _append_context_block(up, ctx_paras, pids)
        else:
            sp = SYSTEM_PROMPT_MULTI.format(target_lang=target_lang, m=len(pids))
            up = _build_user_prompt_multi(target_lang, pids)
        msgs = [
            {"role": "system", "content": sp},
            {"role": "user", "content": [
                {"type": "image", "image": "__PLACEHOLDER__"},
                {"type": "text", "text": up},
            ]},
        ]
        prompt_str = _apply_chat_template_no_think(
            processor, msgs, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs.append({"prompt": prompt_str, "multi_modal_data": {"image": pil}})
        order.append((b_idx, pids))
        prompt_meta.append({
            "batch_idx": b_idx,
            "pids": list(pids),
            "teacher_mode": teacher_mode,
            "teacher_ctx": teacher_mode != "none",
            "ctx_n": ctx_n if teacher_mode == "ocr" else 0,
            "ctx_pids": [p for p, _ in ctx_paras],
            "ctx_chars": sum(len(t) for _, t in ctx_paras),
            "scene_summary": scene_summary if teacher_mode == "scene" else None,
            "scene_chars": len(scene_summary) if scene_summary else 0,
            "system_prompt": sp,
            "user_prompt": up,
        })
    return inputs, order, prompt_meta


# Array terminator for the three-field schema.
_STOP_STRINGS: list[str] = ['}]']


def _make_sampling_params(
    max_tokens: int,
    *,
    temperature: float,
    num_samples: int,
) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        n=num_samples,
        stop=_STOP_STRINGS,
        include_stop_str_in_output=True,
    )


def run_vlm_batches_multi(
    llm: Any,
    inputs: list[dict[str, Any]],
    n_boxes_list: list[int],
    per_box_budget: int,
    overall_cap: int,
    *,
    temperature: float = 0.0,
    num_samples: int = 1,
) -> tuple[list[dict[str, Any]], float]:
    """Submit all batches in one llm.generate() call; returns (results, wall_time).

    Each result has the form::
        {"samples": [{"raw_text", "n_prompt", "n_completion"}, ...]}
    """
    if not inputs:
        return [], 0.0

    caps = [min(overall_cap, max(per_box_budget, per_box_budget * n)) for n in n_boxes_list]
    t0 = time.perf_counter()
    if len(set(caps)) == 1:
        sp = _make_sampling_params(
            caps[0], temperature=temperature, num_samples=num_samples,
        )
        outs = llm.generate(inputs, sampling_params=sp, use_tqdm=False)
    else:
        sps = [
            _make_sampling_params(
                c, temperature=temperature, num_samples=num_samples,
            )
            for c in caps
        ]
        outs = llm.generate(inputs, sampling_params=sps, use_tqdm=False)
    wall = time.perf_counter() - t0

    results: list[dict[str, Any]] = []
    for ro in outs:
        try:
            n_p = len(ro.prompt_token_ids or [])
        except AttributeError:
            n_p = 0
        sample_rows: list[dict[str, Any]] = []
        for out in ro.outputs:
            raw = (out.text or "").strip()
            text = _strip_thinking(raw)
            try:
                n_c = len(out.token_ids or [])
            except AttributeError:
                n_c = 0
            sample_rows.append({
                "raw_text": text,
                "n_prompt": n_p,
                "n_completion": n_c,
            })
        if not sample_rows:
            sample_rows.append({"raw_text": "", "n_prompt": n_p, "n_completion": 0})
        row: dict[str, Any] = {"samples": sample_rows}
        # With a single sample, keep the legacy fields for compatibility with existing logic.
        if len(sample_rows) == 1:
            row.update(sample_rows[0])
        results.append(row)
    return results, wall


def _assign_per_pid_from_batch_results(
    order: list[tuple[int, list[int]]],
    vlm_results: list[dict[str, Any]],
    sample_idx: int,
) -> dict[int, dict[str, Any]]:
    per_pid_vlm: dict[int, dict[str, Any]] = {}
    for i, (_, pids_in_order) in enumerate(order):
        r = (
            vlm_results[i]
            if i < len(vlm_results)
            else {"samples": [{"raw_text": "", "n_prompt": 0, "n_completion": 0}]}
        )
        samples = r.get("samples") or [{"raw_text": "", "n_prompt": 0, "n_completion": 0}]
        sample_r = (
            samples[sample_idx]
            if sample_idx < len(samples)
            else {"raw_text": "", "n_prompt": 0, "n_completion": 0}
        )
        parsed_array, truncated = _safe_parse_json_array(
            sample_r["raw_text"], len(pids_in_order),
        )
        for pos, pid in enumerate(pids_in_order):
            is_leader = pos == 0
            parsed_obj = (
                parsed_array[pos]
                if pos < len(parsed_array)
                else {"_parse_error": True, "_raw": "missing_in_array"}
            )
            per_pid_vlm[pid] = {
                "parsed": parsed_obj,
                "raw_text": sample_r["raw_text"] if is_leader else "",
                "n_prompt": int(sample_r["n_prompt"]) if is_leader else 0,
                "n_completion": int(sample_r["n_completion"]) if is_leader else 0,
                "latency_sec": 0.0,
                "truncation_recovered": truncated,
                "is_batch_leader": is_leader,
                "batch_error": None,
            }
    return per_pid_vlm


def _fill_missing_per_pid_vlm(
    per_pid_vlm: dict[int, dict[str, Any]],
    batches_info: dict[int, dict[str, Any]],
    pid_to_batch_meta: dict[int, dict[str, Any]],
) -> None:
    for _, info in batches_info.items():
        pids_in_order = info["pids_in_order"]
        for pid in pids_in_order:
            if pid in per_pid_vlm:
                continue
            meta = pid_to_batch_meta.get(pid, {})
            per_pid_vlm[pid] = {
                "parsed": {
                    "_parse_error": True,
                    "_raw": meta.get("batch_error", "batch_not_submitted"),
                },
                "raw_text": "",
                "n_prompt": 0,
                "n_completion": 0,
                "latency_sec": 0.0,
                "truncation_recovered": False,
                "is_batch_leader": meta.get("batch_pos", 1) == 1,
                "batch_error": meta.get("batch_error", "batch_not_submitted"),
            }


def _compose_paragraph_outputs(
    *,
    groups: dict[int, Any],
    per_pid_vlm: dict[int, dict[str, Any]],
    prefilter_reasons: dict[int, str],
    padded_by_pid: dict[int, tuple[int, int, int, int]],
    pid_to_batch_meta: dict[int, dict[str, Any]],
    batches_info: dict[int, dict[str, Any]],
    target_lang: str,
    m_per_batch: int,
    bbx: Any,
    processor: Any,
    full_wh: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    sys_pf = SYSTEM_PROMPT_MULTI.format(target_lang=target_lang, m=m_per_batch)
    usr_pf = _build_user_prompt_multi(target_lang, list(range(1, m_per_batch + 1)))

    paragraphs_out: list[dict[str, Any]] = []
    per_paragraph_profile: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for pid in sorted(groups.keys()):
        g = groups[pid]
        item_bboxes = g["item_bboxes"]
        ocr_text_joined = " ".join(g["texts"]).strip()
        batch_meta = pid_to_batch_meta.get(pid)
        info = (
            batches_info.get(batch_meta["batch_id"])
            if batch_meta is not None else None
        )
        padded_bbox = list(padded_by_pid[pid]) if pid in padded_by_pid else None

        prefiltered = pid in prefilter_reasons
        vlm_ok = True
        n_prompt = 0
        n_completion = 0
        raw_preview: str | None = None

        vlm_original_text = ""
        vlm_translated_text = ""
        vlm_bg_type: str | None = None
        vlm_font_family: str | None = None
        vlm_is_bold = False
        vlm_alignment: str | None = None
        vlm_text_orientation: str | None = None
        vlm_overflow: str = ""
        vlm_block: dict[str, Any] = {
            "should_translate": False,
            "original_text": "",
            "translated_text": "",
        }
        should_translate = False
        skip_reason: str | None = None

        if prefiltered:
            skip_reason = _map_rule_to_skip_reason(prefilter_reasons[pid])
            vlm_original_text = ocr_text_joined
            vlm_block = {
                "should_translate": False,
                "original_text": vlm_original_text,
                "translated_text": "",
            }
        elif pid in per_pid_vlm:
            r = per_pid_vlm[pid]
            n_prompt = r["n_prompt"]
            n_completion = r["n_completion"]
            parsed = r["parsed"]
            final = _finalize_translation_object(parsed)
            recovered = bool(r.get("truncation_recovered"))
            if isinstance(final, dict) and final.get("_parse_error"):
                vlm_ok = False
                raw_preview = (
                    final.get("_raw") or r["raw_text"]
                )[:500] or None
                skip_reason = "vlm_parse_error"
            else:
                should_translate = final["should_translate"]
                vlm_original_text = final["original_text"]
                vlm_translated_text = final["translated_text"]
                vlm_block = {
                    "should_translate": should_translate,
                    "original_text": vlm_original_text,
                    "translated_text": vlm_translated_text,
                }
                if recovered:
                    raw_preview = r["raw_text"][:500]

        original_text = vlm_original_text
        translated_text = vlm_translated_text if should_translate else ""

        text_color_hex = bbx._rgb_to_hex(bbx._median_rgb(g.get("text_colors", [])))
        bg_color_hex = bbx._rgb_to_hex(bbx._median_rgb(g.get("bg_colors", [])))
        font_size_px = bbx._estimate_font_size_px(item_bboxes)
        block_angle = _dominant_angle(g.get("angles", []))
        font_family = vlm_font_family if should_translate else "sans-serif"

        if should_translate:
            bg_type = vlm_bg_type or "solid"
            is_bold = vlm_is_bold
            alignment = vlm_alignment or "left"
            text_orientation = vlm_text_orientation or "horizontal_ltr"
            overflow = vlm_overflow or _infer_overflow_strategy(
                original_text, translated_text,
            )
        else:
            bg_type = None
            is_bold = False
            alignment = "left"
            overflow = "keep"

        recommended_font_size_px = bbx._recommend_font_size_px(
            font_size_px, original_text, translated_text,
        )
        valid_hints: list[int] = []

        token_cost: dict[str, Any] | None = None
        if pid in per_pid_vlm and per_pid_vlm[pid]["is_batch_leader"]:
            token_cost = build_vlm_token_cost(
                n_prompt=n_prompt, n_completion=n_completion,
                system_prompt=sys_pf, user_prompt=usr_pf,
                global_wh=full_wh, crop_wh=full_wh,
                image_layout="fullpage_single",
                processor=processor,
                apply_chat_template=_apply_chat_template_no_think,
            )

        entry: dict[str, Any] = {
            "paragraph_no": pid,
            "paragraph_bbox": g["paragraph_bbox"],
            "item_count": len(item_bboxes),
            "item_bboxes": item_bboxes,
            "ocr_text": ocr_text_joined,
            "ocr_mean_confidence": g.get("mean_confidence", 0.0),
            "crop_path": info.get("crop_path") if info is not None else None,
            "padded_bbox": padded_bbox,
            "prefiltered": prefiltered,
            "vlm_early_exit": False,
            "vlm": vlm_block,
            "vlm_ok": vlm_ok,
            "vlm_tokens": {"prompt": n_prompt, "completion": n_completion},
            "vlm_raw_preview": raw_preview,
            "doubao_token_cost": token_cost,
            "vlm_batch_id": batch_meta["batch_id"] if batch_meta else None,
            "vlm_batch_pos": batch_meta["batch_pos"] if batch_meta else None,
            "vlm_batch_size": batch_meta["batch_size"] if batch_meta else None,
            "vlm_is_batch_leader": (
                per_pid_vlm.get(pid, {}).get("is_batch_leader", False)
                if not prefiltered else False
            ),
            "block_type": "body",
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
        if pid in per_pid_vlm and per_pid_vlm[pid].get("is_batch_leader"):
            per_paragraph_profile.append({
                "pid": pid,
                "batch_id": batch_meta["batch_id"] if batch_meta else None,
                "batch_size": batch_meta["batch_size"] if batch_meta else None,
                "crop_wh": [
                    info.get("crop_w", 0) if info is not None else 0,
                    info.get("crop_h", 0) if info is not None else 0,
                ],
                "crop_pixels": (
                    int(info.get("crop_w", 0)) * int(info.get("crop_h", 0))
                    if info is not None else 0
                ),
                "prompt_tokens": n_prompt,
                "completion_tokens": n_completion,
                "doubao_token_cost": token_cost,
            })

    return paragraphs_out, per_paragraph_profile, total_prompt_tokens, total_completion_tokens


# ---------------------------------------------------------------------------
#  Single-image processing
# ---------------------------------------------------------------------------
def _chunk(seq: list[int], m: int) -> list[list[int]]:
    if m <= 0:
        m = 1
    return [seq[i:i + m] for i in range(0, len(seq), m)]


def _sort_pids_by_box_position(
    pids: list[int],
    boxes: dict[int, tuple[int, int, int, int]],
) -> list[int]:
    """Organize inference batches in reading order; output mapping is guaranteed by the explicit ``#pid`` labels."""
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


def process_one(
    *,
    ocr_json_path: Path,
    image_path: Path,
    output_dir: Path,
    llm: Any,
    processor: Any,
    target_lang: str,
    per_box_max_new_tokens: int,
    overall_max_new_tokens: int,
    crop_pad: int,
    m_per_batch: int = 3,
    save_visualize: bool = False,
    save_crops: bool = True,
    write_profile: bool = True,
    tgt_lang_code: str | None = None,
    temperature: float = 0.0,
    num_samples: int = 1,
    ctx_n: int = 0,
    scene_summary: str | None = None,
) -> dict[str, Any]:
    """OCR -> grouping -> rule-based pre-filter -> split into m-batches and render multi-box images -> vLLM continuous batching ->
    parse the JSON arrays -> assemble paragraphs_out.

    ``ctx_n``: number of OCR teacher context paragraphs; 0=off; >0=nearest n; -1=all non-target paragraphs.
    ``scene_summary``: if non-empty, the scene-summary teacher takes priority (OCR context is ignored).
    """
    stem = image_path.stem
    t_start = time.perf_counter()
    bbx = _load_bbx_mod()

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
            "tgt_lang_code": tgt_lang_code,
            "global_info": {"global_angle": 0},
            "m_per_batch": m_per_batch,
            "num_paragraphs": 0,
            "num_vlm_batches": 0,
            "paragraphs": [],
            "_warning": "no_ocr_items",
        }
        (output_dir / f"{stem}__vlm.json").write_text(
            json.dumps(empty, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return empty

    vis_path_str: str | None = None
    if save_visualize:
        vis_path = output_dir / f"{stem}__bbox.jpg"
        draw_bbox_visualization(image_bgr, groups, vis_path, show_item_bbox=True)
        vis_path_str = str(vis_path)

    # Rule-based pre-filter + collect candidate pids for the VLM
    prefilter_reasons: dict[int, str] = {}
    vlm_pids_sorted: list[int] = []
    padded_by_pid: dict[int, tuple[int, int, int, int]] = {}
    for pid in sorted(groups.keys()):
        g = groups[pid]
        bbox = g.get("paragraph_bbox")
        if bbox is None:
            prefilter_reasons[pid] = "empty"
            continue
        pb = bbx._padded_paragraph_bbox(image_bgr, bbox, crop_pad)
        if pb is None:
            prefilter_reasons[pid] = "empty"
            continue
        padded_by_pid[pid] = pb
        joined = " ".join(g["texts"]).strip()
        skip, reason = classify_no_translate(joined)
        if skip:
            prefilter_reasons[pid] = reason or "rule"
        else:
            vlm_pids_sorted.append(pid)

    # Sorting only keeps boxes in the same batch as adjacent as possible; the model
    # output and backfill are anchored by the on-image #pid and the user-given order.
    vlm_pids_sorted = _sort_pids_by_box_position(vlm_pids_sorted, padded_by_pid)

    # Split into batches of size m
    batches: list[list[int]] = _chunk(vlm_pids_sorted, m_per_batch)

    # Render multi-box images
    crop_dir = output_dir / "crops"
    batches_info = frame_multi_paragraphs_on_full_image(
        image_bgr, groups, crop_dir, stem=stem,
        batches=batches, pad=crop_pad, save_to_disk=save_crops,
    )
    t_after_crop = time.perf_counter()

    # Build vLLM inputs
    batch_images: list[tuple[int, PILImage.Image, list[int]]] = []
    pid_to_batch_meta: dict[int, dict[str, Any]] = {}
    batch_crop_paths: dict[int, str | None] = {}

    for b_idx, info in batches_info.items():
        pids_in_order = info["pids_in_order"]
        if not pids_in_order:
            continue
        crop_path = info.get("crop_path")
        batch_crop_paths[b_idx] = crop_path
        pil = info.get("crop_pil")
        if pil is None and crop_path:
            pil = PILImage.open(crop_path).convert("RGB")
        if pil is None:
            for pos, pid in enumerate(pids_in_order, 1):
                pid_to_batch_meta[pid] = {
                    "batch_id": b_idx, "batch_pos": pos,
                    "batch_size": len(pids_in_order),
                    "batch_error": "frame_error:no_pil",
                }
            continue
        batch_images.append((b_idx, pil, pids_in_order))
        for pos, pid in enumerate(pids_in_order, 1):
            pid_to_batch_meta[pid] = {
                "batch_id": b_idx, "batch_pos": pos,
                "batch_size": len(pids_in_order),
            }

    inputs, order, prompt_meta = build_vlm_inputs_multi(
        processor,
        batch_images,
        target_lang,
        ocr_by_pid={
            int(pid): " ".join(g.get("texts") or []).strip()
            for pid, g in groups.items()
        },
        ctx_n=ctx_n,
        scene_summary=scene_summary,
    )
    n_boxes_list = [len(pids) for _, pids in order]

    # Submit in one llm.generate() call
    vlm_results, decode_wall = run_vlm_batches_multi(
        llm, inputs, n_boxes_list,
        per_box_budget=per_box_max_new_tokens,
        overall_cap=overall_max_new_tokens,
        temperature=temperature,
        num_samples=num_samples,
    )
    t_after_decode = time.perf_counter()

    full_wh = (img_w, img_h)
    all_samples_payload: list[dict[str, Any]] = []
    for sample_idx in range(num_samples):
        per_pid_vlm = _assign_per_pid_from_batch_results(order, vlm_results, sample_idx)
        _fill_missing_per_pid_vlm(per_pid_vlm, batches_info, pid_to_batch_meta)
        paragraphs_out, per_paragraph_profile, total_prompt_tokens, total_completion_tokens = (
            _compose_paragraph_outputs(
                groups=groups,
                per_pid_vlm=per_pid_vlm,
                prefilter_reasons=prefilter_reasons,
                padded_by_pid=padded_by_pid,
                pid_to_batch_meta=pid_to_batch_meta,
                batches_info=batches_info,
                target_lang=target_lang,
                m_per_batch=m_per_batch,
                bbx=bbx,
                processor=processor,
                full_wh=full_wh,
            )
        )
        all_samples_payload.append({
            "sample_idx": sample_idx,
            "paragraphs": paragraphs_out,
            "per_batch": per_paragraph_profile,
            "tokens": {
                "total_prompt": total_prompt_tokens,
                "total_completion": total_completion_tokens,
            },
        })

    paragraphs_out = all_samples_payload[0]["paragraphs"]
    per_paragraph_profile = all_samples_payload[0]["per_batch"]
    total_prompt_tokens = all_samples_payload[0]["tokens"]["total_prompt"]
    total_completion_tokens = all_samples_payload[0]["tokens"]["total_completion"]
    per_pid_vlm = _assign_per_pid_from_batch_results(order, vlm_results, 0)
    _fill_missing_per_pid_vlm(per_pid_vlm, batches_info, pid_to_batch_meta)

    # Diagnostics (based on sample 0)
    leader_completions = [
        r["n_completion"] for r in per_pid_vlm.values() if r.get("is_batch_leader")
    ]
    max_comp = max(leader_completions, default=0)
    median_comp = (
        sorted(leader_completions)[len(leader_completions) // 2]
        if leader_completions else 0
    )
    errors = [
        (pid, r["batch_error"]) for pid, r in per_pid_vlm.items()
        if r.get("is_batch_leader") and r.get("batch_error")
    ]

    # Persist the SP/UP actually used for inference (take the first batch; when the
    # teacher carries context it differs from the student)
    if prompt_meta:
        sys_pf = prompt_meta[0]["system_prompt"]
        usr_pf = prompt_meta[0]["user_prompt"]
    else:
        sys_pf = SYSTEM_PROMPT_MULTI.format(target_lang=target_lang, m=m_per_batch)
        usr_pf = _build_user_prompt_multi(target_lang, list(range(1, m_per_batch + 1)))

    num_paragraphs = len(paragraphs_out)
    num_api_calls = sum(1 for r in per_pid_vlm.values() if r.get("is_batch_leader"))
    num_vlm_pids = len(per_pid_vlm)
    avg_sec_per_call = (
        round(decode_wall / num_api_calls, 3) if num_api_calls else 0.0
    )
    avg_sec_per_paragraph = (
        round(decode_wall / num_vlm_pids, 3) if num_vlm_pids else 0.0
    )

    token_rollups = build_token_rollups(
        paragraphs_out,
        system_prompt=sys_pf,
        user_prompt=usr_pf,
        num_vlm_calls=num_api_calls,
        processor=processor,
        apply_chat_template=_apply_chat_template_no_think,
        num_images=1,
        note=(
            f"单阶段整图多框 vLLM (m_per_batch={m_per_batch}, ctx_n={ctx_n}); "
            "每 batch 仅 leader 持有 doubao_token_cost, sum_* 为 vLLM 真实用量; "
            "SP/UP 为实际推理口径（教师含语境块）。"
        ),
    )
    vlm_prompts = vlm_prompts_record(sys_pf, usr_pf)
    # Also keep a summary of the real prompts of each batch, to check whether ctx_n=-1 fills the budget
    vlm_prompts["per_batch"] = [
        {
            "batch_idx": m["batch_idx"],
            "pids": m["pids"],
            "teacher_mode": m.get("teacher_mode"),
            "teacher_ctx": m["teacher_ctx"],
            "ctx_n": m["ctx_n"],
            "ctx_pids": m["ctx_pids"],
            "ctx_chars": m["ctx_chars"],
            "scene_chars": m.get("scene_chars", 0),
            "scene_summary": m.get("scene_summary"),
            "user_prompt": m["user_prompt"],
            "system_prompt": m["system_prompt"],
        }
        for m in prompt_meta
    ]

    all_block_angles = [
        p["block_angle"] for p in paragraphs_out if p.get("block_angle") is not None
    ]
    global_angle = _dominant_angle(all_block_angles) if all_block_angles else 0

    result: dict[str, Any] = {
        "stem": stem,
        "image_path": str(image_path),
        "ocr_json_path": str(ocr_json_path),
        "image_size": [img_w, img_h],
        "ocr_size": ocr_size,
        "target_lang": target_lang,
        "tgt_lang_code": tgt_lang_code,
        "global_info": {"global_angle": global_angle},
        "m_per_batch": m_per_batch,
        "num_samples": num_samples,
        "temperature": temperature,
        "num_paragraphs": num_paragraphs,
        "num_vlm_paragraphs": num_vlm_pids,
        "num_vlm_batches": num_api_calls,
        "vlm_batch_wall_sec": round(decode_wall, 3),
        "vlm_avg_sec_per_api_call": avg_sec_per_call,
        "vlm_avg_sec_per_paragraph": avg_sec_per_paragraph,
        "bbox_visualization": vis_path_str,
        "paragraphs": paragraphs_out,
        "samples": all_samples_payload,
        "vlm_prompts": vlm_prompts,
        "doubao_token_rollups": token_rollups,
    }

    out_json = output_dir / f"{stem}__vlm.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    if write_profile:
        profile = {
            "image_stem": stem,
            "backend": "vllm_single_stage_multi_box",
            "image_size": [img_w, img_h],
            "m_per_batch": m_per_batch,
            "num_paragraphs": num_paragraphs,
            "num_vlm_paragraphs": num_vlm_pids,
            "num_vlm_batches": num_api_calls,
            "batch_wall_sec": round(decode_wall, 3),
            "tokens": {
                "total_prompt": total_prompt_tokens,
                "total_completion": total_completion_tokens,
                "avg_prompt_per_call": (
                    round(total_prompt_tokens / num_api_calls, 1) if num_api_calls else 0.0
                ),
                "avg_completion_per_call": (
                    round(total_completion_tokens / num_api_calls, 1)
                    if num_api_calls else 0.0
                ),
                "avg_prompt_per_paragraph": (
                    round(total_prompt_tokens / num_vlm_pids, 1) if num_vlm_pids else 0.0
                ),
                "avg_completion_per_paragraph": (
                    round(total_completion_tokens / num_vlm_pids, 1)
                    if num_vlm_pids else 0.0
                ),
            },
            "latency": {
                "batch_wall_sec": round(decode_wall, 3),
                "avg_wall_per_api_call_sec": avg_sec_per_call,
                "avg_wall_per_paragraph_sec": avg_sec_per_paragraph,
                "median_completion_tokens": median_comp,
                "max_completion_tokens": max_comp,
            },
            "per_batch": per_paragraph_profile,
            "vlm_prompts": vlm_prompts,
            "doubao_token_rollups": token_rollups,
        }
        (output_dir / f"{stem}__profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        result["profile"] = profile

    total_wall = time.perf_counter() - t_start
    prep_wall = t_after_prep - t_start
    crop_wall = t_after_crop - t_after_prep
    io_wall = total_wall - prep_wall - crop_wall - (t_after_decode - t_after_crop)

    err_tag = f" ERRORS={len(errors)}" if errors else ""
    print(
        f"[{stem}] total={total_wall:5.2f}s | vllm={decode_wall:5.2f}s "
        f"(comp median/max={median_comp}/{max_comp}) | "
        f"prep={prep_wall*1000:4.0f}ms frame={crop_wall*1000:4.0f}ms "
        f"io={io_wall*1000:4.0f}ms | "
        f"paragraphs total={len(groups)} prefilter={len(prefilter_reasons)} "
        f"vlm_pids={num_vlm_pids} batches={num_api_calls} (m={m_per_batch}){err_tag}"
    )
    return result


# ---------------------------------------------------------------------------
#  Batch mode: OCR dir + image dir
# ---------------------------------------------------------------------------
def _find_image_for_ocr(
    ocr_json: Path, image_dir: Path,
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Path | None:
    raw_stem = ocr_json.stem
    candidates = {raw_stem}
    if raw_stem.endswith("_ocr"):
        candidates.add(raw_stem[:-4])
    for stem in candidates:
        for ext in image_exts:
            p = image_dir / f"{stem}{ext}"
            if p.is_file():
                return p
            # bucket subdirectory (one level)
            hits = list(image_dir.glob(f"*/{stem}{ext}"))
            if hits:
                return hits[0]
    # Fallback: rglob with bounded depth effect
    for stem in candidates:
        for ext in image_exts:
            hits = list(image_dir.rglob(f"{stem}{ext}"))
            if hits:
                return hits[0]
    return None


def list_ocr_jsons(ocr_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        [p for p in ocr_dir.rglob("*.json") if p.is_file()],
    )
    return files[:limit] if limit else files


# jsonl tgt_lang / target_language (ISO short codes) -> Chinese language names used in the scheme-three prompt
_LANG_CODE_TO_NAME: dict[str, str] = {
    "zh": "中文",
    "zh-cn": "中文",
    "zh_hans": "中文",
    "zh-hans": "中文",
    "zh-tw": "繁体中文",
    "zh_hant": "繁体中文",
    "zh-hant": "繁体中文",
    "en": "英语",
    "en-us": "英语",
    "en-gb": "英语",
    "vi": "越南语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "es": "西班牙语",
    "es-mx": "西班牙语",
    "es-es": "西班牙语",
    "fr": "法语",
    "ko": "韩语",
    "ja": "日语",
    "th": "泰语",
    "id": "印尼语",
    "ms": "马来语",
    "ur": "乌尔都语",
    "bn": "孟加拉语",
    "pt": "葡萄牙语",
    "de": "德语",
    "it": "意大利语",
    "hi": "印地语",
    "tr": "土耳其语",
}


def map_tgt_lang_code(code: str | None, default: str = "中文") -> str:
    """Online-log tgt_lang (zh/en/zh-TW...) -> language name for the VLM prompt."""
    raw = (code or "").strip()
    if not raw:
        return default
    if any("\u4e00" <= c <= "\u9fff" for c in raw):
        return raw
    key = raw.lower().replace("_", "-")
    if key in _LANG_CODE_TO_NAME:
        return _LANG_CODE_TO_NAME[key]
    base = key.split("-")[0]
    return _LANG_CODE_TO_NAME.get(base, default)


def _load_image_meta_from_jsonl(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """stem -> {image_path, tgt_lang_code, target_lang_name}."""
    out: dict[str, dict[str, Any]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ip = d.get("image_path") or d.get("path")
            if ip:
                p = Path(ip)
            elif d.get("filename"):
                p = Path(d["filename"])
            else:
                continue
            code = d.get("tgt_lang") or d.get("target_language") or ""
            out[p.stem] = {
                "image_path": p,
                "tgt_lang_code": str(code).strip(),
            }
    return out


def _index_ocr_by_stem(ocr_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in ocr_dir.rglob("*.json"):
        if not p.is_file():
            continue
        raw = p.stem
        out[raw] = p
        if raw.endswith("_ocr"):
            out[raw[:-4]] = p
    return out


def build_jobs_from_dirs(
    ocr_dir: Path,
    image_dir: Path | None,
    image_list_jsonl: Path | None,
    limit: int | None,
    *,
    default_target_lang: str = "中文",
) -> list[tuple[Path, Path, str, str]]:
    """Returns [(ocr_json, image, target_lang_name, tgt_lang_code), ...]."""
    if image_list_jsonl is not None:
        ocr_index = _index_ocr_by_stem(ocr_dir)
        meta_by_stem = _load_image_meta_from_jsonl(image_list_jsonl)
        stems = sorted(meta_by_stem.keys())
        if limit is not None:
            stems = stems[:limit]
        jobs: list[tuple[Path, Path, str, str]] = []
        miss_ocr = miss_img = 0
        for stem in stems:
            oj = ocr_index.get(f"{stem}_ocr", ocr_index.get(stem))
            if oj is None:
                miss_ocr += 1
                continue
            meta = meta_by_stem[stem]
            img = meta["image_path"]
            if not img.is_file() and image_dir is not None:
                alt = _find_image_for_ocr(Path(f"{stem}_ocr.json"), image_dir)
                if alt is not None:
                    img = alt
            if not img.is_file():
                miss_img += 1
                continue
            code = meta.get("tgt_lang_code") or ""
            tlang = map_tgt_lang_code(code, default=default_target_lang)
            jobs.append((oj, img, tlang, code or ""))
        print(
            f"[jobs] jsonl stems={len(meta_by_stem)} jobs={len(jobs)} "
            f"missing_ocr={miss_ocr} missing_img={miss_img}"
        )
        return jobs

    if image_dir is None:
        raise SystemExit("未提供 --image-list-jsonl 时必须给 --image-dir")
    jobs = []
    for oj in list_ocr_jsons(ocr_dir, limit):
        img = _find_image_for_ocr(oj, image_dir)
        if img is None:
            continue
        jobs.append((oj, img, default_target_lang, ""))
    return jobs


# ---------------------------------------------------------------------------
#  vLLM engine construction and warmup
# ---------------------------------------------------------------------------
def _build_vllm(args) -> tuple[Any, Any]:
    _warn_if_vllm_without_cuda()
    kind, arch, _ = detect_model_kind(args.model_path)
    if kind != "vl":
        raise SystemExit(f"--model-path 非 VLM (kind={kind})")
    mm_kw: dict[str, Any] = {}
    if args.vllm_max_pixels is not None:
        mm_kw["max_pixels"] = int(args.vllm_max_pixels)
    if args.vllm_min_pixels is not None:
        mm_kw["min_pixels"] = int(args.vllm_min_pixels)
    extra: dict[str, Any] = {"enable_prefix_caching": not args.vllm_disable_prefix_caching}
    # vLLM's InternVL processor in the PPU build does not accept max_pixels.
    # Its own image processor handles the dynamic resolution settings.
    if mm_kw and not arch.lower().startswith("internvl"):
        extra["mm_processor_kwargs"] = mm_kw
    if args.vllm_max_num_seqs is not None:
        extra["max_num_seqs"] = int(args.vllm_max_num_seqs)
    llm, processor, _ = _load_vllm_vl_engine(
        args.model_path, dtype=args.dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=args.vllm_enforce_eager,
        extra_engine_kwargs=extra,
    )
    return llm, processor


def _warmup(llm: Any, processor: Any, target_lang: str, m: int) -> None:
    try:
        from vllm import SamplingParams
    except ImportError:
        return
    dummy = PILImage.new("RGB", (320, 320), (250, 250, 250))
    pids = list(range(1, m + 1))
    sp = SYSTEM_PROMPT_MULTI.format(target_lang=target_lang, m=m)
    up = _build_user_prompt_multi(target_lang, pids)
    msgs = [
        {"role": "system", "content": sp},
        {"role": "user", "content": [
            {"type": "image", "image": "__PLACEHOLDER__"},
            {"type": "text", "text": up},
        ]},
    ]
    prompt_str = _apply_chat_template_no_think(
        processor, msgs, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    sp_params = SamplingParams(max_tokens=32, temperature=0)
    t0 = time.perf_counter()
    llm.generate(
        [{"prompt": prompt_str, "multi_modal_data": {"image": dummy}}],
        sampling_params=sp_params, use_tqdm=False,
    )
    print(f"[warmup] {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "带框外 #pid 的一图多框 SFT 推理：每个框输出固定三字段翻译判断"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ocr-json", type=str, default=None, help="单张: OCR JSON 路径")
    p.add_argument("--image", type=str, default=None, help="单张: 原图路径")
    p.add_argument("--ocr-dir", type=str, default=None, help="批量: OCR JSON 目录")
    p.add_argument("--image-dir", type=str, default=None, help="批量: 原图目录")
    p.add_argument(
        "--image-list-jsonl", type=str, default=None,
        help="只处理 jsonl 中的图片 (image_path/filename); 仍需 --ocr-dir",
    )

    p.add_argument("--output-dir", type=str, required=True, help="输出目录")
    p.add_argument("--model-path",
                   default=None,
                   help="local HuggingFace model directory (required)")
    p.add_argument(
        "--target-lang", default="中文",
        help="目标语言默认值; 若提供 --image-list-jsonl 则优先用每条记录的 tgt_lang "
             "(zh/en/zh-TW... 会映射为中文/英语/繁体中文等), 空字段回退到本默认值",
    )
    p.add_argument(
        "--m", dest="m_per_batch", type=int, default=3,
        help="一张图里同时打多少个段落主框 (一次 VLM 调用覆盖多少段)。默认 3",
    )
    p.add_argument("--crop-pad", type=int, default=6,
                   help="段落 OCR 外接框扩展像素")
    p.add_argument("--per-box-max-new-tokens", type=int, default=512,
                   help="每个框允许的最大生成 token 数 (按框数缩放)")
    p.add_argument("--max-new-tokens", type=int, default=4096,
                   help="单次 forward 总 max_tokens 上限")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="采样温度; num-samples=1 且 temperature=0 时为 greedy")
    p.add_argument("--num-samples", type=int, default=1,
                   help="每个 VLM batch 采样条数 (pass@k 评测用)")
    p.add_argument(
        "--ctx-n", type=int, default=0,
        help="OCR 教师语境段落数: 0=关; >0=最近 n 段; -1=全部非目标段。"
             "若同时给 --scene-summary-json，则优先用场景概括、忽略本项。",
    )
    p.add_argument(
        "--scene-summary-json", type=str, default=None,
        help="场景概括 JSON（stem -> {summary: ...}）。启用后走场景概括教师，不再注入 OCR 上下文。",
    )
    p.add_argument("--limit", type=int, default=None, help="批量模式: 最多处理张数")
    p.add_argument("--sleep-between-images", type=float, default=0.0,
                   help="处理完每张图后额外 sleep 秒数")

    p.add_argument("--visualize", action="store_true",
                   help="写 <stem>__bbox.jpg, 默认关")
    p.add_argument("--no-save-crops", action="store_true",
                   help="不落盘多框整图, 直接内存 PIL")
    p.add_argument("--no-profile", action="store_true",
                   help="不写 <stem>__profile.json")

    p.add_argument("--dtype", default="fp16",
                   choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--vllm-gpu-memory-utilization", type=float,
                   default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95")))
    p.add_argument("--vllm-tensor-parallel-size", type=int,
                   default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--vllm-max-model-len", type=int, default=16384)
    p.add_argument("--vllm-max-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--vllm-min-pixels", type=int, default=None)
    p.add_argument("--vllm-max-num-seqs", type=int, default=256)
    p.add_argument("--vllm-enforce-eager", action="store_true")
    p.add_argument("--vllm-disable-prefix-caching", action="store_true")
    p.add_argument("--no-warmup", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    is_single = args.ocr_json and args.image
    is_batch = bool(args.ocr_dir) and (bool(args.image_dir) or bool(args.image_list_jsonl))
    if not is_single and not is_batch:
        raise SystemExit(
            "请二选一:\n"
            "  (A) --ocr-json + --image      单张模式\n"
            "  (B) --ocr-dir  + (--image-dir 和/或 --image-list-jsonl)  批量模式"
        )

    if args.m_per_batch < 1:
        raise SystemExit("--m 必须 >= 1")
    if args.num_samples < 1:
        raise SystemExit("--num-samples 必须 >= 1")
    if args.temperature < 0:
        raise SystemExit("--temperature 必须 >= 0")
    if args.num_samples > 1 and args.temperature == 0:
        print("[warn] num-samples>1 且 temperature=0, 多条采样可能完全相同")
    if args.ctx_n < -1:
        raise SystemExit("--ctx-n 只能是 -1 / 0 / 正整数")

    scene_map = _load_scene_summary_map(
        Path(args.scene_summary_json).resolve() if args.scene_summary_json else None
    )
    if args.scene_summary_json:
        print(
            f"[scene-summary] loaded {len(scene_map)} stems from {args.scene_summary_json}",
            flush=True,
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # (ocr_json, image, target_lang_name, tgt_lang_code)
    jobs: list[tuple[Path, Path, str, str]] = []
    if is_single:
        ocr_json = Path(args.ocr_json).resolve()
        image = Path(args.image).resolve()
        if not ocr_json.is_file():
            raise SystemExit(f"OCR JSON 不存在: {ocr_json}")
        if not image.is_file():
            raise SystemExit(f"原图不存在: {image}")
        jobs.append((ocr_json, image, args.target_lang, ""))
    else:
        jobs = build_jobs_from_dirs(
            ocr_dir=Path(args.ocr_dir).resolve(),
            image_dir=Path(args.image_dir).resolve() if args.image_dir else None,
            image_list_jsonl=(
                Path(args.image_list_jsonl).resolve() if args.image_list_jsonl else None
            ),
            limit=args.limit,
            default_target_lang=args.target_lang,
        )
        if not jobs:
            raise SystemExit(
                f"没有找到可处理的 OCR/图片对 (ocr-dir={args.ocr_dir})"
            )

    print(
        f"[single-stage-multi-box] m={args.m_per_batch} "
        f"per_box_budget={args.per_box_max_new_tokens} "
        f"max_new_tokens={args.max_new_tokens} "
        f"num_samples={args.num_samples} temperature={args.temperature} "
        f"ctx_n={args.ctx_n} scene_summaries={len(scene_map)} "
        f"default_target_lang={args.target_lang!r} "
        f"per_image_tgt_lang_from_jsonl={bool(args.image_list_jsonl)}"
    )

    llm, processor = _build_vllm(args)
    if not args.no_warmup:
        _warmup(llm, processor, args.target_lang, args.m_per_batch)

    try:
        overall_t0 = time.perf_counter()
        for idx, (ocr_json, image, tlang, tcode) in enumerate(jobs, 1):
            print(
                f"\n=== [{idx}/{len(jobs)}] {image.name} "
                f"tgt={tcode or '-'}->{tlang} ==="
            )
            try:
                scene = scene_map.get(image.stem)
                if scene_map and scene is None:
                    print(f"[warn] no scene summary for stem={image.stem}", flush=True)
                process_one(
                    ocr_json_path=ocr_json,
                    image_path=image,
                    output_dir=output_dir,
                    llm=llm,
                    processor=processor,
                    target_lang=tlang,
                    per_box_max_new_tokens=args.per_box_max_new_tokens,
                    overall_max_new_tokens=args.max_new_tokens,
                    crop_pad=args.crop_pad,
                    m_per_batch=args.m_per_batch,
                    save_visualize=args.visualize,
                    save_crops=not args.no_save_crops,
                    write_profile=not args.no_profile,
                    tgt_lang_code=tcode or None,
                    temperature=args.temperature,
                    num_samples=args.num_samples,
                    ctx_n=0 if scene else args.ctx_n,
                    scene_summary=scene,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"[ERROR] {image.name}: {e!r}")
            if args.sleep_between_images > 0 and idx < len(jobs):
                time.sleep(args.sleep_between_images)
        total = time.perf_counter() - overall_t0
        print(
            f"\n[DONE] processed {len(jobs)} image(s) in {total:.2f}s "
            f"(avg {total / max(1, len(jobs)):.2f}s/image)"
        )
    finally:
        _cleanup(llm, processor)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
