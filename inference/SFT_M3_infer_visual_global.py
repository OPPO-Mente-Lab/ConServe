# Copyright 2026 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT_M3_infer_visual_global.py
=============================
Visual OPSD teacher inference: the same SFT VLM, fed with "a sharp full image + a
low-resolution global image".

- First image I: a sharp framed full image with orange boxes/#PID labels (used to
  read the target text)
- Second image I_global: downsampled from I to about ``--global-max-pixels``
  (default 200704) pixels to emphasize page topic/layout/scene while blurring small
  text details

The prompt explicitly requires: understand the context from the low-resolution
global image first, then translate using the sharp image.

Example usage::

  CUDA_VISIBLE_DEVICES=0 python SFT_M3_infer_visual_global.py \\
    --ocr-dir /path/to/ocr \\
    --image-dir /path/to/images \\
    --output-dir /path/to/output \\
    --model-path /path/to/model \\
    --global-max-pixels 200704 \\
    --m 3 --temperature 0.0 --dtype bf16 \\
    --vllm-max-pixels 1048576 --vllm-gpu-memory-utilization 0.9
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import SFT_M3_optimized as m3  # noqa: E402

from batch_photo_translate_vlm import (  # noqa: E402
    _apply_chat_template_no_think,
    _cleanup,
)

DEFAULT_MODEL = "/path/to/model"
DEFAULT_GLOBAL_MAX_PIXELS = 200704  # ≈ 256 × 28×28 vision tokens

# ---------------------------------------------------------------------------
#  Prompt: force reading the low-resolution global image before translating the sharp one
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_VISUAL_GLOBAL = """图片多区域文字翻译任务。输入包含两张图：
第一张是带橙色粗框与红色 #PID 标签的清晰完整截图，用于识别目标框内文字；
第二张是同一截图的低分辨率全局视图，小字与局部细节被弱化，用于把握页面整体主题、场景、布局和主要视觉对象。
你必须先根据第二张低分辨率全局图理解待翻译页面的全局语境（界面类型、领域、场景与版面结构），再结合第一张清晰图中的橙框逐 PID 翻译；禁止脱离全局语境的直译或通用义项硬翻。
对图中短词、短句（如 step、go 等），必须先结合全局语境判断含义再翻译，禁止按词典默认义孤立直译。
第二张图仅供理解语境，不得把对第二张图的描述写入 original_text 或 translated_text。
图中有 {m} 个橙色粗框；每个框外紧邻一个红色 #PID 标签，框与标签共同标识一个独立处理单元。
按用户给出的 #PID 顺序逐框识别文字并判断是否翻译。清晰可读、非{target_lang}的自然语言必须翻译；已是{target_lang}、纯数字符号、URL或邮箱、代码标识符、单字符、商标Logo、人名或模糊残缺文字不翻译。
只输出严格单行JSON数组，长度必须为 {m}，数组元素与用户给出的 #PID 顺序一一对应；不得遗漏、合并、拆分或交换。
每个数组元素固定且只能包含以下三个字段，字段顺序固定:
需要翻译:{{"should_translate":true,"original_text":"框内原文","translated_text":"结合全局语境的{target_lang}译文"}}
不需翻译:{{"should_translate":false,"original_text":"框内可辨文字","translated_text":""}}
original_text只填写对应橙框内的可见文字，不得混入标签、相邻框、第二张图描述或框外内容；无法辨认时为空字符串。
translated_text仅在should_translate=true时填写；false时必须为空字符串。"""


def _build_user_prompt_visual_global(target_lang: str, pids: list[int]) -> str:
    m = len(pids)
    labels = "、".join(f"#{pid}" for pid in pids)
    order = " → ".join(f"#{pid}" for pid in pids)
    return (
        f"目标语言：{target_lang}。输入为两张图："
        f"第一张=清晰待译图（{m} 个橙框，编号 {labels}）；"
        f"第二张=低分辨率全局语境图（用于页面主题/场景/布局）。\n"
        f"请先根据第二张低分辨率图弄清全局语境，再对第一张清晰图进行翻译；"
        f"短词短句（如 step、go）须先分析语境再译，禁止词典直译。\n"
        f"严格按 {order} 输出长度为 {m} 的三字段JSON数组；"
        f"禁止脱离全局语境的直译；不要输出对第二张图的描述。"
    )


def make_global_image(
    pil: PILImage.Image,
    target_pixels: int = DEFAULT_GLOBAL_MAX_PIXELS,
) -> PILImage.Image:
    """Downsample the sharp image to about ``target_pixels`` total pixels (default 200704)."""
    img = pil.convert("RGB")
    w, h = img.size
    cur = max(1, w * h)
    if cur <= target_pixels:
        return img.copy()
    scale = math.sqrt(target_pixels / float(cur))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    # Fine-tune so nw*nh is as close as possible to target_pixels
    while nw > 1 and nh > 1 and nw * nh > target_pixels:
        if nw >= nh:
            nw -= 1
        else:
            nh -= 1
    return img.resize((nw, nh), PILImage.Resampling.BILINEAR)


def build_vlm_inputs_visual_global(
    processor: Any,
    batch_images: list[tuple[int, PILImage.Image, list[int]]],
    target_lang: str,
    *,
    global_max_pixels: int = DEFAULT_GLOBAL_MAX_PIXELS,
    save_global_dir: Path | None = None,
    stem: str | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, list[int]]], list[dict[str, Any]]]:
    """Two-image teacher input: sharp image + low-resolution global image."""
    inputs: list[dict[str, Any]] = []
    order: list[tuple[int, list[int]]] = []
    prompt_meta: list[dict[str, Any]] = []

    for b_idx, pil, pids in batch_images:
        pil_full = pil.convert("RGB")
        pil_global = make_global_image(pil_full, global_max_pixels)
        if save_global_dir is not None:
            save_global_dir.mkdir(parents=True, exist_ok=True)
            tag = stem or "img"
            out_p = save_global_dir / f"{tag}__batch{b_idx:03d}__global.jpg"
            pil_global.save(out_p, quality=90)

        sp = SYSTEM_PROMPT_VISUAL_GLOBAL.format(
            target_lang=target_lang, m=len(pids),
        )
        up = _build_user_prompt_visual_global(target_lang, pids)
        msgs = [
            {"role": "system", "content": sp},
            {"role": "user", "content": [
                {"type": "image", "image": "__PLACEHOLDER__"},  # sharp image
                {"type": "image", "image": "__PLACEHOLDER__"},  # low-resolution global image
                {"type": "text", "text": up},
            ]},
        ]
        prompt_str = _apply_chat_template_no_think(
            processor, msgs, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs.append({
            "prompt": prompt_str,
            "multi_modal_data": {"image": [pil_full, pil_global]},
        })
        order.append((b_idx, pids))
        gw, gh = pil_global.size
        prompt_meta.append({
            "batch_idx": b_idx,
            "pids": list(pids),
            "teacher_mode": "visual_global",
            "teacher_ctx": True,
            "global_max_pixels": global_max_pixels,
            "global_wh": [gw, gh],
            "global_pixels": gw * gh,
            "full_wh": list(pil_full.size),
            "system_prompt": sp,
            "user_prompt": up,
            "ctx_n": 0,
            "ctx_pids": [],
            "ctx_chars": 0,
            "scene_summary": None,
            "scene_chars": 0,
        })
    return inputs, order, prompt_meta


def _install_patch(*, global_max_pixels: int, save_global: bool) -> None:
    """Replace SFT_M3_optimized's build/process/warmup/vllm hooks with the two-image versions."""
    _state: dict[str, Any] = {"stem": None, "output_dir": None}
    _orig_process_one = m3.process_one

    def _patched_build_vlm_inputs(
        processor: Any,
        batch_images: list[tuple[int, PILImage.Image, list[int]]],
        target_lang: str,
        *,
        ocr_by_pid: dict[int, str] | None = None,
        ctx_n: int = 0,
        scene_summary: str | None = None,
    ):
        del ocr_by_pid, ctx_n, scene_summary
        save_dir = None
        if save_global and _state.get("output_dir"):
            save_dir = Path(_state["output_dir"]) / "global_crops"
        return build_vlm_inputs_visual_global(
            processor,
            batch_images,
            target_lang,
            global_max_pixels=global_max_pixels,
            save_global_dir=save_dir,
            stem=_state.get("stem"),
        )

    def _process_one_wrapped(**kwargs):
        _state["stem"] = Path(kwargs["image_path"]).stem
        _state["output_dir"] = kwargs.get("output_dir")
        kwargs["ctx_n"] = 0
        kwargs["scene_summary"] = None
        return _orig_process_one(**kwargs)

    def _warmup_dual(llm: Any, processor: Any, target_lang: str, m: int) -> None:
        try:
            from vllm import SamplingParams
        except ImportError:
            return
        dummy = PILImage.new("RGB", (320, 320), (250, 250, 250))
        dummy_g = make_global_image(dummy, global_max_pixels)
        pids = list(range(1, m + 1))
        sp = SYSTEM_PROMPT_VISUAL_GLOBAL.format(target_lang=target_lang, m=m)
        up = _build_user_prompt_visual_global(target_lang, pids)
        msgs = [
            {"role": "system", "content": sp},
            {"role": "user", "content": [
                {"type": "image", "image": "__PLACEHOLDER__"},
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
            [{"prompt": prompt_str, "multi_modal_data": {"image": [dummy, dummy_g]}}],
            sampling_params=sp_params, use_tqdm=False,
        )
        print(f"[warmup-visual-global] {time.perf_counter() - t0:.2f}s")

    def _build_vllm_dual(args):
        """Override the default limit_mm_per_prompt image=1 to allow two images."""
        m3._warn_if_vllm_without_cuda()
        from batch_photo_translate_vlm import (  # noqa: WPS433
            _load_vllm_vl_engine,
            detect_model_kind,
        )
        kind, _, _ = detect_model_kind(args.model_path)
        if kind != "vl":
            raise SystemExit(f"--model-path 非 VLM (kind={kind})")
        mm_kw: dict[str, Any] = {}
        if args.vllm_max_pixels is not None:
            mm_kw["max_pixels"] = int(args.vllm_max_pixels)
        if getattr(args, "vllm_min_pixels", None) is not None:
            mm_kw["min_pixels"] = int(args.vllm_min_pixels)
        extra: dict[str, Any] = {
            "enable_prefix_caching": not args.vllm_disable_prefix_caching,
            "limit_mm_per_prompt": {"image": 2},
        }
        if mm_kw:
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
        print("[visual-global] limit_mm_per_prompt={'image': 2}", flush=True)
        return llm, processor

    m3.build_vlm_inputs_multi = _patched_build_vlm_inputs  # type: ignore[assignment]
    m3.process_one = _process_one_wrapped  # type: ignore[assignment]
    m3._warmup = _warmup_dual  # type: ignore[assignment]
    m3._build_vllm = _build_vllm_dual  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="视觉 OPSD 教师推理：清晰图 + 低清全局图(默认 200704 像素)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ocr-json", type=str, default=None)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--ocr-dir", type=str, default=None)
    p.add_argument("--image-dir", type=str, default=None)
    p.add_argument("--image-list-jsonl", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--target-lang", default="中文")
    p.add_argument("--m", dest="m_per_batch", type=int, default=3)
    p.add_argument("--crop-pad", type=int, default=6)
    p.add_argument("--per-box-max-new-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep-between-images", type=float, default=0.0)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--no-save-crops", action="store_true")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument(
        "--global-max-pixels", type=int, default=DEFAULT_GLOBAL_MAX_PIXELS,
        help="低清全局图目标总像素，默认 200704",
    )
    p.add_argument(
        "--save-global-crops", action="store_true", default=True,
        help="将降采样后的全局图落到 output_dir/global_crops（默认开）",
    )
    p.add_argument("--no-save-global-crops", action="store_true",
                   help="不落盘全局图")
    # Placeholder args for compatibility with SFT_M3_optimized (text/scene teacher privileges are ignored in visual mode)
    p.add_argument("--ctx-n", type=int, default=0)
    p.add_argument("--scene-summary-json", type=str, default=None)
    p.add_argument("--dtype", default="bf16",
                   choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--vllm-gpu-memory-utilization", type=float,
                   default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")))
    p.add_argument("--vllm-tensor-parallel-size", type=int,
                   default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--vllm-max-model-len", type=int, default=16384)
    p.add_argument("--vllm-max-pixels", type=int, default=1048576,
                   help="清晰图 processor 上限；全局图已预缩到 global-max-pixels")
    p.add_argument("--vllm-min-pixels", type=int, default=None)
    p.add_argument("--vllm-max-num-seqs", type=int, default=64)
    p.add_argument("--vllm-enforce-eager", action="store_true")
    p.add_argument("--vllm-disable-prefix-caching", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.global_max_pixels < 1024:
        raise SystemExit("--global-max-pixels 过小")
    if args.m_per_batch < 1:
        raise SystemExit("--m 必须 >= 1")
    if args.num_samples < 1:
        raise SystemExit("--num-samples 必须 >= 1")

    save_global = bool(args.save_global_crops) and not bool(args.no_save_global_crops)
    _install_patch(
        global_max_pixels=int(args.global_max_pixels),
        save_global=save_global,
    )

    # Demonstrate a global-image size to sanity-check 200704
    demo = make_global_image(
        PILImage.new("RGB", (1024, 2048), (128, 128, 128)),
        args.global_max_pixels,
    )
    print(
        f"[visual-global] global_max_pixels={args.global_max_pixels} "
        f"demo_from_1024x2048 -> {demo.size[0]}x{demo.size[1]} "
        f"({demo.size[0]*demo.size[1]} px)",
        flush=True,
    )

    # Reuse the argument namespace fields of SFT_M3_optimized.main
    # and go through the same job loop as optimized
    is_single = args.ocr_json and args.image
    is_batch = bool(args.ocr_dir) and (bool(args.image_dir) or bool(args.image_list_jsonl))
    if not is_single and not is_batch:
        raise SystemExit(
            "请二选一:\n"
            "  (A) --ocr-json + --image\n"
            "  (B) --ocr-dir + (--image-dir 和/或 --image-list-jsonl)"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_single:
        jobs = [(
            Path(args.ocr_json).resolve(),
            Path(args.image).resolve(),
            args.target_lang,
            "",
        )]
    else:
        jobs = m3.build_jobs_from_dirs(
            ocr_dir=Path(args.ocr_dir).resolve(),
            image_dir=Path(args.image_dir).resolve() if args.image_dir else None,
            image_list_jsonl=(
                Path(args.image_list_jsonl).resolve() if args.image_list_jsonl else None
            ),
            limit=args.limit,
            default_target_lang=args.target_lang,
        )
        if not jobs:
            raise SystemExit("没有可处理的 OCR/图片对")

    print(
        f"[visual-global-infer] jobs={len(jobs)} m={args.m_per_batch} "
        f"T={args.temperature} model={args.model_path}",
        flush=True,
    )

    llm, processor = m3._build_vllm(args)
    if not args.no_warmup:
        m3._warmup(llm, processor, args.target_lang, args.m_per_batch)

    try:
        t0 = time.perf_counter()
        for idx, (ocr_json, image, tlang, tcode) in enumerate(jobs, 1):
            print(
                f"\n=== [{idx}/{len(jobs)}] {image.name} "
                f"tgt={tcode or '-'}->{tlang} ===",
                flush=True,
            )
            try:
                m3.process_one(
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
                    ctx_n=0,
                    scene_summary=None,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"[ERROR] {image.name}: {e!r}", flush=True)
            if args.sleep_between_images > 0 and idx < len(jobs):
                time.sleep(args.sleep_between_images)
        print(f"[DONE] {len(jobs)} image(s) in {time.perf_counter()-t0:.1f}s", flush=True)
    finally:
        _cleanup(llm, processor)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
