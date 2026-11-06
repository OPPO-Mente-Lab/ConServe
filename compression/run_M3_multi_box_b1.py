#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""
run_M3_multi_box_b1.py — Scheme-3 multi-box + B-1 OCR-guided visual token merge
================================================================================

**Recommended runtime (B-1)**:
  conda: /path/to/data
  vLLM:  0.19.0+v0.1.0.ppu2.1.0.oe  (supports ``--enable-mm-embeds``, Qwen3.5-VL)
  Before launch:
    source /usr/local/PPU_SDK/envsetup.sh   # or setup_vllm_ppu_env.sh
    export LD_LIBRARY_PATH=...

Two backends:
  --b1-backend hf
      Single-process HF direct inference (encode+merge+generate), suitable for validating
      the merge algorithm and paper ablations.
  --b1-backend vllm
      Two stages: HF only runs visual encode+merge -> vLLM ``enable_mm_embeds`` decoding.
      By default all batches of each image are submitted together in a **single**
      ``llm.generate()`` call (fair comparison with method3); the old per-batch serial
      behavior is available via ``--b1-vllm-sequential``.
      On OOM, add ``--b1-embed-cache-dir`` to spill embeds to disk before starting vLLM.

Running B-1 on the vLLM 0.15.0+ppu2.0 image is **not recommended** (it lacks enable_mm_embeds / ships older Qwen3.5 support).

Usage:
  /path/to/data \\
      compression/run_M3_multi_box_b1.py \\
      --b1-backend hf \\
      --ocr-json ... --image ... --output-dir ... \\
      --model-path /path/to/Qwen3.5-9B-Instruct \\
      --bg-merge-factor 4
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_INFERENCE_DIR = _REPO_ROOT / "inference"
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# The OCR-guided merge core lives in ogstb_b1_merge.encode_merge_batch:
#   visual encode -> bbox mask -> background pooled compression -> outputs merged_embeds + merged_grid_thw
from ogstb_b1_merge import encode_merge_batch  # noqa: E402
from vlm_sp_up_record import (  # noqa: E402
    build_token_rollups,
    build_vlm_token_cost,
    vlm_prompts_record,
)

# Keep every non-compression behavior identical to the SFT baseline.  The
# only B-1-specific operation is ``encode_merge_batch`` below, which replaces
# the visual tokens after this common prompt/render/parse pipeline is built.
_BASE_PATH = Path(os.environ.get("CONSERVE_BASE_INFERENCE", Path(__file__).resolve().parents[1] / "inference" / "SFT_M3_optimized.py"))
if not _BASE_PATH.is_file():
    raise FileNotFoundError(f"找不到 SFT M3 基线脚本: {_BASE_PATH}")
# SFT imports m3_pid_labels before it establishes its own dependency paths.
if str(_BASE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH.parent))
_spec = importlib.util.spec_from_file_location("m3_multi_box_base", _BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载: {_BASE_PATH}")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def _resolve_hf_device(device: str) -> str:
    if device in ("auto", "", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_hf_dtype(dtype: str) -> Any:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "auto": "auto",
    }.get(dtype, torch.float16)


# ---------------------------------------------------------------------------
# HF side: visual-only loading (used by vllm backend stage 1, to avoid OOM with the full vLLM model on the same card)
# ---------------------------------------------------------------------------

class _HFVisualEncodeShim:
    """Keep only visual + get_image_features, for B-1 encode_merge."""

    def __init__(self, visual: torch.nn.Module) -> None:
        self.visual = visual
        self.model = self

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        pixel_values = pixel_values.type(self.visual.dtype)
        # extract_visual_embeddings_hf passes return_dict=True; visual.forward does not accept the duplicate kw
        call_kw = {k: v for k, v in kwargs.items() if k != "return_dict"}
        vision_output = self.visual(
            pixel_values, grid_thw=image_grid_thw, **call_kw,
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        if isinstance(image_embeds, torch.Tensor):
            image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    def eval(self) -> "_HFVisualEncodeShim":
        self.visual.eval()
        return self


def _load_visual_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Read only the vision weights from a single or indexed safetensors checkpoint."""
    from safetensors import safe_open

    prefix = "model.visual."
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files: dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            if key.startswith(prefix):
                files.setdefault(filename, []).append(key)
        if not files:
            raise KeyError(f"{index_path} contains no {prefix}* weights")
    else:
        weight_path = model_dir / "model.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json in {model_dir}")
        files = {weight_path.name: []}

    state: dict[str, torch.Tensor] = {}
    for filename, indexed_keys in files.items():
        if Path(filename).name != filename:
            raise ValueError(f"Invalid checkpoint shard name: {filename}")
        shard = model_dir / filename
        if not shard.is_file():
            raise FileNotFoundError(f"Checkpoint shard not found: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            keys = indexed_keys or [key for key in reader.keys() if key.startswith(prefix)]
            for key in keys:
                state[key[len(prefix):]] = reader.get_tensor(key)
    if not state:
        raise KeyError(f"No {prefix}* weights in {model_dir}")
    return state


def _load_hf_visual_only(
    model_path: str,
    dtype: str,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load only the vision encoder (~a few hundred M) to avoid OOM sharing the card with the full vLLM model."""
    from transformers import AutoConfig, AutoProcessor

    hf_dtype = _resolve_hf_dtype(dtype)
    target_device = _resolve_hf_device(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise ValueError(f"{model_path} 缺少 vision_config, 无法 visual-only 加载")

    model_type = getattr(config, "model_type", "")
    if model_type == "qwen3_5":
        from transformers import Qwen3_5VisionModel
        visual_cls = Qwen3_5VisionModel
    elif model_type in ("qwen3_vl", "qwen2_vl"):
        from transformers import Qwen3VLVisionModel
        visual_cls = Qwen3VLVisionModel
    else:
        raise ValueError(
            f"不支持的 model_type={model_type!r} visual-only 加载; "
            "请改用 --b1-backend hf 或提 issue"
        )

    visual = visual_cls(vision_config)
    visual_sd = _load_visual_state_dict(Path(model_path))

    missing, unexpected = visual.load_state_dict(visual_sd, strict=False)
    if missing:
        print(f"[warn] visual-only missing keys: {len(missing)} (前 3 个: {missing[:3]})")
    if unexpected:
        print(f"[warn] visual-only unexpected keys: {len(unexpected)}")

    if hf_dtype != "auto":
        visual = visual.to(dtype=hf_dtype)
    visual = visual.to(target_device)
    visual.eval()
    shim = _HFVisualEncodeShim(visual)
    print(
        f"[info] B-1 visual-only 已加载 ({len(visual_sd)} tensors) → {target_device}"
    )
    return shim, processor


def _load_hf_vlm(model_path: str, dtype: str, device: str = "auto") -> tuple[Any, Any]:
    """Full HF VLM (--b1-backend hf): encode+merge+generate all done inside HF."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import importlib.util
    has_accelerate = importlib.util.find_spec("accelerate") is not None

    hf_dtype = _resolve_hf_dtype(dtype)

    target_device = _resolve_hf_device(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    load_kw: dict[str, Any] = {"trust_remote_code": True}
    if hf_dtype != "auto":
        load_kw["dtype"] = hf_dtype

    # The mt-flow env often lacks accelerate; device_map would raise
    if has_accelerate and device in ("auto", "balanced", "balanced_low_0", "sequential"):
        load_kw["device_map"] = device
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw)
    else:
        if device in ("auto", "balanced", "balanced_low_0", "sequential") and not has_accelerate:
            print(
                "[warn] 未安装 accelerate, 改用单卡加载 "
                f"(.to({target_device!r})); 可 pip install accelerate 启用 device_map"
            )
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw)
        model = model.to(target_device)

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# vLLM side: load the enable_mm_embeds engine and receive HF-precomputed merged visual embeds
# ---------------------------------------------------------------------------

def _build_vllm_b1(args: argparse.Namespace) -> tuple[Any, Any]:
    from batch_photo_translate_vlm import (  # noqa: E402
        _load_vllm_vl_engine,
        _warn_if_vllm_without_cuda,
        detect_model_kind,
    )

    _warn_if_vllm_without_cuda()
    kind, _, _ = detect_model_kind(args.model_path)
    if kind != "vl":
        raise SystemExit(f"--model-path 非 VLM: {kind}")

    mm_kw: dict[str, Any] = {}
    if args.vllm_max_pixels is not None:
        mm_kw["max_pixels"] = int(args.vllm_max_pixels)
    if args.vllm_min_pixels is not None:
        mm_kw["min_pixels"] = int(args.vllm_min_pixels)

    extra: dict[str, Any] = {
        "enable_prefix_caching": not args.vllm_disable_prefix_caching,
        "enable_mm_embeds": True,
    }
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


def _hf_generate_one(
    model: Any,
    processor: Any,
    pil: PILImage.Image,
    bboxes: list[tuple[int, int, int, int]],
    orig_hw: tuple[int, int],
    target_lang: str,
    pids: list[int],
    *,
    bg_merge_factor: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    """HF direct inference for a single batch: first run OCR-guided merge to collect
    compression statistics, then run standard generate for the translation.

    OCR-guided merge pipeline (see ogstb_b1_merge.encode_merge_batch):
      1. processor turns the framed crop image into pixel_values + image_grid_thw
      2. The HF visual encoder outputs N token embeddings (N,D)
      3. Mark a foreground mask on the token grid using OCR paragraph bboxes (inside box=True)
      4. Foreground tokens are kept as-is; every bg_merge_factor background tokens are mean-pooled into 1
      5. Recompute merged_grid_thw so the vLLM placeholder count matches the compressed token count

    Note: HF generate currently still consumes the full image tokens (merged embeds are not
        injected); pack["meta"] only records compression ratios such as
        n_tokens_orig / n_tokens_merged for profile comparison.
    """
    # boxes = per-paragraph OCR bbox (original-image coordinates), mapped to the visual token grid during merge
    pack = encode_merge_batch(
        model, processor, pil, bboxes,
        orig_hw=orig_hw,
        bg_merge_factor=bg_merge_factor,
        device=device,
    )
    sp = base.SYSTEM_PROMPT_MULTI.format(target_lang=target_lang, m=len(pids))
    up = base._build_user_prompt_multi(target_lang, pids)
    messages = [
        {"role": "system", "content": sp},
        {"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": up},
        ]},
    ]
    text = base._apply_chat_template_no_think(
        processor, messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[pil], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_len = int(inputs["input_ids"].shape[-1])
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    wall = time.perf_counter() - t0

    new_ids = out_ids[0, input_len:]
    raw = processor.decode(new_ids, skip_special_tokens=True).strip()
    raw = base._strip_thinking(raw)
    parsed_array, truncated = base._safe_parse_json_array(raw, len(pids))

    return {
        "raw_text": raw,
        "parsed": parsed_array,
        "truncation_recovered": truncated,
        "n_prompt": input_len,
        "n_completion": int(new_ids.shape[-1]),
        "wall_sec": wall,
        "ogstb_b1_meta": pack["meta"],
    }


def _build_b1_profile(
    *,
    stem: str,
    args: argparse.Namespace,
    img_w: int,
    img_h: int,
    num_paragraphs: int,
    num_vlm_pids: int,
    order: list[tuple[int, list[int]]],
    order_batch_meta: list[dict[str, Any]],
    vlm_results: list[dict[str, Any]],
    processor: Any | None,
) -> dict[str, Any]:
    """Aligned with the method3 __profile.json: split vLLM prompt/completion + SP/UP/image tokens."""
    full_wh = (img_w, img_h)
    per_batch: list[dict[str, Any]] = []
    rollup_paragraphs: list[dict[str, Any]] = []
    total_prompt = 0
    total_completion = 0
    total_wall = 0.0
    sum_orig_enc = 0
    sum_merged_enc = 0
    completion_tokens_list: list[int] = []

    ref_sp = base.SYSTEM_PROMPT_MULTI.format(
        target_lang=args.target_lang, m=args.m_per_batch,
    )
    ref_up = ""

    for i, (b_idx, pids) in enumerate(order):
        meta = order_batch_meta[i]
        r = vlm_results[i]
        sp = meta["system_prompt"]
        up = meta["user_prompt"]
        if not ref_up:
            ref_up = up
        info = meta["batch_info"]
        n_prompt = int(r.get("n_prompt") or 0)
        n_completion = int(r.get("n_completion") or 0)
        wall = float(r.get("wall_sec") or 0.0)
        crop_w = int(info.get("crop_w") or img_w)
        crop_h = int(info.get("crop_h") or img_h)
        crop_wh = (crop_w, crop_h)

        token_cost: dict[str, Any] | None = None
        if processor is not None and n_prompt > 0:
            token_cost = build_vlm_token_cost(
                n_prompt=n_prompt,
                n_completion=n_completion,
                system_prompt=sp,
                user_prompt=up,
                global_wh=full_wh,
                crop_wh=crop_wh,
                image_layout="fullpage_single",
                processor=processor,
                apply_chat_template=base._apply_chat_template_no_think,
            )

        ogstb = r.get("ogstb_b1_meta") or {}
        sum_orig_enc += int(ogstb.get("n_tokens_orig") or 0)
        sum_merged_enc += int(ogstb.get("n_tokens_merged") or 0)
        total_prompt += n_prompt
        total_completion += n_completion
        total_wall += wall
        if n_completion > 0:
            completion_tokens_list.append(n_completion)

        per_batch.append({
            "pid": pids[0] if pids else None,
            "batch_id": b_idx,
            "batch_size": len(pids),
            "pids": pids,
            "crop_wh": [crop_w, crop_h],
            "crop_pixels": crop_w * crop_h,
            "prompt_tokens": n_prompt,
            "completion_tokens": n_completion,
            "wall_sec": round(wall, 6),
            "doubao_token_cost": token_cost,
            "ogstb_b1_meta": ogstb or None,
        })
        if token_cost is not None:
            rollup_paragraphs.append({"doubao_token_cost": token_cost})

    num_api_calls = len(order)
    median_comp = (
        sorted(completion_tokens_list)[len(completion_tokens_list) // 2]
        if completion_tokens_list else 0
    )
    max_comp = max(completion_tokens_list) if completion_tokens_list else 0

    if not ref_up and num_api_calls:
        ref_up = base._build_user_prompt_multi(args.target_lang, order[0][1])

    token_rollups = build_token_rollups(
        rollup_paragraphs,
        system_prompt=ref_sp,
        user_prompt=ref_up,
        num_vlm_calls=num_api_calls,
        processor=processor,
        apply_chat_template=base._apply_chat_template_no_think,
        num_images=1,
        note=(
            f"B-1 ({args.b1_backend}) m_per_batch={args.m_per_batch}, "
            f"bg_merge_factor={args.bg_merge_factor}; "
            f"vllm_decode={'sequential' if getattr(args, 'b1_vllm_sequential', False) else 'batched'}; "
            "prompt/completion=vLLM 引擎 token id; "
            "SP/UP/图像=vLLM prompt 减 chat_template 文本(与 method3 profile 同口径); "
            "batched 时 batch_wall_sec=单次 llm.generate 墙钟, per_batch wall 均分; "
            "ogstb_b1_meta.n_tokens_orig/merged 为 encoder 内部计数, 另列。"
        ),
    )
    token_rollups["sum_n_tokens_orig_encoder"] = sum_orig_enc
    token_rollups["sum_n_tokens_merged_encoder"] = sum_merged_enc
    if sum_merged_enc > 0:
        token_rollups["encoder_to_vllm_image_ratio_avg"] = round(
            sum_orig_enc / sum_merged_enc, 4,
        )

    vllm_decode_mode = None
    if args.b1_backend == "vllm":
        vllm_decode_mode = (
            "sequential" if getattr(args, "b1_vllm_sequential", False) else "batched"
        )

    return {
        "image_stem": stem,
        "backend": f"b1_{args.b1_backend}_token_merge",
        "vllm_batch_decode": vllm_decode_mode,
        "image_size": [img_w, img_h],
        "m_per_batch": args.m_per_batch,
        "num_paragraphs": num_paragraphs,
        "num_vlm_paragraphs": num_vlm_pids,
        "num_vlm_batches": num_api_calls,
        "batch_wall_sec": round(total_wall, 3),
        "ogstb_b1_config": {"bg_merge_factor": args.bg_merge_factor},
        "tokens": {
            "total_prompt": total_prompt,
            "total_completion": total_completion,
            "avg_prompt_per_call": (
                round(total_prompt / num_api_calls, 1) if num_api_calls else 0.0
            ),
            "avg_completion_per_call": (
                round(total_completion / num_api_calls, 1) if num_api_calls else 0.0
            ),
            "avg_prompt_per_paragraph": (
                round(total_prompt / num_vlm_pids, 1) if num_vlm_pids else 0.0
            ),
            "avg_completion_per_paragraph": (
                round(total_completion / num_vlm_pids, 1) if num_vlm_pids else 0.0
            ),
        },
        "latency": {
            "batch_wall_sec": round(total_wall, 3),
            "avg_wall_per_api_call_sec": (
                round(total_wall / num_api_calls, 3) if num_api_calls else 0.0
            ),
            "avg_wall_per_paragraph_sec": (
                round(total_wall / num_vlm_pids, 3) if num_vlm_pids else 0.0
            ),
            "median_completion_tokens": median_comp,
            "max_completion_tokens": max_comp,
        },
        "per_batch": per_batch,
        "vlm_prompts": vlm_prompts_record(ref_sp, ref_up),
        "doubao_token_rollups": token_rollups,
    }


def _build_vllm_embed_request(
    prompt_str: str,
    merged_embeds: torch.Tensor,
    merged_grid_thw: torch.Tensor,
) -> dict[str, Any]:
    """Build a vLLM enable_mm_embeds request: skip the vision encoder and feed compressed embeds directly.

    multi_modal_data.image carries:
      - image_embeds:  (N_merged, D)  visual tokens after OCR-guided merge
      - image_grid_thw: (1,3)  grid corresponding to N_merged, for vLLM placeholder allocation
    """
    embeds = merged_embeds.contiguous()
    if torch.cuda.is_available() and embeds.device.type == "cpu":
        embeds = embeds.cuda()
    grid = merged_grid_thw.flatten()
    if grid.numel() != 3:
        raise ValueError(f"merged_grid_thw 应为长度 3, 得到 {tuple(grid.shape)}")
    grid_2d = grid.view(1, 3).to(device=embeds.device, dtype=torch.int64)
    return {
        "prompt": prompt_str,
        "multi_modal_data": {
            "image": {
                "image_embeds": embeds,
                "image_grid_thw": grid_2d,
            },
        },
    }


def _parse_vllm_generate_output(ro: Any) -> dict[str, Any]:
    raw = (ro.outputs[0].text or "").strip()
    return {
        "raw_text": base._strip_thinking(raw),
        "n_prompt": len(ro.prompt_token_ids or []),
        "n_completion": len(ro.outputs[0].token_ids or []),
    }


def _vllm_generate_batches_with_embeds(
    llm: Any,
    requests: list[dict[str, Any]],
    max_new_tokens_list: list[int],
) -> tuple[list[dict[str, Any]], float]:
    """Submit all batches in one llm.generate() call (aligned with method3 run_vlm_batches_multi)."""
    from vllm import SamplingParams

    if not requests:
        return [], 0.0

    t0 = time.perf_counter()
    if len(set(max_new_tokens_list)) == 1:
        sp = SamplingParams(
            max_tokens=max_new_tokens_list[0],
            temperature=0,
            stop=base._STOP_STRINGS,
            include_stop_str_in_output=True,
        )
        outs = llm.generate(requests, sampling_params=sp, use_tqdm=False)
    else:
        sps = [
            SamplingParams(
                max_tokens=c,
                temperature=0,
                stop=base._STOP_STRINGS,
                include_stop_str_in_output=True,
            )
            for c in max_new_tokens_list
        ]
        outs = llm.generate(requests, sampling_params=sps, use_tqdm=False)
    wall = time.perf_counter() - t0

    results: list[dict[str, Any]] = []
    for ro in outs:
        results.append(_parse_vllm_generate_output(ro))
    return results, wall


def _vllm_generate_with_embeds(
    llm: Any,
    processor: Any,
    prompt_str: str,
    merged_embeds: torch.Tensor,
    merged_grid_thw: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Serial single-batch decode (for --b1-vllm-sequential debugging)."""
    req = _build_vllm_embed_request(prompt_str, merged_embeds, merged_grid_thw)
    results, wall = _vllm_generate_batches_with_embeds(llm, [req], [max_new_tokens])
    if not results:
        return {"raw_text": "", "n_prompt": 0, "n_completion": 0, "wall_sec": wall}
    return {**results[0], "wall_sec": wall}


def process_one_b1(
    *,
    args: argparse.Namespace,
    ocr_json_path: Path,
    image_path: Path,
    output_dir: Path,
    hf_model: Any | None,
    hf_processor: Any | None,
    vllm_llm: Any | None,
    vllm_processor: Any | None,
    encode_only: bool = False,
    precomputed_packs: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """B-1 per-image main flow: OCR grouping -> multi-box crops -> encode+merge -> decoding.

    Three inference paths (selected by args.b1_backend / encode_only):
      hf:           encode_merge_batch (statistics) + model.generate (full image tokens)
      vllm stage 1: encode_only=True -> only encode_merge_batch, then release visual VRAM
      vllm stage 2: precomputed merged embeds -> llm.generate(enable_mm_embeds)

    encode_only=True:  only HF visual encode+merge; return packs_by_batch.
    precomputed_packs: vllm stage 2 injects the merged embeds computed in stage 1.
    """
    import cv2

    bbx = base._load_bbx_mod()
    stem = image_path.stem
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr_raw = json.load(f)
    data = ocr_raw.get("data", ocr_raw)
    ocr_items = data.get("items", [])
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    img_h, img_w = image_bgr.shape[:2]
    groups = base.group_by_paragraph(ocr_items)

    prefilter_reasons: dict[int, str] = {}
    vlm_pids: list[int] = []
    padded_by_pid: dict[int, tuple[int, int, int, int]] = {}
    for pid in sorted(groups.keys()):
        g = groups[pid]
        bbox = g.get("paragraph_bbox")
        if bbox is None:
            continue
        pb = bbx._padded_paragraph_bbox(image_bgr, bbox, args.crop_pad)
        if pb is None:
            continue
        padded_by_pid[pid] = pb
        joined = " ".join(g["texts"]).strip()
        skip, reason = base.classify_no_translate(joined)
        if skip:
            prefilter_reasons[pid] = reason or "rule"
        else:
            vlm_pids.append(pid)

    # Same reading-order batching as SFT_M3_optimized.py.  PIDs are still
    # explicit in the rendered image and user prompt, so result mapping stays
    # deterministic after sorting.
    vlm_pids = base._sort_pids_by_box_position(vlm_pids, padded_by_pid)
    batches = base._chunk(vlm_pids, args.m_per_batch)
    crop_dir = output_dir / "crops"
    batches_info = base.frame_multi_paragraphs_on_full_image(
        image_bgr, groups, crop_dir, stem=stem,
        batches=batches, pad=args.crop_pad,
        save_to_disk=not args.no_save_crops,
    )

    device = _resolve_hf_device(args.b1_hf_device)
    vlm_results: list[dict[str, Any]] = []
    order: list[tuple[int, list[int]]] = []
    order_batch_meta: list[dict[str, Any]] = []
    packs_by_batch: dict[int, dict[str, Any]] = {}
    vllm_pending: list[dict[str, Any]] = []

    for b_idx, info in sorted(batches_info.items()):
        pids = info["pids_in_order"]
        if not pids:
            continue
        order.append((b_idx, pids))
        sp = base.SYSTEM_PROMPT_MULTI.format(
            target_lang=args.target_lang, m=len(pids),
        )
        up = base._build_user_prompt_multi(args.target_lang, pids)
        pil = info.get("crop_pil")
        if pil is None and info.get("crop_path"):
            pil = PILImage.open(info["crop_path"]).convert("RGB")
        if pil is None:
            vlm_results.append({"raw_text": "", "n_prompt": 0, "n_completion": 0, "ogstb_b1_meta": {}})
            order_batch_meta.append({
                "system_prompt": sp,
                "user_prompt": up,
                "batch_info": info,
            })
            continue

        # OCR bbox -> original-image coordinates, for encode_merge_batch to mark foreground on the visual token grid
        boxes = [tuple(padded_by_pid[p]) for p in pids]
        msgs = [
            {"role": "system", "content": sp},
            {"role": "user", "content": [
                {"type": "image", "image": "__PLACEHOLDER__"},
                {"type": "text", "text": up},
            ]},
        ]
        prompt_str = base._apply_chat_template_no_think(
            hf_processor or vllm_processor, msgs,
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

        # --- Inference branches ---
        if args.b1_backend == "hf":
            # HF full path: merge statistics + generate
            assert hf_model is not None and hf_processor is not None
            r = _hf_generate_one(
                hf_model, hf_processor, pil, boxes, (img_w, img_h),
                args.target_lang, pids,
                bg_merge_factor=args.bg_merge_factor,
                max_new_tokens=min(args.max_new_tokens, args.per_box_max_new_tokens * len(pids)),
                device=device,
            )
        elif encode_only:
            # vLLM stage 1: only HF visual encode + OCR-guided merge, no decode to disk
            assert hf_model is not None and hf_processor is not None
            pack = encode_merge_batch(
                hf_model, hf_processor, pil, boxes,
                orig_hw=(img_w, img_h),
                bg_merge_factor=args.bg_merge_factor,
                device=device,
            )
            cache_dir = args.b1_embed_cache_dir
            if cache_dir:
                cdir = Path(cache_dir) / stem
                cdir.mkdir(parents=True, exist_ok=True)
                torch.save(pack["merged_embeds"], cdir / f"batch{b_idx:03d}_embeds.pt")
                torch.save(pack["merged_grid_thw"], cdir / f"batch{b_idx:03d}_grid.pt")
            packs_by_batch[b_idx] = {
                **pack,
                "prompt_str": prompt_str,
                "pids": pids,
            }
            continue
        else:
            # vLLM stage 2: decode with merged embeds (stage-1 cache can be reused)
            assert vllm_llm is not None and vllm_processor is not None
            if precomputed_packs is not None and b_idx in precomputed_packs:
                cached = precomputed_packs[b_idx]
                pack = {
                    "merged_embeds": cached["merged_embeds"],
                    "merged_grid_thw": cached["merged_grid_thw"],
                    "meta": cached["meta"],
                }
                prompt_str = cached.get("prompt_str", prompt_str)
            else:
                assert hf_model is not None
                pack = encode_merge_batch(
                    hf_model, hf_processor or vllm_processor, pil, boxes,
                    orig_hw=(img_w, img_h),
                    bg_merge_factor=args.bg_merge_factor,
                    device=device,
                )
            cache_dir = args.b1_embed_cache_dir
            if cache_dir and b_idx not in (precomputed_packs or {}):
                cdir = Path(cache_dir) / stem
                cdir.mkdir(parents=True, exist_ok=True)
                torch.save(pack["merged_embeds"], cdir / f"batch{b_idx:03d}_embeds.pt")
                torch.save(pack["merged_grid_thw"], cdir / f"batch{b_idx:03d}_grid.pt")
            max_tok = min(
                args.max_new_tokens,
                args.per_box_max_new_tokens * len(pids),
            )
            if getattr(args, "b1_vllm_sequential", False):
                try:
                    r = _vllm_generate_with_embeds(
                        vllm_llm, vllm_processor, prompt_str,
                        pack["merged_embeds"], pack["merged_grid_thw"],
                        max_new_tokens=max_tok,
                    )
                    parsed_array, truncated = base._safe_parse_json_array(
                        r.get("raw_text", ""), len(pids),
                    )
                    r["parsed"] = parsed_array
                    r["truncation_recovered"] = truncated
                except Exception as e:
                    r = {
                        "raw_text": "",
                        "parsed": [
                            {"_parse_error": True, "_raw": repr(e)}
                            for _ in range(len(pids))
                        ],
                        "truncation_recovered": False,
                        "n_prompt": 0,
                        "n_completion": 0,
                        "wall_sec": 0.0,
                        "error": repr(e),
                    }
                r["ogstb_b1_meta"] = pack["meta"]
                vlm_results.append(r)
            else:
                vllm_pending.append({
                    "result_idx": len(vlm_results),
                    "b_idx": b_idx,
                    "pids": pids,
                    "pack": pack,
                    "prompt_str": prompt_str,
                    "max_new_tokens": max_tok,
                })
                vlm_results.append(None)
            order_batch_meta.append({
                "system_prompt": sp,
                "user_prompt": up,
                "batch_info": info,
            })
            continue

        vlm_results.append(r)
        order_batch_meta.append({
            "system_prompt": sp,
            "user_prompt": up,
            "batch_info": info,
        })

    # vLLM batched decode: all batches in one llm.generate (default, aligned with method3)
    if (
        not encode_only
        and args.b1_backend == "vllm"
        and vllm_pending
        and vllm_llm is not None
    ):
        requests = [
            _build_vllm_embed_request(
                job["prompt_str"],
                job["pack"]["merged_embeds"],
                job["pack"]["merged_grid_thw"],
            )
            for job in vllm_pending
        ]
        caps = [job["max_new_tokens"] for job in vllm_pending]
        try:
            batch_results, decode_wall = _vllm_generate_batches_with_embeds(
                vllm_llm, requests, caps,
            )
            per_batch_wall = (
                decode_wall / len(batch_results) if batch_results else 0.0
            )
            for job, br in zip(vllm_pending, batch_results):
                parsed_array, truncated = base._safe_parse_json_array(
                    br.get("raw_text", ""), len(job["pids"]),
                )
                r = {
                    **br,
                    "parsed": parsed_array,
                    "truncation_recovered": truncated,
                    "wall_sec": per_batch_wall,
                    "ogstb_b1_meta": job["pack"]["meta"],
                    "vllm_decode_wall_sec": decode_wall,
                }
                vlm_results[job["result_idx"]] = r
        except Exception as e:
            err_r = {
                "raw_text": "",
                "parsed": None,
                "truncation_recovered": False,
                "n_prompt": 0,
                "n_completion": 0,
                "wall_sec": 0.0,
                "error": repr(e),
            }
            for job in vllm_pending:
                r = {
                    **err_r,
                    "parsed": [
                        {"_parse_error": True, "_raw": repr(e)}
                        for _ in range(len(job["pids"]))
                    ],
                    "ogstb_b1_meta": job["pack"]["meta"],
                }
                vlm_results[job["result_idx"]] = r

    if encode_only:
        return {"stem": stem, "packs_by_batch": packs_by_batch}

    for i, r in enumerate(vlm_results):
        if r is None:
            raise RuntimeError(
                f"vllm batched decode 未填充 batch 结果 (index={i}, stem={stem})"
            )

    # Flatten per-pid + synthesize paragraphs (aligned with the multi-box output fields)
    per_pid_vlm: dict[int, dict[str, Any]] = {}
    batch_by_idx = {b_idx: (pids, vlm_results[i]) for i, (b_idx, pids) in enumerate(order)}
    for b_idx, info in batches_info.items():
        pids_in_order = info.get("pids_in_order") or []
        if b_idx not in batch_by_idx:
            continue
        _, r = batch_by_idx[b_idx]
        parsed_array = r.get("parsed") or []
        for pos, pid in enumerate(pids_in_order):
            parsed_obj = (
                parsed_array[pos]
                if pos < len(parsed_array)
                else {"_parse_error": True, "_raw": "missing_in_array"}
            )
            per_pid_vlm[pid] = {
                "parsed": parsed_obj,
                "raw_text": r.get("raw_text", ""),
                "batch_idx": b_idx,
                "batch_pos": pos + 1,
                "ogstb_b1_meta": r.get("ogstb_b1_meta") if pos == 0 else None,
                "error": r.get("error"),
            }

    paragraphs_out: list[dict[str, Any]] = []
    for pid in sorted(groups.keys()):
        g = groups[pid]
        ocr_text_joined = " ".join(g["texts"]).strip()
        prefiltered = pid in prefilter_reasons
        entry: dict[str, Any] = {
            "paragraph_no": pid,
            "paragraph_bbox": g.get("paragraph_bbox"),
            "ocr_text": ocr_text_joined,
            "prefiltered": prefiltered,
            "vlm_ok": True,
            "should_translate": False,
            "skip_reason": None,
            "original_text": "",
            "translated_text": "",
            "vlm_batch_id": None,
            "ogstb_b1_meta": None,
        }
        if prefiltered:
            entry["skip_reason"] = base._map_rule_to_skip_reason(prefilter_reasons[pid])
            # Match the SFT baseline: rule-filtered text is retained as the
            # recognized source text even though it is not sent to the VLM.
            entry["original_text"] = ocr_text_joined
        elif pid in per_pid_vlm:
            pinfo = per_pid_vlm[pid]
            parsed = pinfo["parsed"]
            final = base._finalize_translation_object(parsed)
            entry["vlm_batch_id"] = pinfo.get("batch_idx")
            if pinfo.get("batch_pos") == 1:
                entry["ogstb_b1_meta"] = pinfo.get("ogstb_b1_meta")
            if isinstance(final, dict) and final.get("_parse_error"):
                entry["vlm_ok"] = False
                entry["skip_reason"] = "vlm_parse_error"
                entry["vlm_raw_preview"] = (final.get("_raw") or pinfo.get("raw_text", ""))[:500]
            elif final.get("should_translate"):
                entry["should_translate"] = True
                entry["original_text"] = str(final.get("original_text", ""))
                entry["translated_text"] = str(final.get("translated_text", ""))
            else:
                # A valid SFT three-field negative is not a parse failure or
                # a rule skip; preserve its OCR result without an old-schema
                # synthetic skip reason.
                entry["original_text"] = str(final.get("original_text", ""))
        paragraphs_out.append(entry)

    result = {
        "stem": stem,
        "image_path": str(image_path),
        "ocr_json_path": str(ocr_json_path),
        "image_size": [img_w, img_h],
        "target_lang": args.target_lang,
        "m_per_batch": args.m_per_batch,
        "ogstb_method": "b1_token_merge",
        "ogstb_b1_backend": args.b1_backend,
        "ogstb_b1_config": {
            "bg_merge_factor": args.bg_merge_factor,
        },
        "vllm_batch_decode": (
            "sequential" if getattr(args, "b1_vllm_sequential", False)
            else ("batched" if args.b1_backend == "vllm" else None)
        ),
        "num_paragraphs": len(paragraphs_out),
        "paragraphs": paragraphs_out,
        "batches": [
            {
                "batch_idx": b_idx,
                "pids": pids,
                "vlm_completion": vlm_results[i].get("raw_text", ""),
                "vlm_raw_preview": vlm_results[i].get("raw_text", "")[:500],
                "parsed": vlm_results[i].get("parsed"),
                "truncation_recovered": vlm_results[i].get("truncation_recovered"),
                "n_prompt": vlm_results[i].get("n_prompt"),
                "n_completion": vlm_results[i].get("n_completion"),
                "wall_sec": vlm_results[i].get("wall_sec"),
                "ogstb_b1_meta": vlm_results[i].get("ogstb_b1_meta"),
                "error": vlm_results[i].get("error"),
            }
            for i, (b_idx, pids) in enumerate(order)
        ],
    }
    (output_dir / f"{stem}__vlm_b1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    token_processor = vllm_processor or hf_processor
    profile = _build_b1_profile(
        stem=stem,
        args=args,
        img_w=img_w,
        img_h=img_h,
        num_paragraphs=len(paragraphs_out),
        num_vlm_pids=len(vlm_pids),
        order=order,
        order_batch_meta=order_batch_meta,
        vlm_results=vlm_results,
        processor=token_processor,
    )
    (output_dir / f"{stem}__profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    result["profile"] = profile
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3 multi-box + B-1 token merge")
    p.add_argument("--ocr-json", type=str, default=None)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--ocr-dir", type=str, default=None)
    p.add_argument("--image-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--target-lang", default="中文")
    p.add_argument("--m", dest="m_per_batch", type=int, default=3)
    p.add_argument("--crop-pad", type=int, default=6)
    p.add_argument("--per-box-max-new-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-save-crops", action="store_true")
    p.add_argument("--dtype", default="fp16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument("--vllm-max-model-len", type=int, default=16384)
    p.add_argument("--vllm-max-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--vllm-min-pixels", type=int, default=None)
    p.add_argument("--vllm-max-num-seqs", type=int, default=64)
    p.add_argument("--vllm-enforce-eager", action="store_true")
    p.add_argument("--vllm-disable-prefix-caching", action="store_true")
    p.add_argument(
        "--b1-backend", choices=["hf", "vllm"], default="hf",
        help="hf=HF直推; vllm=HF encode+merge 后 enable_mm_embeds",
    )
    p.add_argument("--bg-merge-factor", type=int, default=4,
                   help="背景 token 每 k 个 merge 为 1")
    p.add_argument("--b1-embed-cache-dir", type=str, default=None,
                   help="vllm backend 时将 merged embed 落盘")
    p.add_argument(
        "--b1-hf-device", type=str, default="cuda",
        help="B-1 visual encode 设备 (vllm 两阶段阶段1); OOM 时可试 cpu",
    )
    p.add_argument(
        "--b1-vllm-sequential", action="store_true",
        help="vllm backend 逐 batch 串行 generate (旧行为); 默认批量提交与 method3 对齐",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path]] = []
    if args.ocr_json and args.image:
        jobs.append((Path(args.ocr_json).resolve(), Path(args.image).resolve()))
    elif args.ocr_dir and args.image_dir:
        ocr_dir = Path(args.ocr_dir).resolve()
        image_dir = Path(args.image_dir).resolve()
        for oj in base.list_ocr_jsons(ocr_dir, args.limit):
            img = base._find_image_for_ocr(oj, image_dir)
            if img:
                jobs.append((oj, img))
    else:
        raise SystemExit("需要 --ocr-json+--image 或 --ocr-dir+--image-dir")

    if not jobs:
        hint = ""
        if args.ocr_dir:
            ocr_dir = Path(args.ocr_dir).resolve()
            sub = ocr_dir / "ocr_json"
            if sub.is_dir() and any(sub.glob("*.json")):
                hint = (
                    f"\n  提示: {ocr_dir} 下无顶层 .json, 但存在 {sub}/\n"
                    f"  请改用: --ocr-dir {sub}"
                )
        raise SystemExit(
            "未找到任何 OCR JSON 与图片的配对, 已中止 (避免空跑加载 vLLM)。\n"
            f"  --ocr-dir={args.ocr_dir or args.ocr_json}\n"
            f"  --image-dir={args.image_dir or args.image}"
            f"{hint}"
        )

    print(f"[info] 共 {len(jobs)} 张图待处理")
    hf_model: Any | None = None
    hf_processor: Any | None = None
    vllm_llm: Any | None = None
    vllm_processor: Any | None = None
    job_precomputed: dict[tuple[Path, Path], dict[int, dict[str, Any]]] = {}

    # vLLM two stages: HF visual-only does OCR-guided merge first; release it, then start vLLM for decoding
    if args.b1_backend == "vllm":
        print("[info] B-1 vllm 阶段 1/2: 仅加载 HF visual → encode+merge …")
        hf_model, hf_processor = _load_hf_visual_only(
            args.model_path, args.dtype, args.b1_hf_device,
        )
        for ocr_json, image in jobs:
            enc = process_one_b1(
                args=args,
                ocr_json_path=ocr_json,
                image_path=image,
                output_dir=output_dir,
                hf_model=hf_model,
                hf_processor=hf_processor,
                vllm_llm=None,
                vllm_processor=None,
                encode_only=True,
            )
            job_precomputed[(ocr_json, image)] = enc["packs_by_batch"]
        del hf_model
        hf_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[info] B-1 vllm 阶段 2/2: 加载 vLLM (enable_mm_embeds) …")
        vllm_llm, vllm_processor = _build_vllm_b1(args)
    else:
        # HF direct inference: load the full VLM, do not start vLLM
        hf_model, hf_processor = _load_hf_vlm(args.model_path, args.dtype, "auto")

    try:
        for idx, (ocr_json, image) in enumerate(jobs, 1):
            print(f"\n=== B-1 [{idx}/{len(jobs)}] {image.name} backend={args.b1_backend} ===")
            try:
                process_one_b1(
                    args=args,
                    ocr_json_path=ocr_json,
                    image_path=image,
                    output_dir=output_dir,
                    hf_model=hf_model,
                    hf_processor=hf_processor,
                    vllm_llm=vllm_llm,
                    vllm_processor=vllm_processor,
                    precomputed_packs=job_precomputed.get((ocr_json, image)),
                )
            except Exception as e:
                traceback.print_exc()
                print(f"[ERROR] {e!r}")
    finally:
        if vllm_llm is not None:
            base._cleanup(vllm_llm, vllm_processor)


if __name__ == "__main__":
    main()
