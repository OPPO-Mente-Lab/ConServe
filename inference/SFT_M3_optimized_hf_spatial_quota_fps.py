#!/usr/bin/env python3
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Three-field SFT M3 inference with HF Spatial-Quota FPS visual compression.

This runner deliberately reuses ``SFT_M3_optimized`` for the rendered image,
student prompt, JSON parser, and result schema.  Only the visual path changes:
HF encodes and prunes image embeddings first, then vLLM receives those embeds
through ``enable_mm_embeds`` for language-model decoding.

Before each compressed request, the runner checks that the HF preprocessing
grid is exactly the grid produced by the vLLM processor with the same
``--vllm-max-pixels``.  This makes the uncompressed visual-token baseline
identical to native vLLM preprocessing instead of merely approximately so.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = Path(os.environ.get("CONSERVE_COMPRESSION_DIR", THIS_DIR.parent / "compression"))
for path in (THIS_DIR, CLUSTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import SFT_M3_optimized as base  # noqa: E402
import run_M3_multi_box_b1 as b1run  # noqa: E402
from ogstb_b1_merge import vllm_image_token_count  # noqa: E402
from ogstb_cluster_merge import (  # noqa: E402
    encode_merge_batch_cluster,
    resize_image_for_vllm_budget,
    resize_image_to_visual_token_target,
)
from run_M3_multi_box_cluster import _load_hf_visual_only_compat  # noqa: E402


_ORIGINAL_FRAME = base.frame_multi_paragraphs_on_full_image
_ORIGINAL_BUILD = base.build_vlm_inputs_multi
_ACTIVE_BATCHES: dict[tuple[str, int], dict[str, Any]] = {}
_CURRENT_STEM: str | None = None


def _frame_with_compression_metadata(*args: Any, **kwargs: Any) -> dict[int, dict[str, Any]]:
    global _CURRENT_STEM
    batches = _ORIGINAL_FRAME(*args, **kwargs)
    stem = str(kwargs["stem"])
    _CURRENT_STEM = stem
    for batch_idx, info in batches.items():
        _ACTIVE_BATCHES[(stem, int(batch_idx))] = info
    return batches


def _build_with_compression_metadata(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], list[tuple[int, list[int]]], list[dict[str, Any]]]:
    inputs, order, prompt_meta = _ORIGINAL_BUILD(*args, **kwargs)
    if _CURRENT_STEM is None:
        raise RuntimeError("missing active image stem for HF FPS cache")
    for request, (batch_idx, _) in zip(inputs, order):
        stem = _CURRENT_STEM
        info = _ACTIVE_BATCHES[(stem, int(batch_idx))]
        request["_hf_fps_key"] = (stem, int(batch_idx))
        request["_hf_fps_boxes"] = [tuple(box) for box in info["padded_bbox_by_pid"].values()]
        request["_hf_fps_orig_hw"] = (int(info["crop_w"]), int(info["crop_h"]))
    return inputs, order, prompt_meta


base.frame_multi_paragraphs_on_full_image = _frame_with_compression_metadata
base.build_vlm_inputs_multi = _build_with_compression_metadata


def _cache_path(root: Path, key: tuple[str, int]) -> Path:
    stem, batch_idx = key
    return root / stem / f"batch{batch_idx:03d}.pt"


def _image_grid(
    processor: Any,
    pil: Any,
    max_pixels: int | None,
    min_pixels: int | None,
    target_visual_tokens: int | None,
) -> torch.Tensor:
    """Use the processor's real image path to derive its merger-grid token count."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("vLLM processor lacks image_processor")
    patch_size = int(getattr(image_processor, "patch_size", 16))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    if target_visual_tokens is None:
        resized_pil, _ = resize_image_for_vllm_budget(
            pil, max_pixels=max_pixels, min_pixels=min_pixels,
            patch_size=patch_size, merge_size=merge_size,
        )
    else:
        resized_pil, _ = resize_image_to_visual_token_target(
            pil, target_tokens=target_visual_tokens,
            patch_size=patch_size, merge_size=merge_size,
        )
    image_kwargs: dict[str, int] = {}
    if max_pixels is not None:
        image_kwargs["max_pixels"] = int(max_pixels)
    if min_pixels is not None:
        image_kwargs["min_pixels"] = int(min_pixels)
    grid = image_processor(
        images=[resized_pil], return_tensors="pt", **image_kwargs,
    ).get("image_grid_thw")
    if grid is None or grid.shape != (1, 3):
        raise RuntimeError("processor did not return one image_grid_thw row")
    return grid[0].detach().cpu().to(torch.int64)


class _EncodeCollector:
    def __init__(self, args: argparse.Namespace, model: Any, processor: Any, cache_root: Path) -> None:
        self.args, self.model, self.processor, self.cache_root = args, model, processor, cache_root

    def run(self, inputs: list[dict[str, Any]], n_boxes: list[int], *_: Any, **__: Any) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        for request in inputs:
            key = request["_hf_fps_key"]
            out = _cache_path(self.cache_root, key)
            out.parent.mkdir(parents=True, exist_ok=True)
            pack = encode_merge_batch_cluster(
                self.model, self.processor, request["multi_modal_data"]["image"], request["_hf_fps_boxes"],
                orig_hw=request["_hf_fps_orig_hw"], device=self.args.hf_device,
                method=self.args.cluster_method, keep_ratio=self.args.keep_ratio,
                target_total_keep_ratio=self.args.target_total_keep_ratio,
                prototype=self.args.prototype, metric=self.args.cluster_metric,
                spatial_weight=self.args.spatial_weight, quota_grid_rows=self.args.quota_grid_rows,
                quota_grid_cols=self.args.quota_grid_cols, quota_min_per_cell=self.args.quota_min_per_cell,
                quota_power=self.args.quota_power, quota_diversity_weight=self.args.quota_diversity_weight,
                seed=self.args.seed, fg_dilate=self.args.fg_dilate,
                max_pixels=self.args.vllm_max_pixels, min_pixels=self.args.vllm_min_pixels,
                target_visual_tokens=self.args.hf_target_visual_tokens,
                return_visualization=getattr(self.args, "save_visualization_cache", False),
            )
            torch.save(pack, out)
            meta = pack["meta"]
            print(
                f"[HF FPS] {key[0]} batch={key[1]} N={meta['n_tokens_orig']} "
                f"F={meta['n_fg_orig']} M={meta['n_tokens_merged']} "
                f"ratio={meta['compress_ratio']} input={meta['resized_input_hw']}", flush=True,
            )
        # Base process_one writes temporary output in the encode pass.  It is
        # overwritten by the decode pass and never presented as a result.
        fake = [{"raw_text": "[]", "n_prompt": 0, "n_completion": 0} for _ in inputs]
        return fake, time.perf_counter() - started


class _CompressedVllm:
    def __init__(self, args: argparse.Namespace, engine: Any, processor: Any, cache_root: Path) -> None:
        self.args, self.engine, self.processor, self.cache_root = args, engine, processor, cache_root

    def run(self, inputs: list[dict[str, Any]], n_boxes: list[int], per_box_budget: int, overall_cap: int, *, temperature: float = 0.0, num_samples: int = 1) -> tuple[list[dict[str, Any]], float]:
        from vllm import SamplingParams

        requests: list[dict[str, Any]] = []
        for request in inputs:
            pack = torch.load(_cache_path(self.cache_root, request["_hf_fps_key"]), map_location="cpu", weights_only=False)
            hf_grid = torch.tensor(pack["meta"]["grid_orig"], dtype=torch.int64)
            vllm_grid = _image_grid(
                self.processor, request["multi_modal_data"]["image"],
                self.args.vllm_max_pixels, self.args.vllm_min_pixels,
                self.args.hf_target_visual_tokens,
            )
            if not torch.equal(hf_grid, vllm_grid):
                raise RuntimeError(
                    f"visual preprocessing mismatch for {request['_hf_fps_key']}: "
                    f"HF grid={hf_grid.tolist()} vLLM grid={vllm_grid.tolist()}. "
                    "Refusing an unfair comparison."
                )
            expected_n = vllm_image_token_count(vllm_grid, merge_size=2)
            actual_n = int(pack["meta"]["n_tokens_orig"])
            if expected_n != actual_n:
                raise RuntimeError(
                    f"token mismatch for {request['_hf_fps_key']}: HF N={actual_n}, vLLM N={expected_n}"
                )
            print(f"[token-check] {request['_hf_fps_key'][0]} batch={request['_hf_fps_key'][1]} HF_N=vLLM_N={actual_n}", flush=True)
            audit = {
                "stem": request["_hf_fps_key"][0],
                "batch_idx": request["_hf_fps_key"][1],
                "hf_grid_thw": hf_grid.tolist(),
                "vllm_grid_thw": vllm_grid.tolist(),
                "uncompressed_visual_tokens": actual_n,
                "compressed_visual_tokens": int(pack["merged_embeds"].shape[0]),
                "compression_meta": pack["meta"],
            }
            with (self.cache_root / "token_audit.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            requests.append(b1run._build_vllm_embed_request(
                request["prompt"], pack["merged_embeds"], pack["merged_grid_thw"],
            ))

        caps = [min(overall_cap, max(per_box_budget, per_box_budget * n)) for n in n_boxes]
        started = time.perf_counter()
        params: Any = SamplingParams(
            max_tokens=caps[0], temperature=temperature, n=num_samples,
            stop=base._STOP_STRINGS, include_stop_str_in_output=True,
        ) if len(set(caps)) == 1 else [
            SamplingParams(max_tokens=cap, temperature=temperature, n=num_samples,
                           stop=base._STOP_STRINGS, include_stop_str_in_output=True)
            for cap in caps
        ]
        outputs = self.engine.generate(requests, sampling_params=params, use_tqdm=False)
        results: list[dict[str, Any]] = []
        for output in outputs:
            samples = [{
                "raw_text": base._strip_thinking((row.text or "").strip()),
                "n_prompt": len(output.prompt_token_ids or []),
                "n_completion": len(row.token_ids or []),
            } for row in output.outputs]
            if not samples:
                samples = [{"raw_text": "", "n_prompt": len(output.prompt_token_ids or []), "n_completion": 0}]
            result: dict[str, Any] = {"samples": samples}
            if len(samples) == 1:
                result.update(samples[0])
            results.append(result)
        return results, time.perf_counter() - started


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT 三字段 prompt + HF Spatial-Quota FPS + vLLM decode")
    parser.add_argument("--ocr-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--m", dest="m_per_batch", type=int, default=3)
    parser.add_argument("--target-lang", default="中文")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--crop-pad", type=int, default=6)
    parser.add_argument("--per-box-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-save-crops", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--vllm-max-pixels", type=int, default=1048576)
    parser.add_argument("--vllm-min-pixels", type=int)
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=64)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-disable-prefix-caching", action="store_true")
    parser.add_argument("--hf-device", default="cuda")
    parser.add_argument(
        "--hf-target-visual-tokens", type=int, default=None,
        help="force this exact pre-FPS HF merger-token count; use 990 to match native vLLM baseline",
    )
    parser.add_argument(
        "--cluster-method", choices=["spatial_quota_fps", "fps"],
        default="spatial_quota_fps",
        help="background selector; fps is the w/o Spatial Quota ablation",
    )
    parser.add_argument(
        "--prototype", choices=["medoid", "centroid"], default="medoid",
        help="medoid keeps real tokens (pruning); centroid pools each assigned cluster",
    )
    parser.add_argument("--keep-ratio", type=float, required=True, help="background FPS keep ratio")
    parser.add_argument("--target-total-keep-ratio", type=float, default=None)
    parser.add_argument("--cluster-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--spatial-weight", type=float, default=0.25)
    parser.add_argument("--quota-grid-rows", type=int, default=4)
    parser.add_argument("--quota-grid-cols", type=int, default=4)
    parser.add_argument("--quota-min-per-cell", type=int, default=1)
    parser.add_argument("--quota-power", type=float, default=0.5)
    parser.add_argument("--quota-diversity-weight", type=float, default=1.0)
    parser.add_argument("--fg-dilate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-visualization-cache", action="store_true",
        help="store dense visual embeddings and SQ-FPS provenance in each HF cache pack",
    )
    return parser.parse_args()


def _run_base_process(args: argparse.Namespace, runner: Any, processor: Any, jobs: list[tuple[Path, Path, str, str]], output_dir: Path) -> None:
    original_run = base.run_vlm_batches_multi
    # ``process_one`` calls the original helper as
    # ``run_vlm_batches_multi(llm, inputs, ...)``.  The backend object itself
    # is only a placeholder here; our runner already owns the HF/vLLM engine.
    def _run_adapter(
        _unused_llm: Any,
        inputs: list[dict[str, Any]],
        n_boxes: list[int],
        per_box_budget: int,
        overall_cap: int,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], float]:
        return runner.run(inputs, n_boxes, per_box_budget, overall_cap, **kwargs)

    base.run_vlm_batches_multi = _run_adapter
    try:
        for idx, (ocr_json, image, target_lang, target_code) in enumerate(jobs, 1):
            print(f"\n=== [{idx}/{len(jobs)}] {image.name} ===", flush=True)
            base.process_one(
                ocr_json_path=ocr_json, image_path=image, output_dir=output_dir, llm=runner,
                processor=processor, target_lang=target_lang,
                per_box_max_new_tokens=args.per_box_max_new_tokens,
                overall_max_new_tokens=args.max_new_tokens, crop_pad=args.crop_pad,
                m_per_batch=args.m_per_batch, save_crops=not args.no_save_crops,
                write_profile=not args.no_profile, tgt_lang_code=target_code or None,
                temperature=args.temperature, num_samples=args.num_samples,
            )
    finally:
        base.run_vlm_batches_multi = original_run


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.keep_ratio <= 1.0:
        raise SystemExit("--keep-ratio must be in (0, 1]")
    if args.target_total_keep_ratio is not None and not 0.0 < args.target_total_keep_ratio <= 1.0:
        raise SystemExit("--target-total-keep-ratio must be in (0, 1]")
    if args.num_samples < 1 or args.m_per_batch < 1:
        raise SystemExit("--num-samples and --m must be >= 1")
    if args.hf_target_visual_tokens is not None and args.hf_target_visual_tokens < 1:
        raise SystemExit("--hf-target-visual-tokens must be >= 1")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / ".hf_spatial_quota_fps_cache"
    jobs = base.build_jobs_from_dirs(
        Path(args.ocr_dir).resolve(), Path(args.image_dir).resolve(), None, args.limit,
        default_target_lang=args.target_lang,
    )
    if not jobs:
        raise SystemExit("no matching OCR/image pairs")

    print(
        f"[phase 1/2] HF encode + {args.cluster_method}; "
        f"prototype={args.prototype}; images={len(jobs)}",
        flush=True,
    )
    hf_model, hf_processor = _load_hf_visual_only_compat(
        args.model_path, args.dtype, args.hf_device,
        max_pixels=args.vllm_max_pixels, min_pixels=args.vllm_min_pixels,
    )
    _run_base_process(args, _EncodeCollector(args, hf_model, hf_processor, cache_root), hf_processor, jobs, output_dir)
    del hf_model, hf_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[phase 2/2] vLLM decode with precomputed compressed embeddings", flush=True)
    audit_path = cache_root / "token_audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    engine, processor = b1run._build_vllm_b1(args)
    try:
        _run_base_process(args, _CompressedVllm(args, engine, processor, cache_root), processor, jobs, output_dir)
    finally:
        base._cleanup(engine, processor)


if __name__ == "__main__":
    main()
