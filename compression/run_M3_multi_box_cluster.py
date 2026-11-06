#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""M3 multi-box inference with deployable background-token clustering.

For real compressed inference use ``--b1-backend vllm``.  The inherited HF
backend is kept only as an encode/metadata smoke test because the current B-1
HF path does not inject externally merged embeddings into ``generate``.
"""
from __future__ import annotations

import argparse
import functools
import gc
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for item in (REPO_ROOT, THIS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_M3_multi_box_b1 as b1run  # noqa: E402
from ogstb_cluster_merge import SUPPORTED_METHODS, encode_merge_batch_cluster  # noqa: E402

base = b1run.base


def _load_hf_visual_only_compat(
    model_path: str,
    dtype: str,
    device: str,
    *,
    max_pixels: int | None,
    min_pixels: int | None,
) -> tuple[Any, Any]:
    """Support both old and new ``run_M3_multi_box_b1`` loader signatures."""
    parameters = inspect.signature(b1run._load_hf_visual_only).parameters
    if "max_pixels" in parameters:
        return b1run._load_hf_visual_only(
            model_path,
            dtype,
            device,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )

    print(
        "[compat] legacy visual-only loader detected (B-1 adapter); "
        "applying pixel limits after loading the vision tower",
    )
    model, processor = b1run._load_hf_visual_only(model_path, dtype, device)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("processor lacks image_processor; cannot apply pixel limits")
    if max_pixels is not None:
        image_processor.max_pixels = int(max_pixels)
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            size["longest_edge"] = int(max_pixels)
    if min_pixels is not None:
        image_processor.min_pixels = int(min_pixels)
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            size["shortest_edge"] = int(min_pixels)
    print(
        "[compat] image processor limits: "
        f"min_pixels={getattr(image_processor, 'min_pixels', None)} "
        f"max_pixels={getattr(image_processor, 'max_pixels', None)}"
    )
    return model, processor


def process_one_cluster(
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
    result = b1run.process_one_b1(
        args=args,
        ocr_json_path=ocr_json_path,
        image_path=image_path,
        output_dir=output_dir,
        hf_model=hf_model,
        hf_processor=hf_processor,
        vllm_llm=vllm_llm,
        vllm_processor=vllm_processor,
        encode_only=encode_only,
        precomputed_packs=precomputed_packs,
    )
    if encode_only or "packs_by_batch" in result:
        return result
    stem = result.get("stem", image_path.stem)
    config = {
        "method": args.cluster_method,
        "keep_ratio": args.keep_ratio,
        "n_clusters": args.n_clusters,
        "spatial_weight": args.spatial_weight,
        "metric": args.cluster_metric,
        "prototype": args.prototype,
        "max_iter": args.cluster_max_iter,
        "minibatch_size": args.minibatch_size,
        "local_radius": args.local_radius,
        "quota_grid_rows": args.quota_grid_rows,
        "quota_grid_cols": args.quota_grid_cols,
        "quota_min_per_cell": args.quota_min_per_cell,
        "quota_power": args.quota_power,
        "quota_diversity_weight": args.quota_diversity_weight,
        "holov_saliency_weight": args.holov_saliency_weight,
        "selector_important_ratio": args.selector_important_ratio,
        "selector_relevance_weight": args.selector_relevance_weight,
        "selector_diversity_weight": args.selector_diversity_weight,
        "weighted_alpha": args.weighted_alpha,
        "weighted_beta": args.weighted_beta,
        "weighted_temperature": args.weighted_temperature,
        "seed": args.seed,
        "fg_dilate": args.fg_dilate,
        "merge_mode": f"cluster_{args.cluster_method}",
    }
    result["ogstb_method"] = f"cluster_{args.cluster_method}"
    result["ogstb_cluster_config"] = config
    profile = result.get("profile")
    if isinstance(profile, dict):
        profile["backend"] = f"cluster_{args.cluster_method}_{args.b1_backend}"
        profile["ogstb_cluster_config"] = config
        (output_dir / f"{stem}__profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    dump = {key: value for key, value in result.items() if key != "profile"}
    (output_dir / f"{stem}__vlm_cluster.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    b1_file = output_dir / f"{stem}__vlm_b1.json"
    if b1_file.exists():
        b1_file.unlink()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3 multi-box + background token clustering")
    parser.add_argument("--ocr-json")
    parser.add_argument("--image")
    parser.add_argument("--ocr-dir")
    parser.add_argument("--image-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-lang", default="中文")
    parser.add_argument("--m", dest="m_per_batch", type=int, default=3)
    parser.add_argument("--crop-pad", type=int, default=6)
    parser.add_argument("--per-box-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-save-crops", action="store_true")
    parser.add_argument("--dtype", default="fp16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument(
        "--vllm-max-pixels", type=int, default=1024 * 1024,
        help="image-pixel cap; default matches the M3 SFT preprocessing setting",
    )
    parser.add_argument("--vllm-min-pixels", type=int)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=64)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-disable-prefix-caching", action="store_true")
    parser.add_argument(
        "--vllm-mm-embed-device", choices=["auto", "cuda", "cpu"], default="auto",
        help="external image_embeds payload device; auto retries CPU on CUDA-to-NumPy errors",
    )
    parser.add_argument("--b1-backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--bg-merge-factor", type=int, default=4,
                        help="B-1 compatibility argument; clustering uses keep-ratio/n-clusters")
    parser.add_argument("--cluster-method", choices=SUPPORTED_METHODS, default="spatial_kmeans")
    parser.add_argument("--keep-ratio", type=float, default=0.10,
                        help="fraction of compressible background representatives")
    parser.add_argument(
        "--target-total-keep-ratio", type=float,
        help="target fraction of all visual tokens after preserving foreground; "
             "derives a per-batch background keep ratio",
    )
    parser.add_argument("--n-clusters", type=int,
                        help="fixed background representative count; overrides keep-ratio")
    parser.add_argument("--spatial-weight", type=float, default=0.25,
                        help="weight of normalized y/x in spatial_kmeans")
    parser.add_argument("--cluster-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--prototype", choices=["centroid", "medoid"], default="centroid")
    parser.add_argument("--cluster-max-iter", type=int, default=20)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--local-radius", type=float, default=2.0,
                        help="local_kmeans radius in expected-cluster-spacing units")
    parser.add_argument("--quota-grid-rows", type=int, default=4,
                        help="spatial_quota_fps: number of horizontal crop bands")
    parser.add_argument("--quota-grid-cols", type=int, default=4,
                        help="spatial_quota_fps: number of vertical crop bands")
    parser.add_argument("--quota-min-per-cell", type=int, default=1,
                        help="spatial_quota_fps: minimum representatives per non-empty cell")
    parser.add_argument("--quota-power", type=float, default=0.5,
                        help="spatial_quota_fps: exponent for cell token-count allocation")
    parser.add_argument("--quota-diversity-weight", type=float, default=1.0,
                        help="spatial_quota_fps: strength of semantic-diversity allocation")
    parser.add_argument("--holov-saliency-weight", type=float, default=1.0,
                        help="holov_adapt: crop saliency-proxy weight")
    parser.add_argument("--selector-important-ratio", type=float, default=0.5,
                        help="vispruner_adapt/crisp_mb: first-stage budget fraction")
    parser.add_argument("--selector-relevance-weight", type=float, default=0.5,
                        help="crisp_mb: target-box relevance weight")
    parser.add_argument("--selector-diversity-weight", type=float, default=1.0,
                        help="crisp_mb: semantic/spatial diversity weight")
    parser.add_argument("--weighted-alpha", type=float, default=0.0,
                        help="weighted baseline: target-box relevance coefficient")
    parser.add_argument("--weighted-beta", type=float, default=1.0,
                        help="weighted baseline: within-chunk deviation coefficient")
    parser.add_argument("--weighted-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fg-dilate", type=int, default=0)
    parser.add_argument("--cluster-debug-dir")
    parser.add_argument("--b1-embed-cache-dir")
    parser.add_argument("--b1-hf-device", default="cuda")
    parser.add_argument("--b1-vllm-sequential", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_clusters is None and not 0.0 < args.keep_ratio <= 1.0:
        raise SystemExit("--keep-ratio must be in (0,1]")
    if args.n_clusters is not None and args.n_clusters < 1:
        raise SystemExit("--n-clusters must be >= 1")
    if args.target_total_keep_ratio is not None:
        if not 0.0 < args.target_total_keep_ratio <= 1.0:
            raise SystemExit("--target-total-keep-ratio must be in (0,1]")
        if args.n_clusters is not None:
            raise SystemExit("--target-total-keep-ratio cannot be combined with --n-clusters")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    b1run.encode_merge_batch = functools.partial(
        encode_merge_batch_cluster,
        method=args.cluster_method,
        keep_ratio=args.keep_ratio,
        target_total_keep_ratio=args.target_total_keep_ratio,
        n_clusters=args.n_clusters,
        max_pixels=args.vllm_max_pixels,
        min_pixels=args.vllm_min_pixels,
        spatial_weight=args.spatial_weight,
        metric=args.cluster_metric,
        prototype=args.prototype,
        max_iter=args.cluster_max_iter,
        minibatch_size=args.minibatch_size,
        local_radius=args.local_radius,
        quota_grid_rows=args.quota_grid_rows,
        quota_grid_cols=args.quota_grid_cols,
        quota_min_per_cell=args.quota_min_per_cell,
        quota_power=args.quota_power,
        quota_diversity_weight=args.quota_diversity_weight,
        holov_saliency_weight=args.holov_saliency_weight,
        selector_important_ratio=args.selector_important_ratio,
        selector_relevance_weight=args.selector_relevance_weight,
        selector_diversity_weight=args.selector_diversity_weight,
        seed=args.seed,
        weighted_alpha=args.weighted_alpha,
        weighted_beta=args.weighted_beta,
        weighted_temperature=args.weighted_temperature,
        fg_dilate=args.fg_dilate,
        debug_dir=args.cluster_debug_dir,
    )

    jobs: list[tuple[Path, Path]] = []
    if args.ocr_json and args.image:
        jobs.append((Path(args.ocr_json).resolve(), Path(args.image).resolve()))
    elif args.ocr_dir and args.image_dir:
        ocr_dir, image_dir = Path(args.ocr_dir).resolve(), Path(args.image_dir).resolve()
        for ocr_json in base.list_ocr_jsons(ocr_dir, args.limit):
            image = base._find_image_for_ocr(ocr_json, image_dir)
            if image:
                jobs.append((ocr_json, image))
    else:
        raise SystemExit("need --ocr-json+--image or --ocr-dir+--image-dir")
    if not jobs:
        raise SystemExit("no OCR/image pairs found")

    print(
        f"[cluster] images={len(jobs)} method={args.cluster_method} "
        f"ratio={args.keep_ratio} k={args.n_clusters} backend={args.b1_backend}"
    )
    hf_model = hf_processor = vllm_llm = vllm_processor = None
    precomputed: dict[tuple[Path, Path], dict[int, dict[str, Any]]] = {}
    if args.b1_backend == "vllm":
        hf_model, hf_processor = _load_hf_visual_only_compat(
            args.model_path, args.dtype, args.b1_hf_device,
            max_pixels=args.vllm_max_pixels,
            min_pixels=args.vllm_min_pixels,
        )
        for ocr_json, image in jobs:
            encoded = process_one_cluster(
                args=args, ocr_json_path=ocr_json, image_path=image,
                output_dir=output_dir, hf_model=hf_model, hf_processor=hf_processor,
                vllm_llm=None, vllm_processor=None, encode_only=True,
            )
            precomputed[(ocr_json, image)] = encoded["packs_by_batch"]
        del hf_model
        hf_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vllm_llm, vllm_processor = b1run._build_vllm_b1(args)
    else:
        print("[warn] HF backend does not inject merged embeds; use it only for encode smoke tests")
        hf_model, hf_processor = b1run._load_hf_vlm(args.model_path, args.dtype, "auto")

    try:
        for index, (ocr_json, image) in enumerate(jobs, 1):
            print(f"\n=== cluster [{index}/{len(jobs)}] {image.name} ===")
            try:
                process_one_cluster(
                    args=args, ocr_json_path=ocr_json, image_path=image,
                    output_dir=output_dir, hf_model=hf_model, hf_processor=hf_processor,
                    vllm_llm=vllm_llm, vllm_processor=vllm_processor,
                    precomputed_packs=precomputed.get((ocr_json, image)),
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                print(f"[ERROR] {exc!r}")
    finally:
        if vllm_llm is not None:
            base._cleanup(vllm_llm, vllm_processor)


if __name__ == "__main__":
    main()
