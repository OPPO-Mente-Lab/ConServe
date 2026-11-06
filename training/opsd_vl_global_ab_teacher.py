#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Build verl OPSD parquet for positive (A) or negative (B) privileged teachers.

A / pos : framed + low-res global + 5-score gold reference prompts
B / neg : framed + low-res global + bad translation + critique (+ optional suggestion)

Example::

  # A
  python training/opsd_vl_global_ab_teacher.py \\
    --mode pos \\
    --annotation-json .../positive_reference.json \\
    --image-list-jsonl .../train_image_list.jsonl \\
    --ocr-dir .../ocr_all \\
    --output-dir .../opsd_ood_agent_pos

  # B
  python training/opsd_vl_global_ab_teacher.py \\
    --mode neg \\
    --annotation-json .../negative_critique.json \\
    ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "inference"


def _to_notebook_path(path: Path, *, mnt_root: Path | None = None, notebook_root: Path | None = None) -> str:
    """Use local absolute paths unless an explicit mount-to-runtime mapping is given."""
    resolved = path.resolve()
    if mnt_root is None and notebook_root is None:
        return str(resolved)
    if mnt_root is None or notebook_root is None:
        raise SystemExit("--mnt-root and --notebook-root must be provided together")
    try:
        relative = resolved.relative_to(mnt_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"path is outside --mnt-root: {resolved}") from exc
    return str(notebook_root / relative)


def _import_modules(code_dir: Path, mode: str):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import SFT_M3_optimized as m3
    from SFT_M3_infer_visual_global import make_global_image

    if mode == "pos":
        from SFT_M3_infer_visual_global_pos_teacher import (
            SYSTEM_PROMPT_POS_TEACHER as sp,
            append_pos_reference_block as append_block,
            build_user_prompt_pos_teacher as build_up,
        )
    else:
        from SFT_M3_infer_visual_global_neg_teacher import (
            SYSTEM_PROMPT_NEG_TEACHER as sp,
            append_neg_critique_block as append_block,
            build_user_prompt_neg_teacher as build_up,
        )
    return m3, make_global_image, sp, build_up, append_block


def _read_image_list(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _index_ocr(ocr_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in sorted(ocr_dir.glob("*.json")):
        stem = p.stem
        for suffix in ("_ocr", ".ocr"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        index[stem] = p
    return index


def _strip_hex_prefix(stem: str) -> str:
    import re

    m = re.match(r"^(?:[0-9a-fA-F]+__)?(.+)$", stem)
    return m.group(1) if m else stem


def _lookup_by_stem(index: dict, stem: str):
    for key in (stem, _strip_hex_prefix(stem)):
        if key in index and index[key]:
            return index[key]
    stripped = _strip_hex_prefix(stem)
    for key, val in index.items():
        if _strip_hex_prefix(key) == stripped and val:
            return val
    return None


def _load_ann_by_stem(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for item in raw.get("images") or []:
        stem = item["stem"]
        by_pid = {}
        for para in item.get("paragraphs") or []:
            by_pid[int(para["paragraph_no"])] = para
        out[stem] = by_pid
    return out


def _batches_for_image(m3, image_bgr, ocr_items, m, crop_pad):
    bbx = m3._load_bbx_mod()
    groups = m3.group_by_paragraph(ocr_items)
    if not groups:
        return {}, []
    padded_by_pid = {}
    candidate_pids = []
    for pid in sorted(groups.keys()):
        g = groups[pid]
        bbox = g.get("paragraph_bbox")
        if bbox is None:
            continue
        pb = bbx._padded_paragraph_bbox(image_bgr, bbox, crop_pad)
        if pb is None:
            continue
        padded_by_pid[pid] = pb
        skip, _ = m3.classify_no_translate(" ".join(g["texts"]).strip())
        if not skip:
            candidate_pids.append(pid)
    ordered = m3._sort_pids_by_box_position(candidate_pids, padded_by_pid)
    return groups, m3._chunk(ordered, m)


def _build_rows(args) -> list[dict[str, Any]]:
    mode = args.mode
    m3, make_global_image, teacher_sp_tpl, build_up, append_block = _import_modules(Path(args.code_dir), mode)
    ann_by_stem = _load_ann_by_stem(Path(args.annotation_json))
    ocr_by_stem = _index_ocr(Path(args.ocr_dir))
    entries = _read_image_list(Path(args.image_list_jsonl))
    # only stems present in annotation (the hex prefix may differ)
    matched = []
    for e in entries:
        ann = _lookup_by_stem(ann_by_stem, e["stem"])
        if ann:
            e = dict(e)
            e["_ann"] = ann
            matched.append(e)
    entries = matched

    out_dir = Path(args.output_dir)
    image_out = out_dir / "images"
    image_out.mkdir(parents=True, exist_ok=True)

    data_source = "opsd_m3_global_pos_teacher" if mode == "pos" else "opsd_m3_global_neg_teacher"
    rows: list[dict[str, Any]] = []
    skipped = {"no_ocr": 0, "unreadable": 0, "no_boxes": 0, "batch_no_ann": 0}

    for entry in entries:
        stem = entry["stem"]
        ann = entry.get("_ann") or _lookup_by_stem(ann_by_stem, stem)
        target_lang = str(entry.get("target_lang") or args.target_lang)
        ocr_path = _lookup_by_stem(ocr_by_stem, stem)
        if ocr_path is None:
            skipped["no_ocr"] += 1
            continue
        image_bgr = cv2.imread(str(entry["image_path"]))
        if image_bgr is None:
            skipped["unreadable"] += 1
            continue
        ocr_raw = json.loads(ocr_path.read_text(encoding="utf-8"))
        ocr_items = ocr_raw.get("data", ocr_raw).get("items", [])
        groups, batches = _batches_for_image(m3, image_bgr, ocr_items, args.m, args.crop_pad)
        if not batches:
            skipped["no_boxes"] += 1
            continue

        batches_info = m3.frame_multi_paragraphs_on_full_image(
            image_bgr, groups, image_out, stem=stem,
            batches=batches, pad=args.crop_pad, save_to_disk=False,
        )
        for b_idx, info in sorted(batches_info.items()):
            pids = info["pids_in_order"]
            hit = [p for p in pids if p in ann]
            if mode == "pos":
                # need reference for majority of batch PIDs
                if len(hit) < max(1, (len(pids) + 1) // 2):
                    skipped["batch_no_ann"] += 1
                    continue
            else:
                # neg: only keep batches that contain at least one critiqued PID
                if not hit:
                    skipped["batch_no_ann"] += 1
                    continue

            framed_pil = info["crop_pil"].convert("RGB")
            global_pil = make_global_image(framed_pil, args.global_max_pixels)
            framed_path = image_out / f"{stem}__b{b_idx:03d}__framed.jpg"
            global_path = image_out / f"{stem}__b{b_idx:03d}__global.jpg"
            framed_pil.save(framed_path, quality=args.jpeg_quality)
            global_pil.save(global_path, quality=args.jpeg_quality)

            student_sp = m3.SYSTEM_PROMPT_MULTI.format(target_lang=target_lang, m=len(pids))
            student_up = m3._build_user_prompt_multi(target_lang, pids)
            teacher_sp = teacher_sp_tpl.format(target_lang=target_lang, m=len(pids))
            teacher_up = append_block(build_up(target_lang, pids), ann, pids)

            rows.append({
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": student_sp},
                    {"role": "user", "content": "<image>\n" + student_up},
                ],
                "images": [{"image": _to_notebook_path(framed_path, mnt_root=args.mnt_root, notebook_root=args.notebook_root), "max_pixels": args.max_pixels}],
                "teacher_prompt": [
                    {"role": "system", "content": teacher_sp},
                    {"role": "user", "content": "<image>\n<image>\n" + teacher_up},
                ],
                "teacher_images": [
                    {"image": _to_notebook_path(framed_path, mnt_root=args.mnt_root, notebook_root=args.notebook_root), "max_pixels": args.max_pixels},
                    {"image": _to_notebook_path(global_path, mnt_root=args.mnt_root, notebook_root=args.notebook_root), "max_pixels": args.global_max_pixels},
                ],
                "ability": "image_translation",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "stem": stem,
                    "source": entry.get("source"),
                    "target_lang": target_lang,
                    "batch_idx": b_idx,
                    "pids": pids,
                    "m": len(pids),
                    "teacher_mode": "positive" if mode == "pos" else "negative",
                    "ann_pids": hit,
                },
            })

    print(f"mode={mode} images={len(entries)} rows={len(rows)} skipped={skipped}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["pos", "neg"], required=True)
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--image-list-jsonl", required=True)
    parser.add_argument("--ocr-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-dir", default=str(CODE_DIR))
    parser.add_argument("--mnt-root", type=Path, help="Local mount root for parquet image paths (use with --notebook-root)")
    parser.add_argument("--notebook-root", type=Path, help="Runtime mount root corresponding to --mnt-root")
    parser.add_argument("--target-lang", default="中文")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--crop-pad", type=int, default=6)
    parser.add_argument("--global-max-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--val-images", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.mnt_root is None) != (args.notebook_root is None):
        parser.error("--mnt-root and --notebook-root must be provided together")

    rows = _build_rows(args)
    if not rows:
        raise SystemExit("没有产出任何样本")

    stems = sorted({r["extra_info"]["stem"] for r in rows})
    random.Random(args.seed).shuffle(stems)
    n_val = max(0, min(args.val_images, len(stems) - 1)) if len(stems) > 1 else 0
    val_stems = set(stems[:n_val])
    train_rows = [r for r in rows if r["extra_info"]["stem"] not in val_stems]
    val_rows = [r for r in rows if r["extra_info"]["stem"] in val_stems]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    stats = {
        "teacher_mode": args.mode,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_images": len(stems) - len(val_stems),
        "val_images": len(val_stems),
        "annotation_json": str(args.annotation_json),
        "train_parquet": _to_notebook_path(train_path, mnt_root=args.mnt_root, notebook_root=args.notebook_root),
        "val_parquet": _to_notebook_path(val_path, mnt_root=args.mnt_root, notebook_root=args.notebook_root),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
