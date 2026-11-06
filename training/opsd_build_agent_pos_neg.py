#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Build positive / negative teacher annotations from agent-eval results.

Positive (A): images with text_score_5 == 5 → gold translated_text per pair.
Negative (B): images with score < 5 → bad translation + evidence + optional suggestion.

Example::

  python training/opsd_build_agent_pos_neg.py \\
    --results-jsonl /path/to/data \\
    --vlm-dir /path/to/data \\
    --output-dir /path/to/data
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_PAIR_RE = re.compile(r":p(\d+)$")


def _pair_pid(pair_id: str) -> int | None:
    m = _PAIR_RE.search(pair_id or "")
    return int(m.group(1)) if m else None


def _norm(s: Any) -> str:
    return " ".join(str(s or "").split()).strip()


def _strip_hex_prefix(stem: str) -> str:
    m = re.match(r"^(?:[0-9a-fA-F]+__)?(.+)$", stem)
    return m.group(1) if m else stem


def _load_vlm_para_map(vlm_dir: Path, stem: str) -> dict[int, dict[str, Any]]:
    candidates = [
        vlm_dir / f"{stem}__vlm.json",
        vlm_dir / f"{_strip_hex_prefix(stem)}__vlm.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        hits = list(vlm_dir.glob(f"*{_strip_hex_prefix(stem)}__vlm.json"))
        path = hits[0] if hits else None
    if path is None or not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for para in raw.get("paragraphs") or []:
        try:
            pid = int(para["paragraph_no"])
        except (KeyError, TypeError, ValueError):
            continue
        vlm = para.get("vlm") or {}
        out[pid] = {
            "paragraph_no": pid,
            "ocr_text": para.get("ocr_text", ""),
            "prefiltered": bool(para.get("prefiltered", False)),
            "should_translate": vlm.get("should_translate", para.get("should_translate")),
            "original_text": vlm.get("original_text", para.get("original_text", "")),
            "translated_text": vlm.get("translated_text", para.get("translated_text", "")),
        }
    return out


def _errors_for_unit(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only 3 fields per error: erroneous fragment / reason / suggested fix."""
    errs = unit.get("final_errors") or unit.get("refined_errors") or unit.get("candidate_errors") or []
    out = []
    for e in errs:
        out.append({
            "target_fragment": e.get("target_fragment"),  # erroneous field/fragment
            "evidence": e.get("evidence"),                # error reason
            "suggestion": e.get("suggestion"),            # suggested fix (may be empty)
        })
    return out


def build(results_jsonl: Path, vlm_dir: Path) -> tuple[dict, dict]:
    return build_many([(results_jsonl, vlm_dir)])


def build_many(sources: list[tuple[Path, Path]]) -> tuple[dict, dict]:
    pos_images: list[dict[str, Any]] = []
    neg_images: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_paths: list[str] = []

    for results_jsonl, vlm_dir in sources:
        source_paths.append(str(results_jsonl))
        for line in Path(results_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stem = row["task_id"]
            key = _strip_hex_prefix(stem)
            if key in seen:
                continue
            seen.add(key)
            scores = row.get("scores") or {}
            if scores.get("text_score_5") is None:
                continue
            score = float(scores["text_score_5"])
            tr = row.get("text_result") or {}
            vlm_map = _load_vlm_para_map(vlm_dir, stem)

            if score >= 5.0:
                paras = []
                for pair in tr.get("text_pairs") or []:
                    pid = _pair_pid(pair.get("pair_id", ""))
                    if pid is None:
                        continue
                    src = _norm(pair.get("source_text"))
                    tgt = _norm(pair.get("translated_text"))
                    vlm = vlm_map.get(pid, {})
                    st = bool(tgt) if tgt != "" else bool(vlm.get("should_translate", False))
                    # empty translation → treat as no-translate when agent left it empty
                    if not tgt:
                        st = False
                    paras.append({
                        "paragraph_no": pid,
                        "should_translate": st,
                        "original_text": src or _norm(vlm.get("original_text")),
                        "translated_text": tgt if st else "",
                        "ocr_text": vlm.get("ocr_text", ""),
                    })
                if paras:
                    pos_images.append({
                        "stem": stem,
                        "text_score_5": score,
                        "teacher_mode": "positive",
                        "paragraphs": paras,
                    })
                continue

            # negative: only paragraphs that have errors
            unit_by_pid: dict[int, dict[str, Any]] = {}
            for unit in tr.get("unit_results") or []:
                pid = _pair_pid(unit.get("pair_id", ""))
                if pid is None:
                    continue
                errs = _errors_for_unit(unit)
                if not errs:
                    continue
                unit_by_pid[pid] = {
                    "paragraph_no": pid,
                    "original_text": _norm(unit.get("source_text")),
                    "translated_text": _norm(unit.get("translated_text")),  # erroneous translation (negative example)
                    "errors": errs,
                }
            # also catch final_errors whose pair was not in unit_results
            for e in tr.get("final_errors") or []:
                pid = _pair_pid(e.get("pair_id", ""))
                if pid is None or pid in unit_by_pid:
                    continue
                pair = next((p for p in (tr.get("text_pairs") or []) if p.get("pair_id") == e.get("pair_id")), None)
                unit_by_pid[pid] = {
                    "paragraph_no": pid,
                    "original_text": _norm((pair or {}).get("source_text")),
                    "translated_text": _norm((pair or {}).get("translated_text")),
                    "errors": _errors_for_unit({"final_errors": [e]}),
                }

            if unit_by_pid:
                neg_images.append({
                    "stem": stem,
                    "text_score_5": score,
                    "teacher_mode": "negative",
                    "paragraphs": [unit_by_pid[k] for k in sorted(unit_by_pid)],
                })

    pos = {
        "model": "agent_positive_teacher",
        "source": source_paths,
        "num_images": len(pos_images),
        "note": "text_score_5==5 的译文作为 A 正教师标准答案",
        "images": pos_images,
    }
    neg = {
        "model": "agent_negative_teacher",
        "source": source_paths,
        "num_images": len(neg_images),
        "note": "非满分图：错误译文 + errors[{target_fragment, evidence, suggestion}]",
        "images": neg_images,
    }
    return pos, neg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-jsonl", action="append", required=True)
    parser.add_argument("--vlm-dir", action="append", required=True, help="含 stem__vlm.json；与 --results-jsonl 一一对应")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if len(args.results_jsonl) != len(args.vlm_dir):
        raise SystemExit("--results-jsonl 与 --vlm-dir 数量必须一致")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = list(zip((Path(p) for p in args.results_jsonl), (Path(p) for p in args.vlm_dir)))
    pos, neg = build_many(sources)
    pos_path = out_dir / "positive_reference.json"
    neg_path = out_dir / "negative_critique.json"
    pos_path.write_text(json.dumps(pos, ensure_ascii=False, indent=2), encoding="utf-8")
    neg_path.write_text(json.dumps(neg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "positive_images": pos["num_images"],
        "negative_images": neg["num_images"],
        "positive_json": str(pos_path),
        "negative_json": str(neg_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
