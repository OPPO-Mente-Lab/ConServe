# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional


def _strip_hex(stem: str) -> str:
    m = re.match(r"^(?:[0-9a-fA-F]+__)?(.+)$", stem)
    return m.group(1) if m else stem


def load_boxes_from_ocr_json(path: Path) -> list[tuple[int, int, int, int]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("data", raw).get("items", raw.get("items", []))
    boxes: list[tuple[int, int, int, int]] = []
    seen = set()

    def poly_to_xyxy(poly):
        if not poly:
            return None
        xs, ys = [], []
        for pt in poly:
            if isinstance(pt, dict):
                xs.append(float(pt.get("x", 0)))
                ys.append(float(pt.get("y", 0)))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
        if not xs:
            return None
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    by_pid: dict[int, list[tuple[int, int, int, int]]] = {}
    if isinstance(raw, dict) and "paragraphs" in raw:
        for para in raw["paragraphs"] or []:
            bb = para.get("padded_bbox") or para.get("paragraph_bbox") or para.get("bbox")
            if bb and len(bb) == 4:
                boxes.append(tuple(int(x) for x in bb))
        if boxes:
            return boxes

    for it in items or []:
        bb = it.get("paragraph_bbox") or it.get("padded_bbox") or it.get("bbox") or it.get("box")
        if not bb or len(bb) != 4:
            bb = poly_to_xyxy(it.get("paragraph_polygon") or it.get("polygon"))
        if not bb or len(bb) != 4:
            continue
        key = tuple(int(x) for x in bb)
        pid = it.get("paragraphNo")
        if pid is not None:
            by_pid.setdefault(int(pid), []).append(key)
        elif key not in seen:
            seen.add(key)
            boxes.append(key)
    if by_pid:
        for pid in sorted(by_pid):
            xs1, ys1, xs2, ys2 = zip(*by_pid[pid])
            key = (min(xs1), min(ys1), max(xs2), max(ys2))
            if key not in seen:
                seen.add(key)
                boxes.append(key)
    return boxes


def load_boxes_from_meta_json(path: Path) -> list[tuple[int, int, int, int]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    boxes: list[tuple[int, int, int, int]] = []
    for b in raw.get("boxes") or []:
        bb = b.get("padded_bbox") or b.get("paragraph_bbox") or b.get("bbox")
        if bb and len(bb) == 4:
            boxes.append(tuple(int(x) for x in bb))
    return boxes


def index_ocr_dir(ocr_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in Path(ocr_dir).glob("*.json"):
        stem = p.name
        for suf in ("_ocr.json", ".ocr.json", ".json"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        out[stem] = p
        out[_strip_hex(stem)] = p
    return out


def lookup_ocr(index: dict[str, Path], stem: str) -> Optional[Path]:
    if stem in index:
        return index[stem]
    s = _strip_hex(stem)
    if s in index:
        return index[s]
    for k, p in index.items():
        if _strip_hex(k) == s:
            return p
    return None


def boxes_for_sft_image(image_path: str | Path) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    """Resolve boxes + (H,W) for a LLaMA-Factory framed sample image."""
    img = Path(image_path)
    meta = img.parent / "meta.json"
    boxes: list[tuple[int, int, int, int]] = []
    if meta.is_file():
        boxes = load_boxes_from_meta_json(meta)
    # try sibling ocr
    for cand in (img.parent / "ocr.json", img.with_name(img.stem + "_ocr.json")):
        if cand.is_file():
            more = load_boxes_from_ocr_json(cand)
            if len(more) > len(boxes):
                boxes = more
            break
    try:
        from PIL import Image

        with Image.open(img) as im:
            w, h = im.size
    except Exception:
        h, w = 1, 1
    return boxes, (h, w)


def normalize_boxes(raw: Any) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    if raw is None:
        return out
    for b in raw:
        if isinstance(b, dict):
            bb = b.get("padded_bbox") or b.get("paragraph_bbox") or b.get("bbox")
        else:
            bb = b
        if bb is None or len(bb) != 4:
            continue
        out.append(tuple(int(x) for x in bb))
    return out
