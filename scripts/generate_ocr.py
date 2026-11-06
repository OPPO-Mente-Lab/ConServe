#!/usr/bin/env python3
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Run PaddleOCR 3 on images and write OCR JSON accepted by ConServe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def convert_result(result: dict, width: int, height: int) -> dict:
    """Assign each recognized text line its own paragraph id."""
    raw = result.get("res", result)
    texts = raw.get("rec_texts") or []
    polygons = raw.get("rec_polys") or []
    scores = raw.get("rec_scores") or []
    if len(texts) != len(polygons):
        raise ValueError("PaddleOCR rec_texts and rec_polys have different lengths")
    items = []
    for index, (text, points) in enumerate(zip(texts, polygons)):
        if not str(text).strip():
            continue
        if len(points) != 4 or any(len(point) != 2 for point in points):
            raise ValueError(f"OCR line {index} must have four [x, y] polygon points")
        item = {
            "paragraphNo": len(items),
            "text": str(text),
            "polygon": [{"x": int(x), "y": int(y)} for x, y in points],
        }
        if index < len(scores):
            item["confidence"] = float(scores[index])
        items.append(item)
    return {"data": {"width": width, "height": height, "items": items}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--lang", default="en", help="PaddleOCR language code (default: en)")
    args = parser.parse_args()
    images = sorted(
        p for p in args.image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        parser.error(f"No images found in {args.image_dir}")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        parser.error("Install PaddleOCR and its inference engine first; see docs/data_formats.md")
        raise AssertionError("unreachable") from exc
    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    for image_path in images:
        relative = image_path.relative_to(args.image_dir)
        destination = args.ocr_dir / relative.parent / f"{image_path.stem}_ocr.json"
        result_pages = ocr.predict(str(image_path))
        if len(result_pages) != 1:
            raise ValueError(f"Expected one OCR page for {image_path}, got {len(result_pages)}")
        with Image.open(image_path) as image:
            width, height = image.size
        payload = convert_result(result_pages[0].json, width, height)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{image_path} -> {destination} ({len(payload['data']['items'])} text lines)")


if __name__ == "__main__":
    main()
