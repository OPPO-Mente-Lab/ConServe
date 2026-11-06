#!/usr/bin/env python3
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic image and matching OCR JSON for the README quickstart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    image_dir = args.out_dir / "images"
    ocr_dir = args.out_dir / "ocr"
    image_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (700, 420), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 38)
    except OSError:
        font = ImageFont.load_default()
    items = []
    for paragraph_no, (text, y) in enumerate(
        [("Open settings", 50), ("Choose your language", 165), ("Save changes", 280)]
    ):
        left, top, right, bottom = draw.textbbox((72, y), text, font=font)
        draw.text((72, y), text, font=font, fill="black")
        polygon = [
            {"x": left, "y": top},
            {"x": right, "y": top},
            {"x": right, "y": bottom},
            {"x": left, "y": bottom},
        ]
        items.append(
            {"paragraphNo": paragraph_no, "text": text, "polygon": polygon, "confidence": 1.0}
        )
    image.save(image_dir / "demo.png")
    ocr = {
        "data": {
            "width": image.width,
            "height": image.height,
            "items": items,
        }
    }
    (ocr_dir / "demo_ocr.json").write_text(json.dumps(ocr, indent=2), encoding="utf-8")
    print(f"Created {image_dir / 'demo.png'} and {ocr_dir / 'demo_ocr.json'}")


if __name__ == "__main__":
    main()
