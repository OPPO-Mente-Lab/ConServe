# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


def prepare_image_pair(
    image_paths: list[Path],
    output_dir: Path,
    max_edge: int = 1600,
    jpeg_quality: int = 88,
) -> list[Path]:
    paths = [Path(path) for path in image_paths]
    if len(paths) != 2:
        raise ValueError("Image transport requires exactly two images")
    if max_edge < 256:
        raise ValueError("max_edge must be at least 256 pixels")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _prepare_one(path, output_dir / f"image_{index}.jpg", max_edge, jpeg_quality)
        for index, path in enumerate(paths, start=1)
    ]


def _prepare_one(source: Path, destination: Path, max_edge: int, jpeg_quality: int) -> Path:
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            if max(image.size) > max_edge:
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "L"}:
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image.convert("RGB"))
                image = background
            elif image.mode == "L":
                image = image.convert("RGB")
            image.save(
                destination,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
                progressive=True,
            )
    except (UnidentifiedImageError, OSError):
        return source
    return destination
