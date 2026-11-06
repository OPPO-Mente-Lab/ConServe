#!/usr/bin/env python3
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

"""Convert M3 SFT manifests to LLaMA-Factory ShareGPT multimodal datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def convert(manifest: Path, destination: Path, output_dir: Path) -> int:
    count = 0
    with manifest.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8") as target:
        target.write("[\n")
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("system", "user", "assistant", "images"):
                if key not in item:
                    raise ValueError(f"{manifest}:{line_no}: missing {key}")
            images = item["images"]
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError(f"{manifest}:{line_no}: expected exactly one image")
            image = Path(images[0])
            if not image.is_absolute():
                image = output_dir / image
            if not image.is_file():
                raise FileNotFoundError(f"{manifest}:{line_no}: image not found: {image}")
            image_ref = os.path.relpath(image, destination.parent)
            example = {
                "messages": [
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": "<image>\n" + item["user"]},
                    {"role": "assistant", "content": item["assistant"]},
                ],
                "images": [image_ref],
            }
            if count:
                target.write(",\n")
            json.dump(example, target, ensure_ascii=False)
            count += 1
        target.write("\n]\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dir", type=Path, required=True, help="directory with manifest_train/eval.jsonl and samples/")
    args = parser.parse_args()
    sft_dir = args.sft_dir.resolve()
    destination = sft_dir / "llamafactory_data"
    destination.mkdir(parents=True, exist_ok=True)

    dataset_info = {}
    for split in ("train", "eval"):
        manifest = sft_dir / f"manifest_{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"SFT manifest not found: {manifest}")
        filename = f"image_translation_m3_{split}.json"
        count = convert(manifest, destination / filename, sft_dir)
        if count == 0:
            raise ValueError(f"SFT manifest has no examples: {manifest}")
        dataset_info[f"image_translation_m3_{split}"] = {
            "file_name": filename,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        print(f"{split}: {count} examples -> {destination / filename}")
    (destination / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
