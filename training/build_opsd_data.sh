#!/usr/bin/env bash
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build the parquet used by OPSD/OPD from caller-provided evaluation records.
: "${AGENT_RESULTS_JSONL:?Set AGENT_RESULTS_JSONL to an Agent results JSONL file}"
: "${VLM_OUTPUT_DIR:?Set VLM_OUTPUT_DIR to the matching VLM output directory}"
: "${IMAGE_LIST_JSONL:?Set IMAGE_LIST_JSONL to an image list with stem and image_path}"
: "${OCR_DIR:?Set OCR_DIR to the OCR JSON directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated annotations and parquet files}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUTPUT_DIR}"
python "${SCRIPT_DIR}/opsd_build_agent_pos_neg.py" \
  --results-jsonl "${AGENT_RESULTS_JSONL}" \
  --vlm-dir "${VLM_OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/annotations"

builder_args=(
  --image-list-jsonl "${IMAGE_LIST_JSONL}"
  --ocr-dir "${OCR_DIR}"
)
if [[ -n "${CODE_DIR:-}" ]]; then
  builder_args+=(--code-dir "${CODE_DIR}")
fi
if [[ -n "${MNT_ROOT:-}" || -n "${NOTEBOOK_ROOT:-}" ]]; then
  : "${MNT_ROOT:?Set MNT_ROOT together with NOTEBOOK_ROOT}"
  : "${NOTEBOOK_ROOT:?Set NOTEBOOK_ROOT together with MNT_ROOT}"
  builder_args+=(--mnt-root "${MNT_ROOT}" --notebook-root "${NOTEBOOK_ROOT}")
fi

generated_modes=()
for mode in pos neg; do
  if [[ "${mode}" == pos ]]; then
    annotation="${OUTPUT_DIR}/annotations/positive_reference.json"
  else
    annotation="${OUTPUT_DIR}/annotations/negative_critique.json"
  fi
  image_count="$(python - "${annotation}" <<'PY'
import json
import sys
from pathlib import Path

print(len(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["images"]))
PY
)"
  if [[ "${image_count}" == 0 ]]; then
    echo "Skipping ${mode}: no matching Agent annotations"
    continue
  fi
  python "${SCRIPT_DIR}/opsd_vl_global_ab_teacher.py" \
    --mode "${mode}" \
    --annotation-json "${annotation}" \
    --output-dir "${OUTPUT_DIR}/${mode}" \
    "${builder_args[@]}"
  generated_modes+=("${mode}")
done

if [[ "${#generated_modes[@]}" == 0 ]]; then
  echo "No OPSD parquet rows were generated" >&2
  exit 1
fi

python - "${OUTPUT_DIR}" "${generated_modes[@]}" <<'PY'
from pathlib import Path
import sys

import pandas as pd

output_dir = Path(sys.argv[1])
modes = sys.argv[2:]
combined_splits = {}
for split in ("train", "val"):
    frames = [
        pd.read_parquet(output_dir / mode / f"{split}.parquet")
        for mode in modes
    ]
    combined_splits[split] = pd.concat(frames, ignore_index=True)

train_columns = combined_splits["train"].columns
if combined_splits["val"].empty:
    combined_splits["val"] = combined_splits["train"].iloc[:0].copy()
else:
    combined_splits["val"] = combined_splits["val"].reindex(columns=train_columns)
for split, combined in combined_splits.items():
    combined.to_parquet(output_dir / f"{split}.parquet", index=False)
    print(f"{split}: {len(combined)} rows -> {output_dir / f'{split}.parquet'}")
if combined_splits["val"].empty:
    print("Validation parquet is empty; supply more distinct images before training.")
PY
