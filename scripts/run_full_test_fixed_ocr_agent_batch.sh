#!/usr/bin/env bash
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run a tag batch into its own directory so concurrent language batches never
# overwrite an active evaluator's JSONL output.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${FULL_TEST_RESULT_ROOT:?Set FULL_TEST_RESULT_ROOT to the inference results directory}"
: "${FULL_TEST_DATA_ROOT:?Set FULL_TEST_DATA_ROOT to the dataset directory containing tag/ocr_json}"
: "${FULL_TEST_OUTPUT_ROOT:?Set FULL_TEST_OUTPUT_ROOT for evaluation outputs}"
RESULT_ROOT="${FULL_TEST_RESULT_ROOT}"
DATA_ROOT="${FULL_TEST_DATA_ROOT}"
MANIFEST_ROOT="${FULL_TEST_OUTPUT_ROOT}/manifests"
OUT_ROOT="${FULL_TEST_OUTPUT_ROOT}/runs"
BUILDER="${REPO_ROOT}/scripts/build_full_test_fixed_ocr_agent_inputs.py"

MODEL="${1:?missing model directory name}"
BATCH="${2:?missing batch label}"
shift 2
[[ "$#" -gt 0 ]] || { echo "provide tags" >&2; exit 2; }
if [[ "${TEXT_CRITIC_DRY_RUN:-0}" != 1 ]]; then
  : "${TEXT_CRITIC_RUNTIME_CONFIG:?Set TEXT_CRITIC_RUNTIME_CONFIG to a private provider configuration}"
  [[ -n "${MODEL_SERVICE_AK:-}" && -n "${MODEL_SERVICE_SK:-}" ]] || { echo "Missing model service credentials" >&2; exit 3; }
fi

NAME="full_test_${MODEL}_fixed_ocr_e2e_${BATCH}"
MANIFEST="${MANIFEST_ROOT}/${NAME}.json"
OUT_DIR="${OUT_ROOT}/${NAME}"
mkdir -p "${MANIFEST_ROOT}" "${OUT_DIR}"

python3 "${BUILDER}" \
  --result-root "${RESULT_ROOT}/${MODEL}" \
  --data-root "${DATA_ROOT}" \
  --tags "$@" \
  --template-out "${MANIFEST}" \
  --summary-out "${MANIFEST_ROOT}/${NAME}_summary.json"

eval_args=(
  --manifest "${MANIFEST}"
  --runtime-config "${TEXT_CRITIC_RUNTIME_CONFIG:-${REPO_ROOT}/configs/text_critic_runtime.example.json}"
  --output-jsonl "${OUT_DIR}/results.jsonl"
  --output-json "${OUT_DIR}/results.pretty.json"
  --max-workers 2
)
export PYTHONPATH="${REPO_ROOT}/agent_eval${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${TEXT_CRITIC_DRY_RUN:-0}" == 1 ]]; then
  python3 -m text_eval_agent.text_critic_batch "${eval_args[@]}" --dry-run
else
  python3 -m text_eval_agent.text_critic_batch "${eval_args[@]}" 2>&1 | tee "${OUT_DIR}/run.log"
fi
