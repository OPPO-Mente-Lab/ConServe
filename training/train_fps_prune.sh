#!/usr/bin/env bash
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Public launcher. Set these paths for the local machine or cluster.
: "${VERL_ROOT:?Set VERL_ROOT to the upstream/modified verl checkout}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the local model/checkpoint}"
: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to the generated training parquet}"
: "${VAL_PARQUET:?Set VAL_PARQUET to the generated validation parquet}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
VERL_TRAIN_SCRIPT="${VERL_TRAIN_SCRIPT:-${REPO_ROOT}/training/opsd_vl_global_caption.sh}"

export TOKEN_COMPRESS=1
export TOKEN_COMPRESS_MODE=prune
export TOKEN_COMPRESS_METHOD="${TOKEN_COMPRESS_METHOD:-spatial_quota_fps}"
export TOKEN_COMPRESS_KEEP_RATIO="${TOKEN_COMPRESS_KEEP_RATIO:-0.30}"
export TOKEN_COMPRESS_GRID_ROWS="${TOKEN_COMPRESS_GRID_ROWS:-4}"
export TOKEN_COMPRESS_GRID_COLS="${TOKEN_COMPRESS_GRID_COLS:-4}"
export TOKEN_COMPRESS_SPATIAL_WEIGHT="${TOKEN_COMPRESS_SPATIAL_WEIGHT:-0.25}"
export TOKEN_COMPRESS_PROTOTYPE="${TOKEN_COMPRESS_PROTOTYPE:-medoid}"

export PROJECT_NAME="${PROJECT_NAME:-conserve_opsd_fps_prune}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-fps_prune_$(date +%y%m%d_%H%M)}"
export TEACHER_MODEL="${TEACHER_MODEL:-${STUDENT_MODEL}}"
export NNODES="${NNODES:-1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-24}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
export TEACHER_EXTRA_PROMPT_TOKENS="${TEACHER_EXTRA_PROMPT_TOKENS:-2560}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

cd "${VERL_ROOT}"
[[ -f "${VERL_TRAIN_SCRIPT}" ]] || { echo "Training script not found: ${VERL_TRAIN_SCRIPT}" >&2; exit 1; }
# The compatible launcher in the checkout consumes the variables above.
bash "${VERL_TRAIN_SCRIPT}"
