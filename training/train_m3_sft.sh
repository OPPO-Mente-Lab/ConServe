#!/usr/bin/env bash
# Copyright 2026 OPPO
# SPDX-License-Identifier: Apache-2.0

# Train the M3 starting checkpoint with an external LLaMA-Factory checkout.
set -euo pipefail

: "${LLAMAFACTORY_ROOT:?Set LLAMAFACTORY_ROOT to a compatible LLaMA-Factory checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3.5-9B checkpoint}"
: "${SFT_DATA_DIR:?Set SFT_DATA_DIR to the exported llamafactory_data directory}"
: "${SFT_OUTPUT_DIR:?Set SFT_OUTPUT_DIR for checkpoints}"

[[ -f "${LLAMAFACTORY_ROOT}/src/train.py" ]] || { echo "LLaMA-Factory checkout not found" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Model config not found: ${MODEL_PATH}" >&2; exit 1; }
for file in dataset_info.json image_translation_m3_train.json image_translation_m3_eval.json; do
  [[ -f "${SFT_DATA_DIR}/${file}" ]] || { echo "SFT data file not found: ${SFT_DATA_DIR}/${file}" >&2; exit 1; }
done

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-4}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-1048576}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_LR="${SFT_LR:-1e-5}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29519}"
NODE_RANK="${NODE_RANK:-0}"

command=(
  python -m torch.distributed.run
  --nnodes "${NNODES}" --nproc_per_node "${N_GPUS_PER_NODE}"
  --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  src/train.py
  --stage sft --do_train --template qwen3_5_nothink
  --model_name_or_path "${MODEL_PATH}" --trust_remote_code
  --dataset_dir "${SFT_DATA_DIR}" --dataset image_translation_m3_train
  --eval_dataset image_translation_m3_eval --eval_strategy steps
  --eval_steps "${SFT_EVAL_STEPS:-100}" --per_device_eval_batch_size 1
  --image_max_pixels "${IMAGE_MAX_PIXELS}"
  --freeze_vision_tower "${FREEZE_VISION_TOWER:-true}"
  --freeze_multi_modal_projector false --freeze_language_model false
  --finetuning_type full --cutoff_len "${MAX_LENGTH}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --learning_rate "${SFT_LR}" --lr_scheduler_type cosine
  --warmup_ratio "${SFT_WARMUP_RATIO:-0.03}"
  --num_train_epochs "${SFT_EPOCHS}" --bf16 --gradient_checkpointing
  --save_steps "${SFT_SAVE_STEPS:-200}" --logging_steps "${SFT_LOGGING_STEPS:-5}"
  --output_dir "${SFT_OUTPUT_DIR}" --report_to none
)
if [[ "${USE_DEEPSPEED:-1}" == 1 ]]; then
  deepspeed_config="${DEEPSPEED_CONFIG:-${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json}"
  [[ -f "${deepspeed_config}" ]] || { echo "DeepSpeed config not found: ${deepspeed_config}" >&2; exit 1; }
  command+=(--deepspeed "${deepspeed_config}")
fi

cd "${LLAMAFACTORY_ROOT}"
if [[ "${SFT_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
