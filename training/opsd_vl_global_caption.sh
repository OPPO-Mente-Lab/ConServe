#!/usr/bin/env bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Based on the verl on-policy distillation FSDP example:
# https://github.com/verl-project/verl/blob/3ba8181/examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh

# Modifications Copyright 2026 OPPO

set -euo pipefail

: "${VERL_ROOT:?Set VERL_ROOT to the compatible verl checkout}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the student checkpoint}"
: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to the OPSD train parquet}"
: "${VAL_PARQUET:?Set VAL_PARQUET to the OPSD validation parquet}"

TEACHER_MODEL="${TEACHER_MODEL:-${STUDENT_MODEL}}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-16}"
STUDENT_NNODES="${STUDENT_NNODES:-${NNODES}}"
TEACHER_NNODES="${TEACHER_NNODES:-${NNODES}}"
STUDENT_GPUS_PER_NODE="${STUDENT_GPUS_PER_NODE:-12}"
TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-4}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
TEACHER_TP="${TEACHER_TP:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-24}"
ROLLOUT_N="${ROLLOUT_N:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
TEACHER_EXTRA_PROMPT_TOKENS="${TEACHER_EXTRA_PROMPT_TOKENS:-2560}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
PROJECT_NAME="${PROJECT_NAME:-conserve_opsd_fps_prune}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fps_prune}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
VERL_PYTHON="${VERL_PYTHON:-python3}"

student_gpus=$((STUDENT_NNODES * STUDENT_GPUS_PER_NODE))
teacher_gpus=$((TEACHER_NNODES * TEACHER_GPUS_PER_NODE))
physical_gpus=$((NNODES * N_GPUS_PER_NODE))
if (( student_gpus < 1 || teacher_gpus < 1 || student_gpus + teacher_gpus != physical_gpus )); then
  echo "Student (${student_gpus}) and teacher (${teacher_gpus}) GPUs must sum to physical GPUs (${physical_gpus})." >&2
  exit 1
fi
if (( teacher_gpus % TEACHER_TP != 0 || student_gpus % ROLLOUT_TP != 0 )); then
  echo "Teacher and rollout tensor parallel sizes must divide their GPU pools." >&2
  exit 1
fi
if (( (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % student_gpus != 0 || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
  echo "PPO_MINI_BATCH_SIZE * ROLLOUT_N must divide across student GPUs; TRAIN_BATCH_SIZE must be a multiple of PPO_MINI_BATCH_SIZE." >&2
  exit 1
fi

[[ -f "${VERL_ROOT}/verl/trainer/main_ppo.py" ]] || { echo "Compatible verl checkout not found: ${VERL_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_PARQUET}" && -f "${VAL_PARQUET}" ]] || { echo "Training and validation parquet files must exist." >&2; exit 1; }

teacher_prompt_length=$((MAX_PROMPT_LENGTH + TEACHER_EXTRA_PROMPT_TOKENS))
student_max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
teacher_max_model_len=$((teacher_prompt_length + MAX_RESPONSE_LENGTH + 1))

command=(
  "${VERL_PYTHON}" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  "data.train_files=${TRAIN_PARQUET}"
  "data.val_files=${VAL_PARQUET}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  data.image_key=images
  data.teacher_prompt_key=teacher_prompt
  data.teacher_image_key=teacher_images
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=True
  data.dataloader_num_workers=0
  +data.apply_chat_template_kwargs.enable_thinking=False
  "actor_rollout_ref.model.path=${STUDENT_MODEL}"
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.use_torch_compile=False
  "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-True}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-True}"
  actor_rollout_ref.rollout.name=vllm
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.4}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.max_model_len=${student_max_model_len}"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent
  +actor_rollout_ref.rollout.limit_images=1
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "trainer.logger=${TRAINER_LOGGER}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${DEFAULT_LOCAL_DIR:-${VERL_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
  "trainer.n_gpus_per_node=${STUDENT_GPUS_PER_NODE}"
  "trainer.nnodes=${STUDENT_NNODES}"
  trainer.val_before_train=False
  "trainer.save_freq=${SAVE_FREQ:-24}"
  "trainer.test_freq=${TEST_FREQ:--1}"
  "trainer.total_epochs=${TOTAL_EPOCHS:-5}"
  distillation.enabled=True
  "distillation.n_gpus_per_node=${TEACHER_GPUS_PER_NODE}"
  "distillation.nnodes=${TEACHER_NNODES}"
  "distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}"
  distillation.teacher_models.teacher_model.inference.name=vllm
  "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}"
  "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL:-0.5}"
  distillation.teacher_models.teacher_model.inference.limit_images=2
  "distillation.teacher_models.teacher_model.inference.prompt_length=${teacher_prompt_length}"
  "distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len}"
  distillation.distillation_loss.loss_mode=forward_kl_topk
  "distillation.distillation_loss.topk=${DISTILLATION_TOPK:-50}"
  distillation.distillation_loss.use_task_rewards=False
  distillation.distillation_loss.use_policy_gradient=False
  distillation.distillation_loss.loss_max_clamp=10.0
  distillation.distillation_loss.log_prob_min_clamp=-10.0
  "$@"
)

cd "${VERL_ROOT}"
if [[ "${CONSERVE_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
