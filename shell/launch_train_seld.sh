#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(python -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x]))' "${GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29571}"
START_STAGE="${START_STAGE:-1}"

QA_VERSION="${QA_VERSION:-default}"
QA_ROOT="${QA_ROOT:-}"
QA_ROOTS="${QA_ROOTS:-}"
SELD_TASK_ID="${SELD_TASK_ID:-235}"
SELD_CKPT="${SELD_CKPT:-${ROOT_DIR}/models_audio/235_qwenmel235_run01_dev_split0_multiaccdoa_foa_model.h5}"
FEATURE_CACHE_MANIFEST="${FEATURE_CACHE_MANIFEST:-${ROOT_DIR}/prepared_datasets/starss23_foa_plus_29cls_20s/seld_feature_cache_235/manifest.json}"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/spatial_qa_runs}"
STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/spatial_lora_235_stage1_spatial_only}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/spatial_lora_235_stage2_adapter_lora}"
STAGE3_DIR="${STAGE3_DIR:-${RUN_ROOT}/spatial_lora_235_stage3_spatial_lora}"

STAGE1_INIT_CKPT="${STAGE1_INIT_CKPT:-}"
STAGE2_RESUME_CKPT="${STAGE2_RESUME_CKPT:-${STAGE1_DIR}/checkpoints/best_trainable.pt}"
STAGE3_RESUME_CKPT="${STAGE3_RESUME_CKPT:-${STAGE2_DIR}/checkpoints/best_trainable.pt}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
FEATURE_CACHE_MAX_ENTRIES="${FEATURE_CACHE_MAX_ENTRIES:-8}"
SAVE_EVERY_N_OPT_STEPS="${SAVE_EVERY_N_OPT_STEPS:-500}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-4}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-8}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-6}"
STAGE1_LR="${STAGE1_LR:-1e-5}"
STAGE2_LR="${STAGE2_LR:-1e-5}"
STAGE3_LR="${STAGE3_LR:-1e-6}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES=(${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj})

if [[ ! -f "${SELD_CKPT}" ]]; then
  echo "Missing SELD checkpoint: ${SELD_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${FEATURE_CACHE_MANIFEST}" ]]; then
  echo "Missing feature cache manifest: ${FEATURE_CACHE_MANIFEST}" >&2
  exit 1
fi

run_train() {
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NPROC}" --master-port="${MASTER_PORT}" \
    "${ROOT_DIR}/train_legacy_spatial_qa.py" "$@"
}

qa_args=()
if [[ -n "${QA_ROOTS}" ]]; then
  read -r -a QA_ROOTS_ARRAY <<< "${QA_ROOTS}"
  qa_args=(--qa-roots "${QA_ROOTS_ARRAY[@]}")
elif [[ -n "${QA_ROOT}" ]]; then
  qa_args=(--qa-roots "${QA_ROOT}")
else
  qa_args=(--qa-version "${QA_VERSION}")
fi

common_args=(
  "${qa_args[@]}"
  --seld-task-id "${SELD_TASK_ID}"
  --seld-checkpoint-path "${SELD_CKPT}"
  --seld-feature-cache-manifest "${FEATURE_CACHE_MANIFEST}"
  --device cuda:0
  --dtype bfloat16
  --spatial-fp32
  --gradient-checkpointing
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --persistent-workers
  --prefetch-factor "${PREFETCH_FACTOR}"
  --feature-cache-max-entries "${FEATURE_CACHE_MAX_ENTRIES}"
  --save-every-n-optimizer-steps "${SAVE_EVERY_N_OPT_STEPS}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES[@]}"
)

if (( START_STAGE <= 1 )); then
  echo "[stage1] spatial_only on ${QA_VERSION}: train adapter/projector only -> ${STAGE1_DIR}"
  stage1_resume_args=()
  if [[ -n "${STAGE1_INIT_CKPT}" ]]; then
    if [[ ! -f "${STAGE1_INIT_CKPT}" ]]; then
      echo "Missing STAGE1_INIT_CKPT: ${STAGE1_INIT_CKPT}" >&2
      exit 1
    fi
    stage1_resume_args=(--resume-checkpoint-path "${STAGE1_INIT_CKPT}" --resume-model-only)
  fi
  run_train \
    "${common_args[@]}" \
    --freeze-spatial-only \
    --lr "${STAGE1_LR}" \
    --epochs "${STAGE1_EPOCHS}" \
    --output-dir "${STAGE1_DIR}" \
    "${stage1_resume_args[@]}"
fi

if (( START_STAGE <= 2 )); then
  if [[ ! -f "${STAGE2_RESUME_CKPT}" ]]; then
    echo "Missing stage2 resume checkpoint: ${STAGE2_RESUME_CKPT}" >&2
    echo "Set START_STAGE=1 to produce it, or set STAGE2_RESUME_CKPT=/path/to/best_trainable.pt." >&2
    exit 1
  fi
  echo "[stage2] adapter_lora on ${QA_VERSION}: train adapter/projector + LLM LoRA -> ${STAGE2_DIR}"
  run_train \
    "${common_args[@]}" \
    --train-adapter-lora \
    --resume-checkpoint-path "${STAGE2_RESUME_CKPT}" \
    --resume-model-only \
    --lr "${STAGE2_LR}" \
    --epochs "${STAGE2_EPOCHS}" \
    --output-dir "${STAGE2_DIR}"
fi

if (( START_STAGE <= 3 )); then
  if [[ ! -f "${STAGE3_RESUME_CKPT}" ]]; then
    echo "Missing stage3 resume checkpoint: ${STAGE3_RESUME_CKPT}" >&2
    echo "Set START_STAGE=2 to produce it, or set STAGE3_RESUME_CKPT=/path/to/best_trainable.pt." >&2
    exit 1
  fi
  echo "[stage3] spatial_lora on ${QA_VERSION}: train spatial encoder + adapter/projector + LLM LoRA -> ${STAGE3_DIR}"
  echo "[stage3] using conservative lr=${STAGE3_LR}"
  run_train \
    "${common_args[@]}" \
    --train-spatial-lora \
    --spatial-backbone-fp32 \
    --resume-checkpoint-path "${STAGE3_RESUME_CKPT}" \
    --resume-model-only \
    --lr "${STAGE3_LR}" \
    --epochs "${STAGE3_EPOCHS}" \
    --output-dir "${STAGE3_DIR}"
fi
