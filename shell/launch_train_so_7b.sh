#!/usr/bin/env bash

# Three-stage SO-7B training. See docs/training.md for inputs and resume rules.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "${GPUS}"
NPROC="${NPROC:-${#gpu_ids[@]}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29573}"
START_STAGE="${START_STAGE:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREVIEW=0
[[ "${CHECK_ONLY}" == 1 || "${DRY_RUN}" == 1 ]] && PREVIEW=1
case "${START_STAGE}" in 1|2|3) ;; *) echo "START_STAGE must be 1, 2, or 3." >&2; exit 1 ;; esac
for key in NPROC NNODES; do
  [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || { echo "${key} must be a positive integer." >&2; exit 1; }
done
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || {
  echo "NODE_RANK must be between 0 and NNODES-1." >&2; exit 1;
}

if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "[ERROR] NNODES=${NNODES} > 1 but MASTER_ADDR is loopback (${MASTER_ADDR}). " >&2
  echo "        Set MASTER_ADDR to the actual IP of rank-0 machine (reachable from all nodes)." >&2
  exit 1
fi

DATASET_ROOT="${SO_DATASET_ROOT:-}"
QA_ROOT="${QA_ROOT:-${DATASET_ROOT:+${DATASET_ROOT}/qa}}"
[[ -n "${QA_ROOT}" ]] || { echo "Set QA_ROOT or SO_DATASET_ROOT." >&2; exit 1; }
AUDIO_ROOT="${AUDIO_ROOT:-${DATASET_ROOT:-$(dirname "${QA_ROOT}")}}"
BEATS_CKPT="${BEATS_CKPT:-${SO_ENCODER_CKPT:-}}"
BEATS_REPO="${BEATS_REPO:-${SO_BEATS_REPO:-}}"
MODEL_ID="${MODEL_ID:-${SO_BASE_MODEL:-Qwen/Qwen2.5-Omni-7B}}"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/so_7b}"
STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1_projector}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2_encoder_lora}"
STAGE3_DIR="${STAGE3_DIR:-${RUN_ROOT}/stage3_beats_lora}"

STAGE2_RESUME_CKPT="${STAGE2_RESUME_CKPT:-${STAGE1_DIR}/checkpoints/best_trainable.pt}"
STAGE3_RESUME_CKPT="${STAGE3_RESUME_CKPT:-${STAGE2_DIR}/checkpoints/best_trainable.pt}"

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SAVE_EVERY_N_OPT_STEPS="${SAVE_EVERY_N_OPT_STEPS:-1000}"
VALID_EVERY_N_OPT_STEPS="${VALID_EVERY_N_OPT_STEPS:-1000}"

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-0}"
QWEN_AUDIO_CACHE_MANIFEST="${QWEN_AUDIO_CACHE_MANIFEST:-}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-3}"

STAGE1_LR="${STAGE1_LR:-5e-5}"
STAGE1_PROJECTOR_LR="${STAGE1_PROJECTOR_LR:-1e-4}"

STAGE2_LR="${STAGE2_LR:-5e-5}"
STAGE2_LORA_LR="${STAGE2_LORA_LR:-5e-5}"
STAGE2_PROJECTOR_LR="${STAGE2_PROJECTOR_LR:-3e-5}"

STAGE3_LR="${STAGE3_LR:-3e-5}"
STAGE3_LORA_LR="${STAGE3_LORA_LR:-3e-5}"
STAGE3_PROJECTOR_LR="${STAGE3_PROJECTOR_LR:-1e-6}"
STAGE3_BEATS_LR="${STAGE3_BEATS_LR:-1e-6}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
read -r -a LORA_TARGET_MODULES <<< "${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}"

if [[ ! -f "${BEATS_CKPT}" ]]; then
  echo "Set SO_ENCODER_CKPT or BEATS_CKPT to an existing SO-Encoder checkpoint: ${BEATS_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${QA_ROOT}" ]]; then
  echo "Missing QA root: ${QA_ROOT}" >&2
  exit 1
fi
for split in train valid; do
  if [[ ! -f "${QA_ROOT}/${split}.jsonl" ]]; then
    echo "Missing ${QA_ROOT}/${split}.jsonl" >&2
    exit 1
  fi
done

echo "==========================================================="
echo " Multi-node config:"
echo "   NNODES=${NNODES}  NODE_RANK=${NODE_RANK}"
echo "   MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "   NPROC (GPUs per node) = ${NPROC}  GPUS=${GPUS}"
echo "   Global world size     = $((NNODES * NPROC))"
echo "   START_STAGE=${START_STAGE}"
echo "==========================================================="

run_train() {
  local command=(env CUDA_VISIBLE_DEVICES="${GPUS}" torchrun
    --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${NPROC}"
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}"
    "${ROOT_DIR}/train_so_qa.py" "$@")
  printf '[command]'; printf ' %q' "${command[@]}"; printf '\n'
  if (( PREVIEW )); then return; fi
  local previous="" output="" value
  for value in "$@"; do
    [[ "${previous}" == --output-dir ]] && output="${value}"
    previous="${value}"
  done
  if [[ -d "${output}" && -n "$(ls -A -- "${output}")" && "${RESUME_EXISTING:-0}" != 1 ]]; then
    echo "Output is not empty: ${output}. Choose a new RUN_ROOT or set RESUME_EXISTING=1 for an intentional resume." >&2
    exit 1
  fi
  "${command[@]}"
}

common_args=(
  --model-id "${MODEL_ID}"
  --beats-checkpoint "${BEATS_CKPT}"
  --beats-repo "${BEATS_REPO}"
  --qa-root "${QA_ROOT}"
  --audio-root "${AUDIO_ROOT}"
  --train-split train
  --valid-split valid
  --device cuda:0
  --dtype bfloat16
  --attn-impl "${ATTN_IMPL}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --persistent-workers
  --prefetch-factor "${PREFETCH_FACTOR}"
  --warmup-ratio 0.03
  --weight-decay 0.01
  --max-grad-norm 1.0
  --save-every-epoch
  --save-every-n-optimizer-steps "${SAVE_EVERY_N_OPT_STEPS}"
  --valid-every-n-optimizer-steps "${VALID_EVERY_N_OPT_STEPS}"
  --valid-generate-max-samples "${VALID_GENERATE_MAX_SAMPLES:-32}"
  --valid-max-new-tokens 96
  --valid-num-beams 1
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES[@]}"
  --lora-target-prefixes thinker.model
)
if (( USE_GRADIENT_CHECKPOINTING == 1 )); then
  common_args+=(--gradient-checkpointing)
  echo "[config] gradient_checkpointing = enabled"
else
  echo "[config] gradient_checkpointing = disabled"
fi
if [[ -n "${QWEN_AUDIO_CACHE_MANIFEST}" ]]; then
  common_args+=(--audio-feature-cache-manifest "${QWEN_AUDIO_CACHE_MANIFEST}")
  echo "[config] audio feature cache = ${QWEN_AUDIO_CACHE_MANIFEST}"
else
  echo "[config] audio feature cache = off"
fi
if [[ "${VALID_GENERATE_FULL:-0}" == "1" ]]; then
  common_args+=(--valid-generate-full)
  echo "[config] valid_generate_full = enabled"
else
  echo "[config] validation generation limit = ${VALID_GENERATE_MAX_SAMPLES:-32}"
fi

if [[ -n "${STAGE1_RESUME_CKPT:-}" && ! -f "${STAGE1_RESUME_CKPT}" ]]; then
  echo "Missing stage1 resume checkpoint: ${STAGE1_RESUME_CKPT}" >&2; exit 1
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then common_args+=(--max-train-samples "${MAX_TRAIN_SAMPLES}"); fi
if [[ -n "${MAX_VALID_SAMPLES:-}" ]]; then common_args+=(--max-valid-samples "${MAX_VALID_SAMPLES}"); fi

if (( START_STAGE <= 1 )); then
  echo "==========================================================="
  echo "[stage1] projector_only (${STAGE1_EPOCHS} epochs, lr=${STAGE1_LR})"
  echo "  → ${STAGE1_DIR}"
  echo "==========================================================="
  stage1_extra=()
  if [[ -n "${STAGE1_RESUME_CKPT:-}" ]]; then
    echo "  resume from: ${STAGE1_RESUME_CKPT}"
    stage1_extra+=(--resume-checkpoint-path "${STAGE1_RESUME_CKPT}")
    if [[ "${STAGE1_RESUME_MODEL_ONLY:-0}" == "1" ]]; then
      stage1_extra+=(--resume-model-only)
      echo "  resume mode: MODEL ONLY (fresh optimizer, restart from epoch 1)"
    else
      echo "  resume mode: FULL (optimizer + scheduler + step counter restored)"
    fi
  fi
  run_train \
    "${common_args[@]}" \
    --projector-only \
    --lr "${STAGE1_LR}" \
    --projector-lr "${STAGE1_PROJECTOR_LR}" \
    --epochs "${STAGE1_EPOCHS}" \
    --output-dir "${STAGE1_DIR}" \
    "${stage1_extra[@]}"
fi

if (( START_STAGE <= 2 )); then
  if (( PREVIEW == 0 || START_STAGE == 2 )) && [[ ! -f "${STAGE2_RESUME_CKPT}" ]]; then
    echo "Missing stage2 resume checkpoint: ${STAGE2_RESUME_CKPT}" >&2
    echo "Set START_STAGE=1 to produce it, or STAGE2_RESUME_CKPT=/path/to/best_trainable.pt." >&2
    exit 1
  fi
  echo "==========================================================="
  echo "[stage2] encoder_lora (${STAGE2_EPOCHS} epochs, lora_lr=${STAGE2_LORA_LR}, proj_lr=${STAGE2_PROJECTOR_LR})"
  echo "  resume from: ${STAGE2_RESUME_CKPT}"
  echo "  → ${STAGE2_DIR}"
  echo "==========================================================="
  run_train \
    "${common_args[@]}" \
    --encoder-lora \
    --resume-checkpoint-path "${STAGE2_RESUME_CKPT}" \
    --resume-model-only \
    --lr "${STAGE2_LR}" \
    --lora-lr "${STAGE2_LORA_LR}" \
    --projector-lr "${STAGE2_PROJECTOR_LR}" \
    --epochs "${STAGE2_EPOCHS}" \
    --output-dir "${STAGE2_DIR}"
fi

if (( START_STAGE <= 3 )); then
  if (( PREVIEW == 0 || START_STAGE == 3 )) && [[ ! -f "${STAGE3_RESUME_CKPT}" ]]; then
    echo "Missing stage3 resume checkpoint: ${STAGE3_RESUME_CKPT}" >&2
    echo "Set START_STAGE=2 to produce it, or STAGE3_RESUME_CKPT=/path/to/best_trainable.pt." >&2
    exit 1
  fi
  echo "==========================================================="
  echo "[stage3] beats_lora (${STAGE3_EPOCHS} epochs, beats_lr=${STAGE3_BEATS_LR}, lora_lr=${STAGE3_LORA_LR}, proj_lr=${STAGE3_PROJECTOR_LR})"
  echo "  resume from: ${STAGE3_RESUME_CKPT}"
  echo "  → ${STAGE3_DIR}"
  echo "==========================================================="
  stage3_extra=()
  if [[ "${STAGE3_RESUME_MODEL_ONLY:-1}" == "1" ]]; then
    stage3_extra+=(--resume-model-only)
    echo "  resume mode: MODEL ONLY (fresh optimizer, restart from epoch 1)"
  else
    echo "  resume mode: FULL (optimizer + scheduler + step counter restored)"
  fi
  run_train \
    "${common_args[@]}" \
    --beats-lora \
    --resume-checkpoint-path "${STAGE3_RESUME_CKPT}" \
    "${stage3_extra[@]}" \
    --lr "${STAGE3_LR}" \
    --lora-lr "${STAGE3_LORA_LR}" \
    --projector-lr "${STAGE3_PROJECTOR_LR}" \
    --beats-lr "${STAGE3_BEATS_LR}" \
    --epochs "${STAGE3_EPOCHS}" \
    --output-dir "${STAGE3_DIR}"
fi

if (( PREVIEW )); then
  echo "Preview complete; no training or output files were created. Planned run: ${RUN_ROOT}"
else
  echo "All requested stages finished. Run dir = ${RUN_ROOT}"
fi
