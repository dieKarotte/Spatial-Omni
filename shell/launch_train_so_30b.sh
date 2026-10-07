#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

CONDA_ENV="${CONDA_ENV:-$(python -c 'import sys; print(sys.prefix)')}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
QWEN3_OMNI_FORK="${QWEN3_OMNI_FORK:-${QWEN3_TRANSFORMERS_FORK:-}}"
CHECK_ONLY="${CHECK_ONLY:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Shared conda Python is not executable: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ENV=/absolute/path/to/spatial-omni-30b." >&2
  exit 1
fi
if [[ "${CHECK_ONLY}" != "0" && "${CHECK_ONLY}" != "1" ]]; then
  echo "[ERROR] CHECK_ONLY must be 0 or 1, got: ${CHECK_ONLY}" >&2
  exit 1
fi

CONDA_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ ! -d "${CONDA_SITE_PACKAGES}" ]]; then
  echo "[ERROR] Conda site-packages not found: ${CONDA_SITE_PACKAGES}" >&2
  exit 1
fi
export PATH="${CONDA_ENV}/bin:${PATH}"
INHERITED_PYTHONPATH="${PYTHONPATH:-}"
INHERITED_PYTHONPATH="${INHERITED_PYTHONPATH#:}"
export PYTHONPATH="${QWEN3_OMNI_FORK:+${QWEN3_OMNI_FORK}:}${ROOT_DIR}:${CONDA_SITE_PACKAGES}${INHERITED_PYTHONPATH:+:${INHERITED_PYTHONPATH}}"
export QWEN3_OMNI_FORK

if ! ENV_SUMMARY="$("${PYTHON_BIN}" - <<'PY'
import sys
import peft
import torch
import transformers
from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerConfig  # noqa: F401

print(
    f"python={sys.version.split()[0]} torch={torch.__version__} "
    f"transformers={transformers.__version__} peft={peft.__version__}"
)
PY
)"; then
  echo "[ERROR] The selected conda environment cannot import the SO-30B stack." >&2
  exit 1
fi

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$("${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x]))' "${GPUS}")}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29577}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-3}"

if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "[ERROR] NNODES=${NNODES} > 1 but MASTER_ADDR is loopback" >&2
  exit 1
fi

ASSET_ROOT="${SO_ASSET_ROOT:-${ROOT_DIR}}"
DATASET_ROOT="${SO_DATASET_ROOT:-${ASSET_ROOT}/SO-Dataset}"
QA_ROOT="${QA_ROOT:-${DATASET_ROOT}/qa}"
AUDIO_ROOT="${AUDIO_ROOT:-${DATASET_ROOT}}"
BEATS_CKPT="${BEATS_CKPT:-${SO_ENCODER_CKPT:-${ASSET_ROOT}/ckpts/SO-Encoder/SO-Encoder.pt}}"
BEATS_REPO="${BEATS_REPO:-${SO_BEATS_REPO:-${ROOT_DIR}}}"
MODEL_ID="${MODEL_ID:-${SO_BASE_MODEL:-${ASSET_ROOT}/ckpts/Qwen3-Omni-30B-A3B-Instruct}}"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/so_30b/$(date +%Y%m%d_%H%M%S)}"
STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1_projector}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2_encoder_lora}"
STAGE3_DIR="${STAGE3_DIR:-${RUN_ROOT}/stage3_beats_lora}"

STAGE2_RESUME_CKPT="${STAGE2_RESUME_CKPT:-${STAGE1_DIR}/checkpoints/best_trainable.pt}"
STAGE3_RESUME_CKPT="${STAGE3_RESUME_CKPT:-${STAGE2_DIR}/checkpoints/best_trainable.pt}"

if [[ -n "${DEVICE_MAP:-}" ]]; then
  DATA_PARALLEL_WORLD=1
  DEFAULT_STAGE1_BATCH=16; DEFAULT_STAGE2_BATCH=8; DEFAULT_STAGE3_BATCH=4
  DEFAULT_STAGE1_ACCUM=4; DEFAULT_STAGE2_ACCUM=8; DEFAULT_STAGE3_ACCUM=16
else
  DATA_PARALLEL_WORLD=$((NNODES * NPROC))
  DEFAULT_STAGE1_BATCH=1; DEFAULT_STAGE2_BATCH=1; DEFAULT_STAGE3_BATCH=1
  DEFAULT_STAGE1_ACCUM=8; DEFAULT_STAGE2_ACCUM=8; DEFAULT_STAGE3_ACCUM=8
fi
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-${BATCH_SIZE:-${DEFAULT_STAGE1_BATCH}}}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${BATCH_SIZE:-${DEFAULT_STAGE2_BATCH}}}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-${BATCH_SIZE:-${DEFAULT_STAGE3_BATCH}}}"
STAGE1_GRAD_ACCUM="${STAGE1_GRAD_ACCUM:-${GRAD_ACCUM_STEPS:-${DEFAULT_STAGE1_ACCUM}}}"
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-${GRAD_ACCUM_STEPS:-${DEFAULT_STAGE2_ACCUM}}}"
STAGE3_GRAD_ACCUM="${STAGE3_GRAD_ACCUM:-${GRAD_ACCUM_STEPS:-${DEFAULT_STAGE3_ACCUM}}}"
BATCH_SIZE="${BATCH_SIZE:-${STAGE1_BATCH_SIZE}}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${STAGE1_GRAD_ACCUM}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAVE_EVERY_N_OPT_STEPS="${SAVE_EVERY_N_OPT_STEPS:-500}"
VALID_EVERY_N_OPT_STEPS="${VALID_EVERY_N_OPT_STEPS:-500}"

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-1}"
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
LORA_TARGET_MODULES=(${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj})
STAGE2_TRAIN_MOE_ROUTER="${STAGE2_TRAIN_MOE_ROUTER:-0}"
STAGE3_TRAIN_MOE_ROUTER="${STAGE3_TRAIN_MOE_ROUTER:-1}"
MOE_ROUTER_LR="${MOE_ROUTER_LR:-1e-6}"
MOE_ROUTER_AUX_LOSS_COEF="${MOE_ROUTER_AUX_LOSS_COEF:-1e-3}"

if [[ ! -f "${BEATS_CKPT}" ]]; then echo "Missing BEATs ckpt: ${BEATS_CKPT}" >&2; exit 1; fi
if [[ ! -d "${QA_ROOT}" ]]; then echo "Missing QA root: ${QA_ROOT}" >&2; exit 1; fi
if [[ ! -d "${MODEL_ID}" ]]; then echo "Missing model dir: ${MODEL_ID}" >&2; exit 1; fi
if [[ ! -f "${MODEL_ID}/config.json" ]]; then echo "Missing model config: ${MODEL_ID}/config.json" >&2; exit 1; fi
if [[ -n "${QWEN3_OMNI_FORK}" && ! -d "${QWEN3_OMNI_FORK}" ]]; then echo "Missing transformers fork: ${QWEN3_OMNI_FORK}" >&2; exit 1; fi
for split in train valid test; do
  [[ -f "${QA_ROOT}/${split}.jsonl" ]] || { echo "Missing ${QA_ROOT}/${split}.jsonl" >&2; exit 1; }
done

echo "==========================================================="
echo " SO-30B training:"
echo "   CONDA_ENV=${CONDA_ENV}"
echo "   PYTHON_BIN=${PYTHON_BIN}"
echo "   PYTHONPATH=${PYTHONPATH}"
echo "   ENV=${ENV_SUMMARY}"
echo "   ASSET_ROOT=${ASSET_ROOT}"
echo "   MODEL_ID=${MODEL_ID}"
echo "   QWEN3_OMNI_FORK=${QWEN3_OMNI_FORK:-<installed transformers>}"
echo "   NNODES=${NNODES}  NODE_RANK=${NODE_RANK}  NPROC=${NPROC}  GPUS=${GPUS}"
if [[ -n "${DEVICE_MAP:-}" ]]; then
  echo "   DEVICE_MAP=${DEVICE_MAP}  → single replica across ${NPROC} GPUs"
  echo "   global_bs = ${BATCH_SIZE} × ${GRAD_ACCUM_STEPS} × 1 (single replica) = $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
else
  echo "   global_bs = ${BATCH_SIZE} × ${GRAD_ACCUM_STEPS} × $((NNODES * NPROC)) = $((BATCH_SIZE * GRAD_ACCUM_STEPS * NNODES * NPROC))"
fi
echo "   START_STAGE=${START_STAGE}  END_STAGE=${END_STAGE}"
echo "   RUN_ROOT=${RUN_ROOT}"
echo "==========================================================="

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[check-only] Environment and input paths are valid; training was not started."
  exit 0
fi

run_train() {
  if [[ -n "${DEVICE_MAP:-}" ]]; then
    echo "[run_train] DEVICE_MAP=${DEVICE_MAP} → single python process (no torchrun)"
    CUDA_VISIBLE_DEVICES="${GPUS}" QWEN3_OMNI_FORK="${QWEN3_OMNI_FORK}" \
      "${PYTHON_BIN}" "${ROOT_DIR}/train_so_qa_qwen3.py" \
        --device-map "${DEVICE_MAP}" \
        "$@"
  else
    CUDA_VISIBLE_DEVICES="${GPUS}" QWEN3_OMNI_FORK="${QWEN3_OMNI_FORK}" \
      "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --nproc_per_node="${NPROC}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${ROOT_DIR}/train_so_qa_qwen3.py" "$@"
  fi
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
  --lora-target-prefixes model.layers
)
if (( USE_GRADIENT_CHECKPOINTING == 1 )); then
  common_args+=(--gradient-checkpointing)
  echo "[config] gradient_checkpointing = ENABLED (30B model 推荐打开)"
else
  echo "[config] gradient_checkpointing = DISABLED"
fi
if [[ -n "${QWEN_AUDIO_CACHE_MANIFEST}" ]]; then
  common_args+=(--audio-feature-cache-manifest "${QWEN_AUDIO_CACHE_MANIFEST}")
  echo "[config] audio feature cache = ${QWEN_AUDIO_CACHE_MANIFEST}"
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  common_args+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VALID_SAMPLES:-}" ]]; then
  common_args+=(--max-valid-samples "${MAX_VALID_SAMPLES}")
fi
if [[ "${VALID_GENERATE_FULL:-0}" == "1" ]]; then
  common_args+=(--valid-generate-full)
fi

if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
  echo "[stage1] projector_only (${STAGE1_EPOCHS} epochs, lr=${STAGE1_LR})"
  stage1_extra=()
  if [[ -n "${STAGE1_RESUME_CKPT:-}" ]]; then
    stage1_extra+=(--resume-checkpoint-path "${STAGE1_RESUME_CKPT}")
    if [[ "${STAGE1_RESUME_MODEL_ONLY:-0}" == "1" ]]; then
      stage1_extra+=(--resume-model-only)
    fi
  fi
  echo "[stage1] projector_only (${STAGE1_EPOCHS} epochs, BS=${STAGE1_BATCH_SIZE} GRAD_ACCUM=${STAGE1_GRAD_ACCUM} → global=$((STAGE1_BATCH_SIZE*STAGE1_GRAD_ACCUM*DATA_PARALLEL_WORLD)))"
  run_train \
    "${common_args[@]}" \
    --batch-size "${STAGE1_BATCH_SIZE}" \
    --grad-accum-steps "${STAGE1_GRAD_ACCUM}" \
    --projector-only \
    --lr "${STAGE1_LR}" \
    --projector-lr "${STAGE1_PROJECTOR_LR}" \
    --epochs "${STAGE1_EPOCHS}" \
    --output-dir "${STAGE1_DIR}" \
    "${stage1_extra[@]}"
fi

if (( START_STAGE <= 2 && END_STAGE >= 2 )); then
  if [[ ! -f "${STAGE2_RESUME_CKPT}" ]]; then
    echo "Missing stage2 resume checkpoint: ${STAGE2_RESUME_CKPT}" >&2; exit 1
  fi
  echo "[stage2] encoder_lora (${STAGE2_EPOCHS} epochs, lora_lr=${STAGE2_LORA_LR}, BS=${STAGE2_BATCH_SIZE} GRAD_ACCUM=${STAGE2_GRAD_ACCUM} → global=$((STAGE2_BATCH_SIZE*STAGE2_GRAD_ACCUM*DATA_PARALLEL_WORLD)))"
  stage2_router_args=()
  if (( STAGE2_TRAIN_MOE_ROUTER == 1 )); then
    stage2_router_args+=(
      --train-moe-router
      --moe-router-lr "${MOE_ROUTER_LR}"
      --moe-router-aux-loss-coef "${MOE_ROUTER_AUX_LOSS_COEF}"
    )
  fi
  run_train \
    "${common_args[@]}" \
    --batch-size "${STAGE2_BATCH_SIZE}" \
    --grad-accum-steps "${STAGE2_GRAD_ACCUM}" \
    --encoder-lora \
    --resume-checkpoint-path "${STAGE2_RESUME_CKPT}" \
    --resume-model-only \
    --lr "${STAGE2_LR}" \
    --lora-lr "${STAGE2_LORA_LR}" \
    --projector-lr "${STAGE2_PROJECTOR_LR}" \
    "${stage2_router_args[@]}" \
    --epochs "${STAGE2_EPOCHS}" \
    --output-dir "${STAGE2_DIR}"
fi

if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
  if [[ ! -f "${STAGE3_RESUME_CKPT}" ]]; then
    echo "Missing stage3 resume checkpoint: ${STAGE3_RESUME_CKPT}" >&2; exit 1
  fi
  echo "[stage3] beats_lora (${STAGE3_EPOCHS} epochs, BS=${STAGE3_BATCH_SIZE} GRAD_ACCUM=${STAGE3_GRAD_ACCUM} → global=$((STAGE3_BATCH_SIZE*STAGE3_GRAD_ACCUM*DATA_PARALLEL_WORLD)))"
  stage3_extra=()
  if [[ "${STAGE3_RESUME_MODEL_ONLY:-1}" == "1" ]]; then
    stage3_extra+=(--resume-model-only)
  fi
  if (( STAGE3_TRAIN_MOE_ROUTER == 1 )); then
    stage3_extra+=(
      --train-moe-router
      --moe-router-lr "${MOE_ROUTER_LR}"
      --moe-router-aux-loss-coef "${MOE_ROUTER_AUX_LOSS_COEF}"
    )
  fi
  run_train \
    "${common_args[@]}" \
    --batch-size "${STAGE3_BATCH_SIZE}" \
    --grad-accum-steps "${STAGE3_GRAD_ACCUM}" \
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

echo "All requested stages finished. RUN_ROOT=${RUN_ROOT}"
