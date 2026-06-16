#!/usr/bin/env bash
# 三阶段训练 SO-30B（SO-Encoder + Qwen3-Omni-MoE-30B-A3B）
# (clone of shell/launch_train_so_7b.sh, adapted for Qwen3)
#
# 需要：spatial-omni-30b conda env（transformers w/ qwen3_omni_moe，可选 flash-attn）。
#       如用自定义 transformers fork，设 QWEN3_TRANSFORMERS_FORK 指向其 src/。
# 模型：Qwen3-Omni-30B-A3B-Instruct (MoE, 128 experts, top-8, hidden=2048)
#
# 显存：30B BF16 ≈ 60GB，单卡 40GB A100 不够；后续 stage3 会接入 DeepSpeed
# ZeRO-3（见 configs/ds_zero3_so30b.json）。当前脚本走 torchrun，需要 8 卡 ZeRO-3
# 或 device_map=auto 才能跑起来。先用 BATCH_SIZE=1 GRAD_ACCUM_STEPS=8 试探。
#
# 用法：
#   bash shell/launch_train_so_30b.sh                # 单机 8 卡三阶段
#   START_STAGE=2 bash shell/launch_train_so_30b.sh   # 从 stage2 开始

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# ------------------------------------------------------------------
# 分布式 / GPU
# ------------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(python -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x]))' "${GPUS}")}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29577}"
START_STAGE="${START_STAGE:-1}"

if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "[ERROR] NNODES=${NNODES} > 1 but MASTER_ADDR is loopback" >&2
  exit 1
fi

# ------------------------------------------------------------------
# Qwen3 fork bootstrap
# ------------------------------------------------------------------
QWEN3_OMNI_FORK="${QWEN3_OMNI_FORK:-${QWEN3_TRANSFORMERS_FORK}}"
export QWEN3_OMNI_FORK

# ------------------------------------------------------------------
# 数据 / checkpoint 路径
# ------------------------------------------------------------------
QA_ROOT="${QA_ROOT:-${SO_DATASET_ROOT}}"
BEATS_CKPT="${BEATS_CKPT:-${SO_ENCODER_CKPT:-${SO_BEATS_REPO}/checkpoints/so_encoder/best.pt}}"
BEATS_REPO="${BEATS_REPO:-${SO_BEATS_REPO}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

# ------------------------------------------------------------------
# 输出目录
# ------------------------------------------------------------------
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/so_30b}"
STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1_projector}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2_encoder_lora}"
STAGE3_DIR="${STAGE3_DIR:-${RUN_ROOT}/stage3_beats_lora}"

STAGE2_RESUME_CKPT="${STAGE2_RESUME_CKPT:-${STAGE1_DIR}/checkpoints/best_trainable.pt}"
STAGE3_RESUME_CKPT="${STAGE3_RESUME_CKPT:-${STAGE2_DIR}/checkpoints/best_trainable.pt}"

# ------------------------------------------------------------------
# batch / DataLoader / checkpoint 频率
# 实测真实 easy 数据 + device_map=auto + GC 显存上限：
#   8 卡：stage1 BS<=32, stage2 BS<=16, stage3 BS<=8（unfreeze BEATs 后激活更大）
#   4 卡：stage1/2 BS<=16, stage3 BS<=4
# 默认推荐 8 卡，per-stage BS 自适应。global_bs 通过 GRAD_ACCUM 调到 64。
# ------------------------------------------------------------------
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-${BATCH_SIZE:-16}}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${BATCH_SIZE:-8}}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-${BATCH_SIZE:-4}}"
STAGE1_GRAD_ACCUM="${STAGE1_GRAD_ACCUM:-${GRAD_ACCUM_STEPS:-4}}"   # 16*4*1 = 64
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-${GRAD_ACCUM_STEPS:-8}}"   # 8*8*1  = 64
STAGE3_GRAD_ACCUM="${STAGE3_GRAD_ACCUM:-${GRAD_ACCUM_STEPS:-16}}"  # 4*16*1 = 64
# Legacy single-knob fallback (used by smoke / overrides):
BATCH_SIZE="${BATCH_SIZE:-${STAGE1_BATCH_SIZE}}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-${STAGE1_GRAD_ACCUM}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAVE_EVERY_N_OPT_STEPS="${SAVE_EVERY_N_OPT_STEPS:-500}"
VALID_EVERY_N_OPT_STEPS="${VALID_EVERY_N_OPT_STEPS:-500}"

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-1}"
QWEN_AUDIO_CACHE_MANIFEST="${QWEN_AUDIO_CACHE_MANIFEST:-}"

# ------------------------------------------------------------------
# 训练 schedule
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# LoRA — only attention proj layers; expert MLPs (SparseMoeBlock) untouched.
# ------------------------------------------------------------------
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES=(${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj})

# ------------------------------------------------------------------
# 前置检查
# ------------------------------------------------------------------
if [[ ! -f "${BEATS_CKPT}" ]]; then echo "Missing BEATs ckpt: ${BEATS_CKPT}" >&2; exit 1; fi
if [[ ! -d "${QA_ROOT}" ]]; then echo "Missing QA root: ${QA_ROOT}" >&2; exit 1; fi
if [[ ! -d "${MODEL_ID}" ]]; then echo "Missing model dir: ${MODEL_ID}" >&2; exit 1; fi
if [[ ! -d "${QWEN3_OMNI_FORK}" ]]; then echo "Missing transformers fork: ${QWEN3_OMNI_FORK}" >&2; exit 1; fi
for split in train valid test; do
  [[ -f "${QA_ROOT}/${split}.jsonl" ]] || { echo "Missing ${QA_ROOT}/${split}.jsonl" >&2; exit 1; }
done

echo "==========================================================="
echo " SO-30B training:"
echo "   MODEL_ID=${MODEL_ID}"
echo "   QWEN3_OMNI_FORK=${QWEN3_OMNI_FORK}"
echo "   NNODES=${NNODES}  NODE_RANK=${NODE_RANK}  NPROC=${NPROC}  GPUS=${GPUS}"
if [[ -n "${DEVICE_MAP:-}" ]]; then
  echo "   DEVICE_MAP=${DEVICE_MAP}  → single replica across ${NPROC} GPUs"
  echo "   global_bs = ${BATCH_SIZE} × ${GRAD_ACCUM_STEPS} × 1 (single replica) = $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
else
  echo "   global_bs = ${BATCH_SIZE} × ${GRAD_ACCUM_STEPS} × $((NNODES * NPROC)) = $((BATCH_SIZE * GRAD_ACCUM_STEPS * NNODES * NPROC))"
fi
echo "   START_STAGE=${START_STAGE}"
echo "   RUN_ROOT=${RUN_ROOT}"
echo "==========================================================="

run_train() {
  if [[ -n "${DEVICE_MAP:-}" ]]; then
    # device_map mode: single process holds the whole sharded model.
    # NPROC is forced to 1 (8 cards run one replica via accelerate).
    echo "[run_train] DEVICE_MAP=${DEVICE_MAP} → single python process (no torchrun)"
    CUDA_VISIBLE_DEVICES="${GPUS}" QWEN3_OMNI_FORK="${QWEN3_OMNI_FORK}" \
      python "${ROOT_DIR}/train_so_qa_qwen3.py" \
        --device-map "${DEVICE_MAP}" \
        "$@"
  else
    # DDP mode: one replica per GPU (only viable with ZeRO-3 / FSDP wired in).
    CUDA_VISIBLE_DEVICES="${GPUS}" QWEN3_OMNI_FORK="${QWEN3_OMNI_FORK}" \
      torchrun \
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
  # NOTE: Qwen3 wrapper makes model.thinker == model, so the LoRA prefix is
  # `model.layers` directly (not `thinker.model` like Qwen2.5).
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
if [[ "${VALID_GENERATE_FULL:-0}" == "1" ]]; then
  common_args+=(--valid-generate-full)
fi

# ------------------------------------------------------------------
# Stage 1: projector_only
# ------------------------------------------------------------------
if (( START_STAGE <= 1 )); then
  echo "[stage1] projector_only (${STAGE1_EPOCHS} epochs, lr=${STAGE1_LR})"
  stage1_extra=()
  if [[ -n "${STAGE1_RESUME_CKPT:-}" ]]; then
    stage1_extra+=(--resume-checkpoint-path "${STAGE1_RESUME_CKPT}")
    if [[ "${STAGE1_RESUME_MODEL_ONLY:-0}" == "1" ]]; then
      stage1_extra+=(--resume-model-only)
    fi
  fi
  echo "[stage1] projector_only (${STAGE1_EPOCHS} epochs, BS=${STAGE1_BATCH_SIZE} GRAD_ACCUM=${STAGE1_GRAD_ACCUM} → global=$((STAGE1_BATCH_SIZE*STAGE1_GRAD_ACCUM)))"
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

# ------------------------------------------------------------------
# Stage 2: encoder_lora
# ------------------------------------------------------------------
if (( START_STAGE <= 2 )); then
  if [[ ! -f "${STAGE2_RESUME_CKPT}" ]]; then
    echo "Missing stage2 resume checkpoint: ${STAGE2_RESUME_CKPT}" >&2; exit 1
  fi
  echo "[stage2] encoder_lora (${STAGE2_EPOCHS} epochs, lora_lr=${STAGE2_LORA_LR}, BS=${STAGE2_BATCH_SIZE} GRAD_ACCUM=${STAGE2_GRAD_ACCUM} → global=$((STAGE2_BATCH_SIZE*STAGE2_GRAD_ACCUM)))"
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
    --epochs "${STAGE2_EPOCHS}" \
    --output-dir "${STAGE2_DIR}"
fi

# ------------------------------------------------------------------
# Stage 3: beats_lora
# ------------------------------------------------------------------
if (( START_STAGE <= 3 )); then
  if [[ ! -f "${STAGE3_RESUME_CKPT}" ]]; then
    echo "Missing stage3 resume checkpoint: ${STAGE3_RESUME_CKPT}" >&2; exit 1
  fi
  echo "[stage3] beats_lora (${STAGE3_EPOCHS} epochs, BS=${STAGE3_BATCH_SIZE} GRAD_ACCUM=${STAGE3_GRAD_ACCUM} → global=$((STAGE3_BATCH_SIZE*STAGE3_GRAD_ACCUM)))"
  stage3_extra=()
  if [[ "${STAGE3_RESUME_MODEL_ONLY:-1}" == "1" ]]; then
    stage3_extra+=(--resume-model-only)
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
