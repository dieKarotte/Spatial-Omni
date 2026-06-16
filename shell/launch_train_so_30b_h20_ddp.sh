#!/usr/bin/env bash
# H20 (95GB) DDP launcher for SO-30B (SO-Encoder + Qwen3-Omni-MoE-30B-A3B).
#
# Differences vs shell/launch_train_so_30b.sh (which also supports device_map=auto):
#   * 30B BF16 (~60GB) FITS on a single 95GB GPU (e.g. H20), so we run TRUE DDP:
#     one replica per GPU, full data parallelism, NO model sharding. (On 40GB
#     A100 you instead need DEVICE_MAP=auto or DeepSpeed ZeRO-3 — see README §4b.)
#   * If flash-attn 2 is installed, use flash_attention_2 for speed + memory.
#     (Falls back cleanly: set ATTN_IMPL=sdpa if flash-attn is unavailable.)
#   * Per-GPU batch sizes are lower (model is replicated, not sharded); global
#     batch is preserved via gradient accumulation (= 64 across stages).
#
# Usage:
#   conda activate spatial-omni-30b
#   GPUS=0,1,2,3,4,5,6,7 \
#   MODEL_ID=/path/to/Qwen3-Omni-30B-A3B-Instruct \
#   QWEN3_TRANSFORMERS_FORK=/path/to/transformers/src \  # only if using a custom fork
#   SO_ENCODER_CKPT=/path/to/so_encoder/best.pt \
#   SO_DATASET_ROOT=/path/to/SO-Dataset/qa \
#   RUN_ROOT=./runs/so30b_h20_ddp \
#     bash shell/launch_train_so_30b_h20_ddp.sh
#
# Smoke test (32 samples, stage1 only, BS=1):
#   MAX_TRAIN_SAMPLES=32 STAGE1_EPOCHS=1 STAGE1_BATCH_SIZE=1 STAGE1_GRAD_ACCUM=1 \
#     RUN_ROOT=./runs/so30b_ddp_smoke \
#     bash shell/launch_train_so_30b_h20_ddp.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# ------------------------------------------------------------------
# Force DDP: do NOT export DEVICE_MAP. The base script's run_train()
# branches on ${DEVICE_MAP:-} being non-empty (device_map=auto = single
# sharded replica; empty = torchrun DDP).
# ------------------------------------------------------------------
unset DEVICE_MAP

# ------------------------------------------------------------------
# Attention impl. flash_attention_2 needs flash-attn installed; if you don't
# have it, override with ATTN_IMPL=sdpa.
# ------------------------------------------------------------------
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# Per-GPU batch sizes (DDP, 8 replicas of full 30B model on 95GB cards).
# Conservative starting point — bump after a smoke test confirms headroom.
#   stage1 (projector_only):  lightest activations
#   stage2 (encoder_lora):    + LLM LoRA optimizer state
#   stage3 (beats_lora):      + SO-Encoder unfrozen (extra activations)
# global_bs = per_gpu_bs * grad_accum * 8 GPUs = 64 across all stages.
export STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
export STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-1}"
export STAGE1_GRAD_ACCUM="${STAGE1_GRAD_ACCUM:-2}"   # 4 * 2 * 8 = 64
export STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-4}"   # 2 * 4 * 8 = 64
export STAGE3_GRAD_ACCUM="${STAGE3_GRAD_ACCUM:-8}"   # 1 * 8 * 8 = 64

# NCCL safety: networked/NFS QA roots can stall checkpoint saves; widen the
# collective timeout. (The trainer reads SO_NCCL_TIMEOUT_MIN.)
export SO_NCCL_TIMEOUT_MIN="${SO_NCCL_TIMEOUT_MIN:-120}"

# Default RUN_ROOT differs from the device_map run so artefacts don't collide.
export RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/so30b_h20_ddp}"

echo "==========================================================="
echo " SO-30B H20 DDP launcher → forwarding to launch_train_so_30b.sh"
echo "   ATTN_IMPL=${ATTN_IMPL}"
echo "   per-GPU BS: stage1=${STAGE1_BATCH_SIZE} stage2=${STAGE2_BATCH_SIZE} stage3=${STAGE3_BATCH_SIZE}"
echo "   GRAD_ACCUM: stage1=${STAGE1_GRAD_ACCUM} stage2=${STAGE2_GRAD_ACCUM} stage3=${STAGE3_GRAD_ACCUM}"
echo "   SO_NCCL_TIMEOUT_MIN=${SO_NCCL_TIMEOUT_MIN}"
echo "   RUN_ROOT=${RUN_ROOT}"
echo "==========================================================="

exec bash "${ROOT_DIR}/shell/launch_train_so_30b.sh" "$@"
