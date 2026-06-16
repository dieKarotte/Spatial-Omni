#!/usr/bin/env bash
# Curriculum training: SO-30B (SO-Encoder + Qwen3-Omni-MoE-30B-A3B) on H20 DDP.
#
#   easy   : full 3-stage (projector_only -> encoder_lora -> beats_lora)
#   medium : continue stage3 (beats_lora) from easy/stage3 best
#   hard   : continue stage3 (beats_lora) from medium/stage3 best
#
# Each phase re-uses shell/launch_train_so_30b_h20_ddp.sh (which forwards to
# launch_train_so_30b.sh in DDP mode). The three difficulty levels are just
# three QA roots you provide; the medium/hard phases continue stage3 from the
# previous phase's best checkpoint with a lower LR so they don't stomp the
# earlier adapter/LoRA weights.
#
# Provide the three QA roots (each must contain train.jsonl + valid.jsonl):
#   EASY_QA=/path/to/qa_easy
#   MEDIUM_QA=/path/to/qa_medium      # e.g. medium + a small easy replay mix
#   HARD_QA=/path/to/qa_hard          # e.g. hard + a small medium/easy replay mix
# (A pure "hard" split with no train/valid won't work for the hard phase —
#  use a hard+replay mixture that has train/valid, or SKIP_HARD=1.)
#
# Usage:
#   conda activate spatial-omni-30b
#   GPUS=0,1,2,3,4,5,6,7 \
#   MODEL_ID=/path/to/Qwen3-Omni-30B-A3B-Instruct \
#   SO_ENCODER_CKPT=/path/to/so_encoder/best.pt \
#   EASY_QA=/path/to/qa_easy MEDIUM_QA=/path/to/qa_medium HARD_QA=/path/to/qa_hard \
#   RUN_ROOT_BASE=./runs/so30b_curriculum \
#     bash shell/launch_train_so_30b_curriculum.sh
#
# Skip phases via env vars:
#   SKIP_EASY=1  SKIP_MEDIUM=1  SKIP_HARD=1
#   EASY_START_STAGE=2  # resume easy from stage 2 (e.g. stage1 already done)
#
# Outputs (under ${RUN_ROOT_BASE:-./runs/so30b_curriculum}):
#   easy/stage1_projector/checkpoints/best_trainable.pt
#   easy/stage2_encoder_lora/checkpoints/best_trainable.pt
#   easy/stage3_beats_lora/checkpoints/best_trainable.pt
#   medium/stage3_beats_lora/checkpoints/best_trainable.pt
#   hard/stage3_beats_lora/checkpoints/best_trainable.pt

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# ------------------------------------------------------------------
# QA roots for the three difficulty levels (REQUIRED unless the phase is
# skipped). No internal defaults — provide your own paths.
# ------------------------------------------------------------------
EASY_QA="${EASY_QA:-${SO_DATASET_ROOT:+${SO_DATASET_ROOT}/qa}}"
MEDIUM_QA="${MEDIUM_QA:-}"
HARD_QA="${HARD_QA:-}"

RUN_ROOT_BASE="${RUN_ROOT_BASE:-${ROOT_DIR}/runs/so30b_curriculum}"

# H20 DDP defaults (the H20 script fills these if missing; set here so the
# curriculum is self-documenting).
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
export STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-1}"
export STAGE1_GRAD_ACCUM="${STAGE1_GRAD_ACCUM:-2}"   # 4 * 2 * 8 = 64
export STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-4}"   # 2 * 4 * 8 = 64
export STAGE3_GRAD_ACCUM="${STAGE3_GRAD_ACCUM:-8}"   # 1 * 8 * 8 = 64

# Default epoch counts per phase (the H20 script picks these up too).
export STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
export STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
export STAGE3_EPOCHS="${STAGE3_EPOCHS:-3}"

# Medium/hard get fewer epochs since we only fine-tune stage3 further.
MEDIUM_STAGE3_EPOCHS="${MEDIUM_STAGE3_EPOCHS:-2}"
HARD_STAGE3_EPOCHS="${HARD_STAGE3_EPOCHS:-2}"

# Lower LRs for the medium/hard continuation phases to avoid stomping on the
# easy-trained adapter / LoRA weights.
MEDIUM_STAGE3_LR="${MEDIUM_STAGE3_LR:-2e-5}"
MEDIUM_STAGE3_LORA_LR="${MEDIUM_STAGE3_LORA_LR:-2e-5}"
MEDIUM_STAGE3_PROJECTOR_LR="${MEDIUM_STAGE3_PROJECTOR_LR:-1e-6}"
MEDIUM_STAGE3_BEATS_LR="${MEDIUM_STAGE3_BEATS_LR:-5e-7}"

HARD_STAGE3_LR="${HARD_STAGE3_LR:-1e-5}"
HARD_STAGE3_LORA_LR="${HARD_STAGE3_LORA_LR:-1e-5}"
HARD_STAGE3_PROJECTOR_LR="${HARD_STAGE3_PROJECTOR_LR:-1e-6}"
HARD_STAGE3_BEATS_LR="${HARD_STAGE3_BEATS_LR:-5e-7}"

H20_LAUNCHER="${ROOT_DIR}/shell/launch_train_so_30b_h20_ddp.sh"

# Pre-flight: data presence (each phase only checks its own splits)
check_qa() {
  local d="$1"; local label="$2"
  [[ -n "$d" ]] || { echo "[$label] QA dir not set (pass ${label^^}_QA=...)" >&2; exit 1; }
  [[ -d "$d" ]] || { echo "[$label] missing QA dir: $d" >&2; exit 1; }
  for split in train valid; do
    [[ -f "$d/$split.jsonl" ]] || { echo "[$label] missing $d/$split.jsonl" >&2; exit 1; }
  done
}

# ------------------------------------------------------------------
# Phase 1: easy - full 3-stage pipeline
# ------------------------------------------------------------------
EASY_RUN_ROOT="${EASY_RUN_ROOT:-${RUN_ROOT_BASE}/easy}"
if [[ "${SKIP_EASY:-0}" != "1" ]]; then
  check_qa "$EASY_QA" "easy"
  echo "==========================================================="
  echo " [Phase 1/3] EASY - 3-stage curriculum training"
  echo "   QA_ROOT  = ${EASY_QA}"
  echo "   RUN_ROOT = ${EASY_RUN_ROOT}"
  echo "   START_STAGE = ${EASY_START_STAGE:-1}"
  echo "==========================================================="
  QA_ROOT="${EASY_QA}" \
  RUN_ROOT="${EASY_RUN_ROOT}" \
  START_STAGE="${EASY_START_STAGE:-1}" \
    bash "${H20_LAUNCHER}"
else
  echo "[skip] SKIP_EASY=1, leaving ${EASY_RUN_ROOT} alone"
fi

EASY_STAGE3_BEST="${EASY_RUN_ROOT}/stage3_beats_lora/checkpoints/best_trainable.pt"

# ------------------------------------------------------------------
# Phase 2: medium - continue from easy stage3 best, beats_lora
# ------------------------------------------------------------------
MEDIUM_RUN_ROOT="${MEDIUM_RUN_ROOT:-${RUN_ROOT_BASE}/medium}"
if [[ "${SKIP_MEDIUM:-0}" != "1" ]]; then
  check_qa "$MEDIUM_QA" "medium"
  if [[ ! -f "$EASY_STAGE3_BEST" ]]; then
    echo "[medium] missing easy stage3 best ckpt: $EASY_STAGE3_BEST" >&2
    echo "         (set SKIP_MEDIUM=1 to skip this phase, or run easy first)" >&2
    exit 1
  fi
  echo "==========================================================="
  echo " [Phase 2/3] MEDIUM - stage3 continue"
  echo "   QA_ROOT  = ${MEDIUM_QA}"
  echo "   RUN_ROOT = ${MEDIUM_RUN_ROOT}"
  echo "   resume   = ${EASY_STAGE3_BEST}"
  echo "   epochs=${MEDIUM_STAGE3_EPOCHS}  lr=${MEDIUM_STAGE3_LR}  lora_lr=${MEDIUM_STAGE3_LORA_LR}"
  echo "==========================================================="
  QA_ROOT="${MEDIUM_QA}" \
  RUN_ROOT="${MEDIUM_RUN_ROOT}" \
  START_STAGE=3 \
  STAGE3_RESUME_CKPT="${EASY_STAGE3_BEST}" \
  STAGE3_RESUME_MODEL_ONLY=1 \
  STAGE3_EPOCHS="${MEDIUM_STAGE3_EPOCHS}" \
  STAGE3_LR="${MEDIUM_STAGE3_LR}" \
  STAGE3_LORA_LR="${MEDIUM_STAGE3_LORA_LR}" \
  STAGE3_PROJECTOR_LR="${MEDIUM_STAGE3_PROJECTOR_LR}" \
  STAGE3_BEATS_LR="${MEDIUM_STAGE3_BEATS_LR}" \
    bash "${H20_LAUNCHER}"
else
  echo "[skip] SKIP_MEDIUM=1, leaving ${MEDIUM_RUN_ROOT} alone"
fi

MEDIUM_STAGE3_BEST="${MEDIUM_RUN_ROOT}/stage3_beats_lora/checkpoints/best_trainable.pt"

# ------------------------------------------------------------------
# Phase 3: hard - continue from medium stage3 best
# ------------------------------------------------------------------
HARD_RUN_ROOT="${HARD_RUN_ROOT:-${RUN_ROOT_BASE}/hard}"
if [[ "${SKIP_HARD:-0}" != "1" ]]; then
  check_qa "$HARD_QA" "hard"
  if [[ ! -f "$MEDIUM_STAGE3_BEST" ]]; then
    echo "[hard] missing medium stage3 best ckpt: $MEDIUM_STAGE3_BEST" >&2
    echo "       (set SKIP_HARD=1 to skip this phase, or run medium first)" >&2
    exit 1
  fi
  echo "==========================================================="
  echo " [Phase 3/3] HARD - stage3 continue"
  echo "   QA_ROOT  = ${HARD_QA}"
  echo "   RUN_ROOT = ${HARD_RUN_ROOT}"
  echo "   resume   = ${MEDIUM_STAGE3_BEST}"
  echo "   epochs=${HARD_STAGE3_EPOCHS}  lr=${HARD_STAGE3_LR}  lora_lr=${HARD_STAGE3_LORA_LR}"
  echo "==========================================================="
  QA_ROOT="${HARD_QA}" \
  RUN_ROOT="${HARD_RUN_ROOT}" \
  START_STAGE=3 \
  STAGE3_RESUME_CKPT="${MEDIUM_STAGE3_BEST}" \
  STAGE3_RESUME_MODEL_ONLY=1 \
  STAGE3_EPOCHS="${HARD_STAGE3_EPOCHS}" \
  STAGE3_LR="${HARD_STAGE3_LR}" \
  STAGE3_LORA_LR="${HARD_STAGE3_LORA_LR}" \
  STAGE3_PROJECTOR_LR="${HARD_STAGE3_PROJECTOR_LR}" \
  STAGE3_BEATS_LR="${HARD_STAGE3_BEATS_LR}" \
    bash "${H20_LAUNCHER}"
else
  echo "[skip] SKIP_HARD=1, leaving ${HARD_RUN_ROOT} alone"
fi

echo ""
echo "==========================================================="
echo " Curriculum complete."
echo "   easy   = ${EASY_RUN_ROOT}"
echo "   medium = ${MEDIUM_RUN_ROOT}"
echo "   hard   = ${HARD_RUN_ROOT}"
echo "==========================================================="
