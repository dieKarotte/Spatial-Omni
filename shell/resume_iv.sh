#!/usr/bin/env bash
# Shortcut: resume an IV / Neural-IV stage2 training run with autorestart,
# full optimizer/scheduler state restored.
#
# Configure via env vars:
#   RUN_ROOT     — run directory (default: runs/iv_neural_iv)
#   SPATIAL_ENCODER_TYPE  — iv | neural_iv (default: neural_iv)
#
# Just invoke this shell with no args; it'll pick the latest step_*.pt
# from the run dir and continue. If it crashes it auto-restarts up to 10x.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/iv_neural_iv}"
STAGE2_CKPT_DIR="${RUN_ROOT}/stage2_encoder_lora/checkpoints"

# Always pick the latest step_*.pt (not best_trainable.pt, which may lag
# behind step ckpts by a few epochs).
LATEST=$(ls "${STAGE2_CKPT_DIR}"/step_*_trainable.pt 2>/dev/null | sort -V | tail -n 1)
if [[ -z "${LATEST}" || ! -s "${LATEST}" ]]; then
  echo "[ERROR] no step_*_trainable.pt under ${STAGE2_CKPT_DIR}" >&2
  exit 1
fi
echo "[resume-neural-iv] using latest step ckpt: ${LATEST}"

RUN_ROOT="${RUN_ROOT}" \
SPATIAL_ENCODER_TYPE=neural_iv \
START_STAGE=2 \
QA_ROOT=${SO_DATASET_ROOT} \
SAVE_EVERY_N_OPT_STEPS="${SAVE_EVERY_N_OPT_STEPS:-500}" \
STAGE2_RESUME_CKPT="${LATEST}" \
STAGE2_RESUME_MODEL_ONLY=0 \
  bash "${ROOT_DIR}/shell/train_with_autorestart.sh" "$@"
