#!/usr/bin/env bash
# Auto-restart wrapper for long-running IV / Neural-IV training.
#
# Why this exists:
#   NCCL hangs / CUDA driver bugs / single-node transient failures are
#   essentially unavoidable in multi-day 8-GPU runs. Instead of babysitting
#   the screen at 3am, this script:
#     1. Runs the training launch command
#     2. If it exits non-zero (NCCL timeout, OOM, SIGSEGV, etc.), waits
#        COOLDOWN_S seconds (to let NCCL/CUDA clean up)
#     3. Re-discovers the most recent step_*_trainable.pt and resumes from it
#     4. Bails out after MAX_RETRIES attempts (to avoid infinite restart
#        loops if the problem is systematic — bad data, broken ckpt, etc.)
#
# Usage:
#   # First invocation (clean start):
#   SPATIAL_ENCODER_TYPE=neural_iv bash shell/train_with_autorestart.sh
#
#   # Resume an existing run (e.g. after you've already kicked off once
#   # and it crashed):
#   SPATIAL_ENCODER_TYPE=neural_iv START_STAGE=2 bash shell/train_with_autorestart.sh
#
# All env vars understood by launch_train_spatial_iv_qa.sh are forwarded
# transparently, e.g. BATCH_SIZE, STAGE2_LR, etc.

set -uo pipefail   # intentionally NOT -e: we want to trap non-zero and retry

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

MAX_RETRIES="${MAX_RETRIES:-10}"
COOLDOWN_S="${COOLDOWN_S:-60}"
LAUNCHER="${LAUNCHER:-shell/launch_train_spatial_iv_qa.sh}"

# Where checkpoints live. Default mirrors launch_train_spatial_iv_qa.sh.
SPATIAL_ENCODER_TYPE="${SPATIAL_ENCODER_TYPE:-iv}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/iv_${SPATIAL_ENCODER_TYPE}}"
STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1_projector}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2_encoder_lora}"
START_STAGE="${START_STAGE:-1}"

# NCCL robustness defaults (can override from env). Chosen to reduce
# incidence of hangs on shared cluster storage / PCIe-only nodes.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"
export SO_NCCL_TIMEOUT_MIN="${SO_NCCL_TIMEOUT_MIN:-120}"

pick_latest_checkpoint() {
  # $1 = stage dir (STAGE1_DIR or STAGE2_DIR)
  local stage_dir="$1"
  local ckpt_dir="${stage_dir}/checkpoints"
  if [[ ! -d "${ckpt_dir}" ]]; then
    echo ""
    return
  fi
  # Prefer step_*_trainable.pt (largest number), fall back to last_trainable.pt
  local latest
  latest=$(ls "${ckpt_dir}"/step_*_trainable.pt 2>/dev/null \
           | sort -V | tail -n 1)
  if [[ -n "${latest}" && -s "${latest}" ]]; then
    echo "${latest}"
    return
  fi
  if [[ -s "${ckpt_dir}/last_trainable.pt" ]]; then
    echo "${ckpt_dir}/last_trainable.pt"
    return
  fi
  echo ""
}

for attempt in $(seq 1 "${MAX_RETRIES}"); do
  echo ""
  echo "==========================================================="
  echo "[autorestart] attempt ${attempt}/${MAX_RETRIES}"
  echo "[autorestart] encoder=${SPATIAL_ENCODER_TYPE} stage=${START_STAGE}"
  echo "==========================================================="

  extra_env=()

  if (( attempt > 1 )); then
    # On retry: figure out which stage was running and resume from latest ckpt
    if (( START_STAGE == 1 )); then
      stage_dir="${STAGE1_DIR}"
    else
      stage_dir="${STAGE2_DIR}"
    fi

    latest_ckpt="$(pick_latest_checkpoint "${stage_dir}")"
    if [[ -z "${latest_ckpt}" ]]; then
      echo "[autorestart] no checkpoint to resume from in ${stage_dir}" >&2
      echo "[autorestart] aborting — initial attempt must have crashed before" >&2
      echo "              the first save; fix the error and restart manually." >&2
      exit 1
    fi
    echo "[autorestart] resuming from: ${latest_ckpt}"

    # Preserve optimizer / scheduler / step counter so LR schedule continues.
    # If you want a fresh optimizer (e.g. you changed LR), set RESUME_MODEL_ONLY=1.
    if [[ "${RESUME_MODEL_ONLY:-0}" == "1" ]]; then
      extra_env+=("STAGE${START_STAGE}_RESUME_MODEL_ONLY=1")
    else
      # Autorestart default: full resume (keep optimizer)
      extra_env+=("STAGE${START_STAGE}_RESUME_MODEL_ONLY=0")
    fi

    if (( START_STAGE == 1 )); then
      extra_env+=("STAGE1_RESUME_CKPT=${latest_ckpt}")
    else
      extra_env+=("STAGE2_RESUME_CKPT=${latest_ckpt}")
    fi
  fi

  # shellcheck disable=SC2068
  env ${extra_env[@]+"${extra_env[@]}"} \
      SPATIAL_ENCODER_TYPE="${SPATIAL_ENCODER_TYPE}" \
      START_STAGE="${START_STAGE}" \
      bash "${ROOT_DIR}/${LAUNCHER}" "$@"
  rc=$?

  if (( rc == 0 )); then
    echo "[autorestart] training finished successfully (attempt ${attempt})"
    exit 0
  fi

  echo "[autorestart] attempt ${attempt} failed with exit code ${rc}" >&2
  echo "[autorestart] cooling down ${COOLDOWN_S}s before retry..." >&2

  # Kill any dangling torchrun / python / nvidia-smi zombies on this box.
  pkill -9 -f "torchrun" 2>/dev/null || true
  pkill -9 -f "train_spatial_iv_qa" 2>/dev/null || true
  sleep "${COOLDOWN_S}"
done

echo "[autorestart] giving up after ${MAX_RETRIES} attempts" >&2
exit 2
