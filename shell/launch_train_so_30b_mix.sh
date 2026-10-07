#!/usr/bin/env bash
# Explicit local replay manifests; check mode does not start training.
set -euo pipefail
MODE="${1:-check}"
case "${MODE}" in check|smoke|full) ;; *) echo "Usage: $0 [check|smoke|full]" >&2; exit 2;; esac
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
: "${SO_BASE_MODEL:?Set SO_BASE_MODEL to the downloaded base model directory}"
: "${SO_ENCODER_CKPT:?Set SO_ENCODER_CKPT to SO-Encoder.pt}"
: "${SO_DATASET_ROOT:?Set SO_DATASET_ROOT to the directory containing qa/ and audio/}"
: "${REPLAY_QA_ROOT:?Set REPLAY_QA_ROOT to a local replay JSONL file or split directory}"
: "${REPLAY_AUDIO_ROOT:?Set REPLAY_AUDIO_ROOT to the replay audio directory}"
: "${RESUME_CKPT:?Set RESUME_CKPT to the SO-30B adaptation checkpoint}"
"${PYTHON_BIN}" - "${SO_BASE_MODEL}" "${SO_ENCODER_CKPT}" "${SO_DATASET_ROOT}" "${REPLAY_QA_ROOT}" "${REPLAY_AUDIO_ROOT}" "${RESUME_CKPT}" <<'CHECK'
import sys
from pathlib import Path
import torch, transformers, peft
for raw in sys.argv[1:]:
    if not Path(raw).exists():
        raise SystemExit(f"Input does not exist: {raw}")
for name in ("train", "valid"):
    if not (Path(sys.argv[3]) / "qa" / f"{name}.jsonl").is_file():
        raise SystemExit(f"Missing qa/{name}.jsonl under {sys.argv[3]}")
print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}")
CHECK
if [[ "${MODE}" == smoke ]]; then
  NPROC="${NPROC:-2}"
  TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
  LIMITS=(--max-train-samples 128 --max-valid-samples 16)
else
  NPROC="${NPROC:-8}"
  LIMITS=()
fi
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/runs/so_30b_mix/$(date +%Y%m%d_%H%M%S)_${MODE}}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse output directory: ${OUTPUT_DIR}; choose a new OUTPUT_DIR." >&2
  exit 1
fi
# Scale the default learning rate from an effective batch of 32.
LR="${LR:-$("${PYTHON_BIN}" -c 'import sys; print(1e-5 * int(sys.argv[1]) * int(sys.argv[2]) * int(sys.argv[3]) / 32)' "${NPROC}" "${BATCH_SIZE:-1}" "${GRAD_ACCUM_STEPS:-4}")}"
ARGS=(
  --model-id "${SO_BASE_MODEL}"
  --beats-checkpoint "${SO_ENCODER_CKPT}" --beats-repo "${ROOT_DIR}"
  --qa-root "${SO_DATASET_ROOT}/qa" --audio-roots "${SO_DATASET_ROOT}" "${REPLAY_AUDIO_ROOT}"
  --replay-qa-root "${REPLAY_QA_ROOT}" --mixed-spatial-replay
  --train-subset-ratio "${TRAIN_SUBSET_RATIO:-0.2}" --mix-full-replay
  --replay-max-text-chars 8000 --null-alignment-weight 0.05
  --beats-lora --train-moe-router --moe-router-aux-loss-coef 0.001
  --lora-target-prefixes model.layers --lora-target-modules q_proj k_proj v_proj o_proj
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
  --resume-checkpoint-path "${RESUME_CKPT}" --resume-model-only
  --batch-size "${BATCH_SIZE:-1}" --grad-accum-steps "${GRAD_ACCUM_STEPS:-4}"
  --num-workers "${NUM_WORKERS:-$(( ($(nproc --all) + 2 * NPROC - 1) / (2 * NPROC) ))}"
  --epochs "${EPOCHS:-2}" --lr "${LR:-1e-5}" --spatial-null-lr "${SPATIAL_NULL_LR:-${LR}}"
  --dtype bfloat16 --attn-impl "${ATTN_IMPL:-sdpa}" --gradient-checkpointing
  --save-every-n-optimizer-steps 500 --save-every-epoch --seed "${SEED:-42}"
  --output-dir "${OUTPUT_DIR}" "${LIMITS[@]}"
)
COMMAND=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC}" "${ROOT_DIR}/train_so_qa_qwen3.py" "${ARGS[@]}")
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "${MODE}" == check ]]; then
  echo "[check-only] Inputs checked; training was not started."
  exit 0
fi
exec "${COMMAND[@]}"
