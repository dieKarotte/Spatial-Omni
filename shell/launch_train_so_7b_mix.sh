#!/usr/bin/env bash
# Continue SO-7B training with local mono replay. See docs/training.md.
set -euo pipefail

MODE="${1:-check}"
case "${MODE}" in check|train) ;; *) echo "Usage: $0 [check|train]" >&2; exit 1 ;; esac
PREVIEW=0
if [[ "${MODE}" == check || "${CHECK_ONLY:-0}" == 1 || "${DRY_RUN:-0}" == 1 ]]; then PREVIEW=1; fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "${GPUS}"
NPROC="${NPROC:-${#gpu_ids[@]}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29574}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SPATIAL_REPLAY_RATIO="${SPATIAL_REPLAY_RATIO:-3}"
for key in NPROC NNODES BATCH_SIZE GRAD_ACCUM_STEPS SPATIAL_REPLAY_RATIO; do
  [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || { echo "${key} must be a positive integer." >&2; exit 1; }
done
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || {
  echo "NODE_RANK must be between 0 and NNODES-1." >&2; exit 1;
}
if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == 127.0.0.1 || "${MASTER_ADDR}" == localhost ]]; then
  echo "Set MASTER_ADDR to the rank-0 host for multi-node training." >&2; exit 1
fi
GLOBAL_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC * NNODES))
# Reference recipe: global batch 32, learning rate 1e-5. Explicit LR takes precedence.
LR="${LR:-$(awk -v batch="${GLOBAL_BATCH}" 'BEGIN {printf "%.8g", 1e-5 * batch / 32}')}"
DATASET_ROOT="${SO_DATASET_ROOT:-}"
QA_ROOT="${QA_ROOT:-${DATASET_ROOT:+${DATASET_ROOT}/qa}}"
[[ -n "${QA_ROOT}" ]] || { echo "Set QA_ROOT or SO_DATASET_ROOT." >&2; exit 1; }
AUDIO_ROOT="${AUDIO_ROOT:-${DATASET_ROOT:-$(dirname "${QA_ROOT}")}}"
BEATS_CKPT="${BEATS_CKPT:-${SO_ENCODER_CKPT:-}}"
BEATS_REPO="${BEATS_REPO:-${SO_BEATS_REPO:-}}"
MODEL_ID="${MODEL_ID:-${SO_BASE_MODEL:-ModelQzx/ModelQzx2.5-Omni-7B}}"
RESUME_CKPT="${RESUME_CKPT:-}"
REPLAY_QA_ROOT="${REPLAY_QA_ROOT:-}"
REPLAY_TRAIN_SPLIT="${REPLAY_TRAIN_SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
[[ -n "${OUTPUT_DIR}" ]] || { echo "Set OUTPUT_DIR to a new run directory shared by all ranks." >&2; exit 1; }
for file in "${BEATS_CKPT}" "${RESUME_CKPT}" "${QA_ROOT}/train.jsonl" "${QA_ROOT}/valid.jsonl"; do
  [[ -f "${file}" ]] || { echo "Missing required input: ${file}. Check SO_ENCODER_CKPT, RESUME_CKPT and QA_ROOT." >&2; exit 1; }
done
[[ -n "${REPLAY_QA_ROOT}" && -f "${REPLAY_QA_ROOT}/${REPLAY_TRAIN_SPLIT}.jsonl" ]] || {
  echo "Set REPLAY_QA_ROOT to a directory containing ${REPLAY_TRAIN_SPLIT}.jsonl with local audio_path/question/answer records." >&2; exit 1;
}

train_args=(
  --model-id "${MODEL_ID}" --beats-checkpoint "${BEATS_CKPT}" --beats-repo "${BEATS_REPO}"
  --so-repo "${ROOT_DIR}" --beats-lora --resume-checkpoint-path "${RESUME_CKPT}"
  --qa-root "${QA_ROOT}" --audio-root "${AUDIO_ROOT}" --train-split train --valid-split valid
  --replay-qa-root "${REPLAY_QA_ROOT}" --replay-train-split "${REPLAY_TRAIN_SPLIT}"
  --mixed-spatial-replay --spatial-replay-ratio "${SPATIAL_REPLAY_RATIO}"
  --null-alignment-weight 0.05 --device cuda:0 --dtype bfloat16
  --attn-impl "${ATTN_IMPL:-sdpa}" --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" --num-workers "${NUM_WORKERS}"
  --persistent-workers --prefetch-factor 2 --lr "${LR}" --epochs "${EPOCHS:-1}"
  --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 --save-every-epoch
  --save-every-n-optimizer-steps "${SAVE_EVERY_N_OPT_STEPS:-5000}"
  --valid-every-n-optimizer-steps "${VALID_EVERY_N_OPT_STEPS:-5000}"
  --valid-generate-max-samples 32 --valid-max-new-tokens 96 --valid-num-beams 1
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
  --lora-target-modules q_proj k_proj v_proj o_proj --lora-target-prefixes thinker.model
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${RESUME_MODEL_ONLY:-1}" == 1 ]]; then train_args+=(--resume-model-only); fi
if [[ "${USE_GRADIENT_CHECKPOINTING:-0}" == 1 ]]; then train_args+=(--gradient-checkpointing); fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then train_args+=(--max-train-samples "${MAX_TRAIN_SAMPLES}"); fi
if [[ -n "${MAX_VALID_SAMPLES:-}" ]]; then train_args+=(--max-valid-samples "${MAX_VALID_SAMPLES}"); fi
launch_cmd=(env CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON}" -m torch.distributed.run --nnodes="${NNODES}" --node_rank="${NODE_RANK}"
  --nproc_per_node="${NPROC}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/train_so_qa.py" "${train_args[@]}")
printf '[config] world=%s global_batch=%s lr=%s output=%s\n' "$((NPROC * NNODES))" "${GLOBAL_BATCH}" "${LR}" "${OUTPUT_DIR}"
printf '[command]'; printf ' %q' "${launch_cmd[@]}"; printf '\n'
if (( PREVIEW )); then
  echo "Preview complete; no training, downloads or output files were created."
  exit 0
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(ls -A -- "${OUTPUT_DIR}")" && "${RESUME_EXISTING:-0}" != 1 ]]; then
  echo "Output is not empty: ${OUTPUT_DIR}. Choose a new OUTPUT_DIR or set RESUME_EXISTING=1 for an intentional resume." >&2
  exit 1
fi
exec "${launch_cmd[@]}"
