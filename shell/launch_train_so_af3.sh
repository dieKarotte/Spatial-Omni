#!/usr/bin/env bash
# Three-stage training. MODE=check prints commands; smoke/full run them.
set -euo pipefail
MODE="${1:-check}"
case "$MODE" in check|smoke|full) ;; *) echo "Usage: $0 check|smoke|full" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${SO_BASE_MODEL:?Set SO_BASE_MODEL to the local base-model directory}"
: "${SO_ENCODER_CKPT:?Set SO_ENCODER_CKPT to the SO-Encoder checkpoint}"
: "${SO_DATASET_ROOT:?Set SO_DATASET_ROOT to the extracted training dataset}"
for input in "$SO_BASE_MODEL" "$SO_ENCODER_CKPT" "$SO_DATASET_ROOT/qa/train.jsonl" "$SO_DATASET_ROOT/qa/valid.jsonl"; do
  [[ -e "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
done
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/so-af3/$(date -u +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-8}"
WORKERS="${NUM_WORKERS:-20}"
EXTRA=()
if [[ "$MODE" == smoke ]]; then
  GPUS=2
  WORKERS=2
  EXTRA+=(--max-train-samples 256 --max-valid-samples 16)
fi
STAGES=(projector_only encoder_lora beats_lora)
BATCHES=(4 4 3)
EPOCHS=(2 3 3)
PROJECTOR_LRS=(1e-4 3e-5 1e-6)
LORA_LRS=(5e-5 5e-5 3e-5)
ACCUM="${GRAD_ACCUM_STEPS:-2}"
PREVIOUS=""
for index in 0 1 2; do
  stage="${STAGES[$index]}"
  batch="${BATCHES[$index]}"
  epochs="${EPOCHS[$index]}"
  reference_batch=$((batch * 2 * 8))
  if [[ "$MODE" == smoke ]]; then batch=2; epochs=2; fi
  effective_batch=$((batch * ACCUM * GPUS))
  # Preserve the reference learning rates at the original eight-GPU batch.
  scale="$(awk -v effective="$effective_batch" -v reference="$reference_batch" 'BEGIN {print effective/reference}')"
  proj_lr="$(awk -v lr="${PROJECTOR_LRS[$index]}" -v scale="$scale" 'BEGIN {printf "%.9g", lr*scale}')"
  lora_lr="$(awk -v lr="${LORA_LRS[$index]}" -v scale="$scale" 'BEGIN {printf "%.9g", lr*scale}')"
  beats_lr="$(awk -v scale="$scale" 'BEGIN {printf "%.9g", 1e-6*scale}')"
  output="$OUTPUT_ROOT/stage$((index+1))_$stage"
  cmd=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS"
       "train_so_af3.py" --model-dir "$SO_BASE_MODEL" --beats-checkpoint "$SO_ENCODER_CKPT"
       --qa-root "$SO_DATASET_ROOT/qa" --audio-root "$SO_DATASET_ROOT"
       --train-mode "$stage" --output-dir "$output" --batch-size "$batch"
       --epochs "$epochs" --grad-accum-steps "$ACCUM" --num-workers "$WORKERS"
       --lr "$proj_lr" --lora-lr "$lora_lr" --beats-lr "$beats_lr")
  [[ -n "$PREVIOUS" ]] && cmd+=(--resume-checkpoint-path "$PREVIOUS")
  cmd+=("${EXTRA[@]}")
  printf "%q " "${cmd[@]}"; printf "\n"
  if [[ "$MODE" != check ]]; then
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"
  fi
  PREVIOUS="$output/checkpoints/last_trainable.pt"
done
