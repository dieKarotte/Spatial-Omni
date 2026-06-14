#!/usr/bin/env bash
# Unified test-split bench for all Spatial-Omni model variants.
#
# Purpose:
#   Run generation on the test splits for each requested model×stage,
#   producing predictions.jsonl that
#   scripts/score_test_predictions.py can evaluate.
#
# Default targets (MODELS env var): 4 IV baseline checkpoints —
#   - iv          stage1 (projector-only)
#   - iv          stage2 (encoder_lora, projector + LoRA)
#   - neural_iv   stage1
#   - neural_iv   stage2
# If you also want the BEATs main model (so_7b stage2/stage3),
# add "beats:stage2" / "beats:stage3" to MODELS.
# Spatial-AF3 is also supported via "af3:stage2" /
# "af3:stage3" — uses scripts/bench_test_generate_af3.py.
#
# Usage:
#   bash shell/launch_bench_test.sh                              # 4 IV × splits
#   MODELS="iv:stage2 neural_iv:stage2" bash shell/launch_bench_test.sh
#   SPLITS="test" MODELS="iv:stage1" bash shell/launch_bench_test.sh
#   GPUS=0,1,2,3 BATCH_SIZE=2 bash shell/launch_bench_test.sh
#
#   # BEATs main model (needs the SO-7B bench script, not IV):
#   MODELS="beats:stage2 beats:stage3" bash shell/launch_bench_test.sh
#
#   # Spatial-AF3 baseline:
#   MODELS="af3:stage3" bash shell/launch_bench_test.sh
#
# Output layout (mirrors the existing bench_test_generate.py default):
#   <run_dir>/bench/<split>/<ckpt_name>/predictions.jsonl
# e.g.:
#   runs/iv/stage2_encoder_lora/bench/test/best/predictions.jsonl
#
# After each run emits predictions.jsonl, the script optionally calls
# scripts/score_test_predictions.py to compute task-aware metrics + LLM judge.
# Set RUN_SCORING=0 to skip scoring (useful if you just want predictions now
# and will score later in one pass with a different LLM judge config).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${REPO_ROOT}"

# ------------------------------------------------------------------
# Data + splits
# ------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-${SO_DATASET_ROOT}}"
SPLITS="${SPLITS:-test}"
SPLIT_NAME="${SPLIT_NAME:-test}"

# ------------------------------------------------------------------
# Which model×stage combinations to bench
# Format: <encoder>:<stage>  where stage ∈ {stage1, stage2, stage3 (beats only)}
# ------------------------------------------------------------------
MODELS="${MODELS:-neural_iv:stage2 iv:stage2}"

# ------------------------------------------------------------------
# Run directories (<run_dir>/<stage_subdir>/checkpoints/best_trainable.pt)
# ------------------------------------------------------------------
IV_RUN_ROOT="${IV_RUN_ROOT:-${REPO_ROOT}/runs/iv}"
NEURAL_IV_RUN_ROOT="${NEURAL_IV_RUN_ROOT:-${REPO_ROOT}/runs/neural_iv}"
BEATS_RUN_ROOT="${BEATS_RUN_ROOT:-${REPO_ROOT}/runs/so_7b}"
AF3_RUN_ROOT="${AF3_RUN_ROOT:-${REPO_ROOT}/runs/af3}"

STAGE1_SUBDIR="${STAGE1_SUBDIR:-stage1_projector}"
STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_encoder_lora}"
STAGE3_SUBDIR="${STAGE3_SUBDIR:-stage3_beats_lora}"
CKPT_NAME="${CKPT_NAME:-best_trainable.pt}"

# ------------------------------------------------------------------
# Inference config
# ------------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${NPROC:-${#GPU_ARRAY[@]}}"
MASTER_PORT="${MASTER_PORT:-29551}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-auto}"  # flash_attention_2 if available, else sdpa
MONO_AUDIO_ZERO_SPATIAL_TOKENS="${MONO_AUDIO_ZERO_SPATIAL_TOKENS:-0}"
MONO_AUDIO_W_CHANNEL_SPATIAL_ENCODER="${MONO_AUDIO_W_CHANNEL_SPATIAL_ENCODER:-0}"
# DROP_MONO_AUDIO=1: decoder-only baseline. Spatial encoder runs as usual,
# Qwen mono <|AUDIO|> branch is disabled (no audio_token in prompt, no
# input_features). Output dir auto-suffixes with __drop_mono_audio.
# Mutually exclusive with the MONO_AUDIO_* compat modes above.
DROP_MONO_AUDIO="${DROP_MONO_AUDIO:-0}"
if [[ "${MONO_AUDIO_ZERO_SPATIAL_TOKENS}" == "1" && "${MONO_AUDIO_W_CHANNEL_SPATIAL_ENCODER}" == "1" ]]; then
  echo "[ERROR] MONO_AUDIO_ZERO_SPATIAL_TOKENS and MONO_AUDIO_W_CHANNEL_SPATIAL_ENCODER are mutually exclusive" >&2
  exit 1
fi
if [[ "${DROP_MONO_AUDIO}" == "1" && ( "${MONO_AUDIO_ZERO_SPATIAL_TOKENS}" == "1" || "${MONO_AUDIO_W_CHANNEL_SPATIAL_ENCODER}" == "1" ) ]]; then
  echo "[ERROR] DROP_MONO_AUDIO is mutually exclusive with MONO_AUDIO_* compat modes" >&2
  exit 1
fi

# QWEN_AUDIO_CACHE_MANIFEST= to disable (reads wav directly, slower but no cache bugs)
QWEN_AUDIO_CACHE_MANIFEST="${QWEN_AUDIO_CACHE_MANIFEST:-}"

# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------
RUN_SCORING="${RUN_SCORING:-0}"                 # 0 = skip scoring entirely
USE_LLM_JUDGE="${USE_LLM_JUDGE:-0}"             # 1 = --llm-judge
LLM_CONCURRENCY="${LLM_CONCURRENCY:-16}"
SKIP_EXISTING_BENCH="${SKIP_EXISTING_BENCH:-1}"    # skip if predictions.jsonl already exists

# ------------------------------------------------------------------
# Helper: pick (encoder, run_root, stage_subdir, bench_script) for a token
# ------------------------------------------------------------------
resolve_model_spec() {
  local spec="$1"
  local encoder stage
  encoder="${spec%:*}"
  stage="${spec#*:}"

  local run_root stage_subdir bench_script
  case "${encoder}" in
    iv)
      run_root="${IV_RUN_ROOT}"
      bench_script="scripts/bench_test_generate_iv.py"
      ;;
    neural_iv)
      run_root="${NEURAL_IV_RUN_ROOT}"
      bench_script="scripts/bench_test_generate_iv.py"
      ;;
    beats)
      run_root="${BEATS_RUN_ROOT}"
      bench_script="scripts/bench_test_generate.py"
      ;;
    af3|spatial_flamingo|spatial-flamingo)
      run_root="${AF3_RUN_ROOT}"
      bench_script="scripts/bench_test_generate_af3.py"
      ;;
    *)
      echo "[ERROR] unknown encoder in MODELS spec: '${encoder}' (expected iv/neural_iv/beats/af3)" >&2
      return 1
      ;;
  esac
  case "${stage}" in
    stage1) stage_subdir="${STAGE1_SUBDIR}" ;;
    stage2) stage_subdir="${STAGE2_SUBDIR}" ;;
    stage3)
      if [[ "${encoder}" != "beats" && "${encoder}" != "af3" \
            && "${encoder}" != "spatial_flamingo" && "${encoder}" != "spatial-flamingo" ]]; then
        echo "[ERROR] stage3 only exists for beats/af3 (got encoder=${encoder})" >&2
        return 1
      fi
      stage_subdir="${STAGE3_SUBDIR}"
      ;;
    *)
      echo "[ERROR] unknown stage in MODELS spec: '${stage}' (expected stage1/stage2/stage3)" >&2
      return 1
      ;;
  esac
  echo "${encoder}|${run_root}|${stage_subdir}|${bench_script}"
}

# ------------------------------------------------------------------
# Pretty print config
# ------------------------------------------------------------------
echo "==========================================================="
echo " Test-split bench (all models)"
echo "   DATA_ROOT  = ${DATA_ROOT}"
echo "   SPLITS     = ${SPLITS}"
echo "   MODELS     = ${MODELS}"
echo "   GPUS       = ${GPUS}  NPROC=${NPROC}"
echo "   BATCH_SIZE = ${BATCH_SIZE}  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "   ATTN_IMPL  = ${ATTN_IMPL}"
echo "   MONO_ZERO  = ${MONO_AUDIO_ZERO_SPATIAL_TOKENS}  MONO_W=${MONO_AUDIO_W_CHANNEL_SPATIAL_ENCODER}  DROP_MONO=${DROP_MONO_AUDIO}"
echo "   RUN_SCORING= ${RUN_SCORING}  LLM_JUDGE=${USE_LLM_JUDGE}"
echo "==========================================================="

# ------------------------------------------------------------------
# For each (model, split), run generation then (optionally) scoring
# ------------------------------------------------------------------
for spec in ${MODELS}; do
  parts="$(resolve_model_spec "${spec}")" || { echo "[abort] bad model spec"; exit 1; }
  IFS='|' read -r encoder run_root stage_subdir bench_script <<< "${parts}"

  ckpt_path="${run_root}/${stage_subdir}/checkpoints/${CKPT_NAME}"
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[skip] missing checkpoint: ${ckpt_path}"
    continue
  fi

  for split in ${SPLITS}; do
    qa_root="${DATA_ROOT}/${split}"
    if [[ ! -f "${qa_root}/${SPLIT_NAME}.jsonl" ]]; then
      echo "[skip] missing ${qa_root}/${SPLIT_NAME}.jsonl"
      continue
    fi

    output_dir="${run_root}/${stage_subdir}/bench/${split}"
    # The bench scripts auto-suffix the output dir with __drop_mono_audio
    # when --drop-mono-audio is set; mirror that here so SKIP_EXISTING_BENCH
    # checks the correct predictions.jsonl path.
    if [[ "${DROP_MONO_AUDIO}" == "1" ]]; then
      output_dir="${output_dir}__drop_mono_audio"
    fi
    ckpt_tag="${CKPT_NAME%_trainable.pt}"
    predictions_jsonl="${output_dir}/${ckpt_tag}/predictions.jsonl"

    echo ""
    echo "==========================================================="
    echo "[run] model=${spec}  split=${split}"
    echo "      ckpt      = ${ckpt_path}"
    echo "      script    = ${bench_script}"
    echo "      output_dir= ${output_dir}"
    echo "==========================================================="

    if [[ "${SKIP_EXISTING_BENCH}" == "1" && -s "${predictions_jsonl}" ]]; then
      echo "[skip-bench] predictions.jsonl already exists: ${predictions_jsonl}"
    else
      mkdir -p "${output_dir}"

      extra=()
      if [[ -n "${QWEN_AUDIO_CACHE_MANIFEST}" ]]; then
        extra+=(--audio-feature-cache-manifest "${QWEN_AUDIO_CACHE_MANIFEST}")
      fi
      # --attn-impl is only supported by IV/Neural-IV bench scripts.
      # BEATs (bench_test_generate.py) and AF3 (bench_test_generate_af3.py)
      # don't expose it and will error on unrecognized args.
      if [[ "${encoder}" == "iv" || "${encoder}" == "neural_iv" ]]; then
        extra+=(--attn-impl "${ATTN_IMPL}")
      fi
      if [[ "${encoder}" == "beats" || "${encoder}" == "af3" \
            || "${encoder}" == "spatial_flamingo" || "${encoder}" == "spatial-flamingo" ]]; then
        if [[ "${MONO_AUDIO_ZERO_SPATIAL_TOKENS}" == "1" ]]; then
          extra+=(--mono-audio-zero-spatial-tokens)
        fi
        if [[ "${MONO_AUDIO_W_CHANNEL_SPATIAL_ENCODER}" == "1" ]]; then
          extra+=(--mono-audio-w-channel-spatial-encoder)
        fi
      fi
      # --drop-mono-audio is supported by all four bench scripts
      # (iv / neural_iv / beats / af3) for the decoder-only baseline.
      if [[ "${DROP_MONO_AUDIO}" == "1" ]]; then
        extra+=(--drop-mono-audio)
      fi

      # Slightly different port per run so repeated torchrun calls don't
      # collide in the same shell session.
      PORT=$((MASTER_PORT + RANDOM % 100))

      CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
          --nnodes=1 \
          --nproc_per_node="${NPROC}" \
          --master_port="${PORT}" \
          "${REPO_ROOT}/${bench_script}" \
          --checkpoint-paths "${ckpt_path}" \
          --qa-root "${qa_root}" \
          --split "${SPLIT_NAME}" \
          --output-dir "${output_dir}" \
          --batch-size "${BATCH_SIZE}" \
          --num-workers "${NUM_WORKERS}" \
          --max-new-tokens "${MAX_NEW_TOKENS}" \
          --dtype "${DTYPE}" \
          "${extra[@]}" \
          "$@"
    fi

    if [[ ! -f "${predictions_jsonl}" ]]; then
      echo "[ERROR] predictions not produced: ${predictions_jsonl}" >&2
      continue
    fi

    # -------------------------------- Scoring --------------------------------
    if [[ "${RUN_SCORING}" == "1" ]]; then
      score_json="$(dirname "${predictions_jsonl}")/score_result.json"
      if [[ -s "${score_json}" && "${SKIP_EXISTING_BENCH}" == "1" ]]; then
        echo "[skip-score] already exists: ${score_json}"
      else
        echo "-----------------------------------------------------------"
        echo "[score] ${predictions_jsonl}"
        echo "-----------------------------------------------------------"
        score_args=(
          python "${REPO_ROOT}/scripts/score_test_predictions.py"
          --predictions-jsonl "${predictions_jsonl}"
          --qa-root "${qa_root}"
          --split "${SPLIT_NAME}"
          --output-json "${score_json}"
        )
        if [[ "${USE_LLM_JUDGE}" == "1" ]]; then
          score_args+=(--llm-judge --llm-concurrency "${LLM_CONCURRENCY}")
        fi
        "${score_args[@]}" || echo "[warn] scoring failed for ${predictions_jsonl}"
      fi
    fi
  done
done

echo ""
echo "==========================================================="
echo "All requested (model, split) combinations finished."
echo "Predictions live under each run's bench/<split>/<ckpt>/predictions.jsonl"
echo "==========================================================="
