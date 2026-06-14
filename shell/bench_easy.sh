#!/usr/bin/env bash
# Run all easy-split benches for paper comparison.
#
# Coverage:
#   so-7b          × stage1 / stage2 / stage3  (so_7b)
#   iv             × stage1 / stage2           (iv)            -- IV training has no stage3
#   neural-iv      × stage1 / stage2           (neural_iv)     -- ditto
#   zero-spatial   × stage3 best only          (so_7b stage3 ckpt, spatial audio zeroed)
#
# Each bench writes predictions.jsonl + bench_summary.json under:
#   <run-dir>/<stage>/bench/<split>/best/
# except zero-spatial which auto-appends __ablation_zero to output dir.
#
# Usage:
#   bash shell/bench_easy_all.sh                         # run everything
#   SKIP_EXISTING=1 bash shell/bench_easy_all.sh         # skip already-done benches
#   BASELINES=spatial-omni,iv bash shell/bench_easy_all.sh   # only these
#   STAGES=stage2_encoder_lora bash shell/bench_easy_all.sh  # only this stage
#
# Tunables:
#   GPUS, NPROC, BATCH_SIZE, NUM_BEAMS, MAX_NEW_TOKENS, MAX_SAMPLES
#
# After this runs, score per-task with:
#   python scripts/score_test_predictions.py \
#       --predictions-jsonl <out_dir>/best/predictions.jsonl \
#       --azimuth-threshold-deg 20 --elevation-threshold-deg 10

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${ROOT_DIR}"

# -------------------------- user-tunable --------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(python -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x]))' "${GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29597}"

QA_ROOT="${QA_ROOT:-${SO_DATASET_ROOT}}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
DTYPE="${DTYPE:-bfloat16}"
# IV / Neural-IV bench defaults to attn-impl=auto -> flash_attention_2 if installed,
# which can OOM on 8-GPU DDP (flash-attn pre-allocates large workspace per rank).
# Force sdpa here; matches what the BEATs path uses by default. Override with
# ATTN_IMPL=flash_attention_2 if you really want it.
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MAX_SAMPLES="${MAX_SAMPLES:-}"       # empty = all; set to small int for smoke
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Filter which baselines / stages to run (comma-separated)
BASELINES="${BASELINES:-so-7b}"
STAGES="${STAGES:-stage3_beats_lora}"

# ----------------------- ckpt layout ------------------------------
RUNS_ROOT="${ROOT_DIR}/runs"
SO_7B_RUN="${RUNS_ROOT}/so_7b"
IV_RUN="${RUNS_ROOT}/iv"
NEURAL_IV_RUN="${RUNS_ROOT}/neural_iv"

# Baseline -> available stages (stage3 not supported by IV training shell)
declare -A BASELINE_STAGES=(
  [so-7b]="stage1_projector stage2_encoder_lora stage3_beats_lora"
  [iv]="stage1_projector stage2_encoder_lora"
  [neural-iv]="stage1_projector stage2_encoder_lora"
  [zero-spatial]="stage3_beats_lora"
)
declare -A BASELINE_RUNDIR=(
  [so-7b]="${SO_7B_RUN}"
  [iv]="${IV_RUN}"
  [neural-iv]="${NEURAL_IV_RUN}"
  [zero-spatial]="${SO_7B_RUN}"   # same ckpt, ablation is injected by --baseline
)

# Comma-split helpers
IFS=',' read -ra WANT_BASELINES <<< "${BASELINES}"
IFS=',' read -ra WANT_STAGES <<< "${STAGES}"

in_array() {
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

# ----------------------- precheck paths --------------------------
if [[ ! -d "${QA_ROOT}" ]]; then
  echo "[ERROR] QA root not found: ${QA_ROOT}" >&2; exit 1
fi
if [[ ! -f "${QA_ROOT}/${SPLIT}.jsonl" ]]; then
  echo "[ERROR] ${QA_ROOT}/${SPLIT}.jsonl not found" >&2; exit 1
fi

echo "=========================================================="
echo " bench_easy_all"
echo "   qa_root     = ${QA_ROOT}"
echo "   split       = ${SPLIT}"
echo "   baselines   = ${BASELINES}"
echo "   stages      = ${STAGES}"
echo "   num_beams   = ${NUM_BEAMS}   batch_size = ${BATCH_SIZE}"
echo "   max_samples = ${MAX_SAMPLES:-ALL}"
echo "   skip_exist  = ${SKIP_EXISTING}"
echo "   gpus=${GPUS}  nproc=${NPROC}"
echo "=========================================================="

# ----------------------- one-bench helper ------------------------
run_bench_one() {
  local baseline="$1"
  local stage="$2"
  local run_dir="$3"
  local ckpt_path="${run_dir}/${stage}/checkpoints/best_trainable.pt"

  # zero-spatial uses the SAME ckpt as so-7b stage3; the bench script
  # auto-appends `__ablation_<mode>` to the output dir when --spatial-ablation
  # is set (via run_bench.py's baseline=zero-spatial injection). So just pass
  # the plain `bench/<split>` path -- do NOT manually add __ablation_zero here
  # or the suffix would be appended twice.
  #
  # To keep beam=1 and beam=4 results side-by-side (neither overwriting the
  # other), append `__beam<N>` to the split dir for N>1. Default beam=1 stays
  # in the plain `bench/<split>/` path so earlier beam=1 runs are not renamed.
  local split_dir="${SPLIT}"
  if [[ "${NUM_BEAMS}" != "1" ]]; then
    split_dir="${SPLIT}__beam${NUM_BEAMS}"
  fi
  local out_dir="${run_dir}/${stage}/bench/${split_dir}"

  local tag="[${baseline} / ${stage}]"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "${tag} SKIP (missing ckpt: ${ckpt_path})"
    return 0
  fi

  # best_trainable.pt -> predictions land under <out_dir>/best/ (sub-script uses ckpt stem).
  # For zero-spatial the sub-script appends "__ablation_zero" to out_dir before writing,
  # so the expected path has that suffix too.
  local expected_dir="${out_dir}"
  if [[ "${baseline}" == "zero-spatial" ]]; then
    expected_dir="${out_dir}__ablation_zero"
  fi
  local expected="${expected_dir}/best/predictions.jsonl"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${expected}" ]]; then
    echo "${tag} SKIP (predictions exists: ${expected})"
    return 0
  fi

  echo ""
  echo "----------------------------------------------------------"
  echo "${tag}"
  echo "  ckpt  = ${ckpt_path}"
  echo "  out   = ${out_dir}"
  echo "----------------------------------------------------------"

  local -a cmd=(
    torchrun
      --nproc_per_node="${NPROC}"
      --master_port="${MASTER_PORT}"
      scripts/run_bench.py
      --baseline "${baseline}"
      --qa-root "${QA_ROOT}"
      --split "${SPLIT}"
      --checkpoint-paths "${ckpt_path}"
      --output-dir "${out_dir}"
      --batch-size "${BATCH_SIZE}"
      --num-workers "${NUM_WORKERS}"
      --num-beams "${NUM_BEAMS}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --dtype "${DTYPE}"
  )
  # --attn-impl is only accepted by the IV sub-script; run_bench.py forwards
  # it to bench_test_generate_iv and silently drops it for so-7b / AF3.
  if [[ -n "${ATTN_IMPL}" ]]; then
    cmd+=(--attn-impl "${ATTN_IMPL}")
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  # Let every torchrun worker see ALL GPUs so LOCAL_RANK->cuda:LOCAL_RANK works.
  # We rely on setup_distributed() to torch.cuda.set_device(LOCAL_RANK) BEFORE
  # any cuda allocation. For the bench path specifically, we ALSO pass
  # --device cuda:LOCAL_RANK explicitly via a small wrapper env so that any
  # code path that reads args.device before setup_distributed runs stays on
  # the right card.
  CUDA_VISIBLE_DEVICES="${GPUS}" \
    TOKENIZERS_PARALLELISM=false \
    "${cmd[@]}"
}

# ----------------------- main loop -------------------------------
total=0
done_count=0
for baseline in "${!BASELINE_STAGES[@]}"; do
  in_array "${baseline}" "${WANT_BASELINES[@]}" || continue
  run_dir="${BASELINE_RUNDIR[${baseline}]}"
  for stage in ${BASELINE_STAGES[${baseline}]}; do
    in_array "${stage}" "${WANT_STAGES[@]}" || continue
    total=$((total + 1))
    if run_bench_one "${baseline}" "${stage}" "${run_dir}"; then
      done_count=$((done_count + 1))
    else
      echo "[bench_easy_all] WARNING: ${baseline}/${stage} returned non-zero" >&2
    fi
  done
done

echo ""
echo "=========================================================="
echo " bench_easy_all done: ${done_count} / ${total} succeeded"
echo ""
echo " score each with:"
echo "   python scripts/score_test_predictions.py \\"
echo "       --predictions-jsonl <out>/best/predictions.jsonl \\"
echo "       --azimuth-threshold-deg 20 --elevation-threshold-deg 10"
echo "=========================================================="
