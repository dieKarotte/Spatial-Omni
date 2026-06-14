#!/usr/bin/env bash
# 两阶段训练 IV / Neural-IV spatial baseline + Qwen2.5-Omni
#
# 数据：${SO_DATASET_ROOT}/qa/{train,valid,test}.jsonl
# Encoder 选择：SPATIAL_ENCODER_TYPE=iv  → 纯 IV + MLP (116K 可训练)
#               SPATIAL_ENCODER_TYPE=neural_iv → IV + CNN + MLP (102K 可训练)
#
# 使用示例：
#   # IV baseline 完整 2 阶段
#   SPATIAL_ENCODER_TYPE=iv bash shell/launch_train_spatial_iv_qa.sh
#   # Neural-IV baseline 完整 2 阶段
#   SPATIAL_ENCODER_TYPE=neural_iv bash shell/launch_train_spatial_iv_qa.sh
#   # 从 stage2 开始（需 stage1 best checkpoint 已存在）
#   SPATIAL_ENCODER_TYPE=iv START_STAGE=2 bash shell/launch_train_spatial_iv_qa.sh
#
# 多机多卡（参考 SO-7B 脚本，相同语义）：
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29575 \
#       SPATIAL_ENCODER_TYPE=iv bash shell/launch_train_spatial_iv_qa.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# ------------------------------------------------------------------
# Encoder 类型
# ------------------------------------------------------------------
SPATIAL_ENCODER_TYPE="${SPATIAL_ENCODER_TYPE:-iv}"
if [[ "${SPATIAL_ENCODER_TYPE}" != "iv" && "${SPATIAL_ENCODER_TYPE}" != "neural_iv" ]]; then
  echo "[ERROR] SPATIAL_ENCODER_TYPE must be 'iv' or 'neural_iv', got '${SPATIAL_ENCODER_TYPE}'" >&2
  exit 1
fi

# ------------------------------------------------------------------
# 分布式 / GPU
# ------------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(python -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x]))' "${GPUS}")}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29575}"
START_STAGE="${START_STAGE:-1}"

if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "[ERROR] NNODES=${NNODES} > 1 but MASTER_ADDR is loopback (${MASTER_ADDR}). " >&2
  echo "        Set MASTER_ADDR to the actual IP of rank-0 machine (reachable from all nodes)." >&2
  exit 1
fi

# ------------------------------------------------------------------
# 数据 / 外部依赖路径
# ------------------------------------------------------------------
QA_ROOT="${QA_ROOT:-${SO_DATASET_ROOT}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Omni-7B}"
BASELINE_REPO_PATH="${BASELINE_REPO_PATH:-${DCASE_BASELINE_REPO}}"
SELD_FEATURE_STATS_DIR="${SELD_FEATURE_STATS_DIR:-${SELD_FEATURE_STATS_DIR}}"

# ------------------------------------------------------------------
# 输出目录
# ------------------------------------------------------------------
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/iv_${SPATIAL_ENCODER_TYPE}}"
STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1_projector}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2_encoder_lora}"
STAGE2_RESUME_CKPT="${STAGE2_RESUME_CKPT:-${STAGE1_DIR}/checkpoints/best_trainable.pt}"

# ------------------------------------------------------------------
# batch / DataLoader / checkpoint 频率
# ------------------------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAVE_EVERY_N_OPT_STEPS="${SAVE_EVERY_N_OPT_STEPS:-2000}"
VALID_EVERY_N_OPT_STEPS="${VALID_EVERY_N_OPT_STEPS:-2000}"

# 性能相关（与 launch_train_so_7b.sh 对齐）：
# - ATTN_IMPL：注意力实现
#     * "auto"（默认）：有 flash-attn 则走 flash_attention_2，否则退化到 sdpa。
#     * "flash_attention_2"：需 `pip install flash-attn`（本仓库环境已装 2.8.3）
#     * "sdpa"：⚠️ Qwen2.5-Omni 的 sdpa 路径在 bf16 + gradient_checkpointing(use_reentrant=False)
#       + 含 padding 的 causal mask 下，反向会产生 NaN grad（实测 neural_iv stage1
#       skip_g=100% / 1370/1370 opt-step，训练不推进）。勿在 IV 路径上用，除非同时关闭 GC。
#     * "eager"：最慢，仅在 flash-attn / sdpa 都不行时回退
# - USE_GRADIENT_CHECKPOINTING=0/1：40GB A100 + bs=4 + LoRA 通常不需要 GC（关掉 -40% 时间）
# - QWEN_AUDIO_CACHE_MANIFEST：离线预提 Qwen mel 特征的 manifest.json 路径（强烈推荐）
# - IV_MODULES_FP32=0/1：把 IV/Neural-IV 的 adapter（conv_encoder / token_norm /
#   token_head）和 projector 保留在 fp32。feature_bridge（STFT + log-mel + 归一化）
#   始终在 fp32 + no_grad 下执行（它没有可训练参数），这里的开关只控制后面的小 MLP。
#   ⚠️ 经验：开启后在 flash-attn + DDP + clip_grad_norm_ 下反而会造成 100% NaN-grad
#   （混合 dtype 与 clip_grad_norm_ 的交互），**不建议默认开启**。仅当 bf16 下明确
#   定位到 adapter 内部出现 NaN 时再考虑打开。
IV_MODULES_FP32="${IV_MODULES_FP32:-0}"

# ATTN_IMPL：sdpa（推荐，与 SO-7B/30B 一致，与 flagship train_args.json 对齐）
# auto 会自动探到 flash-attn 2，会触发 padding_side='right' + fa2 的 strict check
# 在新版 transformers 下 (>= 4.42)，导致训练 forward 直接 raise。
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-1}"
QWEN_AUDIO_CACHE_MANIFEST="${QWEN_AUDIO_CACHE_MANIFEST:-}"

# 预期规模：全局 bs = 4 × 2 × 8 = 64，train=787K → 每 epoch ≈ 12300 opt step
# stage1 × 3 epoch ≈ 36900 step；stage2 × 3 epoch ≈ 36900 step

# ------------------------------------------------------------------
# 训练 schedule
# ------------------------------------------------------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"

# stage1：IV adapter + projector 全新随机初始化，使用较保守的 lr。
# 最初版本用 1e-4 会在 Neural-IV 路径下偶发 NaN（bf16 + energy 归一化 + large CNN init
# 共振），实测 5e-5 稳定。
STAGE1_LR="${STAGE1_LR:-5e-5}"
STAGE1_PROJECTOR_LR="${STAGE1_PROJECTOR_LR:-5e-5}"

# stage2：projector 已预训练，LoRA 随机初始化；参考 BEATs stage2 的配比
STAGE2_LR="${STAGE2_LR:-3e-5}"
STAGE2_LORA_LR="${STAGE2_LORA_LR:-3e-5}"
STAGE2_PROJECTOR_LR="${STAGE2_PROJECTOR_LR:-1e-5}"

# grad clip：默认 0.5，比 BEATs 路径（1.0）更保守，防 IV 路径首轮梯度爆炸
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"

# ------------------------------------------------------------------
# IV 超参（DCASE 默认值）
# ------------------------------------------------------------------
IV_TOKEN_DIM="${IV_TOKEN_DIM:-256}"
IV_PROJECTOR_HIDDEN_DIM="${IV_PROJECTOR_HIDDEN_DIM:-512}"
IV_NUM_MEL_BINS="${IV_NUM_MEL_BINS:-64}"
IV_BAND_POOL="${IV_BAND_POOL:-0}"
IV_OUTPUT_SCALE="${IV_OUTPUT_SCALE:-0.02}"
IV_FEATURE_TO_SELD_RATIO="${IV_FEATURE_TO_SELD_RATIO:-5}"
IV_DOWNSAMPLE_FACTOR="${IV_DOWNSAMPLE_FACTOR:-4}"
NEURAL_IV_HIDDEN_CHANNELS="${NEURAL_IV_HIDDEN_CHANNELS:-64}"

# ------------------------------------------------------------------
# LoRA
# ------------------------------------------------------------------
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES=(${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj})

# ------------------------------------------------------------------
# 前置检查
# ------------------------------------------------------------------
if [[ ! -d "${QA_ROOT}" ]]; then
  echo "Missing QA root: ${QA_ROOT}" >&2
  exit 1
fi
for split in train valid test; do
  if [[ ! -f "${QA_ROOT}/${split}.jsonl" ]]; then
    echo "Missing ${QA_ROOT}/${split}.jsonl" >&2
    exit 1
  fi
done
if [[ ! -d "${BASELINE_REPO_PATH}" ]]; then
  echo "Missing DCASE baseline repo: ${BASELINE_REPO_PATH}" >&2
  exit 1
fi
if [[ ! -d "${SELD_FEATURE_STATS_DIR}" ]]; then
  echo "Missing SELD feature stats dir: ${SELD_FEATURE_STATS_DIR}" >&2
  exit 1
fi

echo "==========================================================="
echo " IV baseline training:"
echo "   SPATIAL_ENCODER_TYPE = ${SPATIAL_ENCODER_TYPE}"
echo "   NNODES=${NNODES}  NODE_RANK=${NODE_RANK}"
echo "   MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "   NPROC (GPUs per node) = ${NPROC}  GPUS=${GPUS}"
echo "   Global world size     = $((NNODES * NPROC))"
echo "   START_STAGE=${START_STAGE}"
echo "   RUN_ROOT=${RUN_ROOT}"
echo "==========================================================="

# ------------------------------------------------------------------
# 通用 torchrun 包装
# ------------------------------------------------------------------
run_train() {
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
      --nnodes="${NNODES}" \
      --node_rank="${NODE_RANK}" \
      --nproc_per_node="${NPROC}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      "${ROOT_DIR}/train_spatial_iv_qa.py" "$@"
}

# 所有 stage 共用的 flag
common_args=(
  --model-id "${MODEL_ID}"
  --spatial-encoder-type "${SPATIAL_ENCODER_TYPE}"
  --baseline-repo-path "${BASELINE_REPO_PATH}"
  --seld-feature-stats-dir "${SELD_FEATURE_STATS_DIR}"
  --iv-token-dim "${IV_TOKEN_DIM}"
  --iv-projector-hidden-dim "${IV_PROJECTOR_HIDDEN_DIM}"
  --iv-num-mel-bins "${IV_NUM_MEL_BINS}"
  --iv-band-pool "${IV_BAND_POOL}"
  --iv-output-scale "${IV_OUTPUT_SCALE}"
  --iv-feature-to-seld-ratio "${IV_FEATURE_TO_SELD_RATIO}"
  --iv-downsample-factor "${IV_DOWNSAMPLE_FACTOR}"
  --neural-iv-hidden-channels "${NEURAL_IV_HIDDEN_CHANNELS}"
  --qa-root "${QA_ROOT}"
  --train-split train
  --valid-split valid
  --device cuda:0
  --dtype bfloat16
  --attn-impl "${ATTN_IMPL}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --persistent-workers
  --prefetch-factor "${PREFETCH_FACTOR}"
  --warmup-ratio 0.03
  --weight-decay 0.01
  --max-grad-norm "${MAX_GRAD_NORM}"
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
  --lora-target-prefixes thinker.model
)
if (( USE_GRADIENT_CHECKPOINTING == 1 )); then
  common_args+=(--gradient-checkpointing)
  echo "[config] gradient_checkpointing = ENABLED（减速但省显存，40GB A100 + bs=4 通常可关）"
else
  echo "[config] gradient_checkpointing = DISABLED（40GB A100 + LoRA 推荐关掉加速）"
fi
if [[ -n "${QWEN_AUDIO_CACHE_MANIFEST}" ]]; then
  common_args+=(--audio-feature-cache-manifest "${QWEN_AUDIO_CACHE_MANIFEST}")
  echo "[config] audio feature cache = ${QWEN_AUDIO_CACHE_MANIFEST}"
else
  echo "[config] audio feature cache = OFF（每个 batch 要 ~400ms 做 mel，强烈建议预计算 cache）"
fi
if [[ "${VALID_GENERATE_FULL:-0}" == "1" ]]; then
  common_args+=(--valid-generate-full)
  echo "[config] valid_generate_full = ON (整个 valid 集都生成；每 epoch 耗时增加，但保存全量 predictions)"
else
  echo "[config] valid_generate_full = OFF (仅生成 ${VALID_GENERATE_MAX_SAMPLES:-32} 条；设 VALID_GENERATE_FULL=1 保存全集)"
fi
if (( IV_MODULES_FP32 == 1 )); then
  common_args+=(--iv-modules-fp32)
  echo "[config] iv_modules_fp32 = ON (adapter + projector pinned to fp32)"
else
  echo "[config] iv_modules_fp32 = OFF (bf16; set IV_MODULES_FP32=1 if neural_iv stage1 sees NaN grads)"
fi
echo "[config] attn_impl = ${ATTN_IMPL}"

# ------------------------------------------------------------------
# Stage 1: projector_only
# ------------------------------------------------------------------
if (( START_STAGE <= 1 )); then
  echo "==========================================================="
  echo "[stage1] projector_only (${STAGE1_EPOCHS} epochs, lr=${STAGE1_LR})"
  echo "  → ${STAGE1_DIR}"
  echo "==========================================================="
  stage1_extra=()
  # Optional resume for stage1 (used by autorestart wrapper after a crash).
  # STAGE1_RESUME_CKPT=/path/to/step_XXXXX_trainable.pt preserves progress.
  # Default (full resume): optimizer + scheduler + step counter restored so
  # the LR schedule continues. Set STAGE1_RESUME_MODEL_ONLY=1 to only
  # reload trainable weights and restart epoch 1 with fresh optimizer.
  if [[ -n "${STAGE1_RESUME_CKPT:-}" ]]; then
    echo "  resume from: ${STAGE1_RESUME_CKPT}"
    stage1_extra+=(--resume-checkpoint-path "${STAGE1_RESUME_CKPT}")
    if [[ "${STAGE1_RESUME_MODEL_ONLY:-0}" == "1" ]]; then
      stage1_extra+=(--resume-model-only)
      echo "  resume mode: MODEL ONLY (fresh optimizer, restart from epoch 1)"
    else
      echo "  resume mode: FULL (optimizer + scheduler + step counter restored)"
    fi
  fi
  run_train \
    "${common_args[@]}" \
    --projector-only \
    --lr "${STAGE1_LR}" \
    --projector-lr "${STAGE1_PROJECTOR_LR}" \
    --epochs "${STAGE1_EPOCHS}" \
    --output-dir "${STAGE1_DIR}" \
    "${stage1_extra[@]}"
fi

# ------------------------------------------------------------------
# Stage 2: encoder_lora (IV adapter + projector + LLM LoRA)
# ------------------------------------------------------------------
if (( START_STAGE <= 2 )); then
  if [[ ! -f "${STAGE2_RESUME_CKPT}" ]]; then
    echo "Missing stage2 resume checkpoint: ${STAGE2_RESUME_CKPT}" >&2
    echo "Set START_STAGE=1 to produce it, or STAGE2_RESUME_CKPT=/path/to/best_trainable.pt." >&2
    exit 1
  fi
  echo "==========================================================="
  echo "[stage2] encoder_lora (${STAGE2_EPOCHS} epochs, lora_lr=${STAGE2_LORA_LR}, proj_lr=${STAGE2_PROJECTOR_LR})"
  echo "  resume from: ${STAGE2_RESUME_CKPT}"
  echo "  → ${STAGE2_DIR}"
  echo "==========================================================="
  stage2_extra=()
  # STAGE2_RESUME_MODEL_ONLY controls whether we re-init the optimizer.
  # Default = 1 (legacy behaviour: brand-new stage starting from stage1 best
  # ckpt, so fresh optimizer / LR schedule is correct).
  # Autorestart sets STAGE2_RESUME_MODEL_ONLY=0 to keep optimizer state and
  # continue the LR schedule where it left off.
  if [[ "${STAGE2_RESUME_MODEL_ONLY:-1}" == "1" ]]; then
    stage2_extra+=(--resume-model-only)
    echo "  resume mode: MODEL ONLY (fresh optimizer, restart from epoch 1)"
  else
    echo "  resume mode: FULL (optimizer + scheduler + step counter restored)"
  fi
  run_train \
    "${common_args[@]}" \
    --encoder-lora \
    --resume-checkpoint-path "${STAGE2_RESUME_CKPT}" \
    --lr "${STAGE2_LR}" \
    --lora-lr "${STAGE2_LORA_LR}" \
    --projector-lr "${STAGE2_PROJECTOR_LR}" \
    --epochs "${STAGE2_EPOCHS}" \
    --output-dir "${STAGE2_DIR}" \
    "${stage2_extra[@]}"
fi

echo "All requested stages finished. Run dir = ${RUN_ROOT}"
