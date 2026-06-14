#!/usr/bin/env bash
# 三阶段训练 SO-7B（SO-Encoder + Qwen2.5-Omni-7B）
#
# 数据：${SO_DATASET_ROOT}/qa/{train,valid,test}.jsonl
# Encoder ckpt：通过 --beats-checkpoint 或 SO_ENCODER_CKPT 环境变量指定
# stage1 projector_only (2 epoch) → stage2 encoder_lora (3 epoch) → stage3 beats_lora (3 epoch)
#
# 使用示例：
# ─── 单机多卡 ───
#   bash shell/launch_train_so_7b.sh              # 单机 8 卡三阶段
#   START_STAGE=2 bash shell/launch_train_so_7b.sh   # 从 stage2 开始
#   GPUS=0,1,2,3 bash shell/launch_train_so_7b.sh    # 单机 4 卡
#
# ─── 多机多卡（ALL 机器必须共享同一 NFS 路径，且都能 ping 通 rank0）───
# 假设 2 机 16 卡，rank0 机 IP = 10.0.0.1，rank1 机 IP = 10.0.0.2。两台机器上分别执行：
#   # 机器 0（rank 0）
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29573 \
#       bash shell/launch_train_so_7b.sh
#   # 机器 1（rank 1）
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29573 \
#       bash shell/launch_train_so_7b.sh
#
# 4 机 32 卡：NNODES=4，NODE_RANK=0/1/2/3，MASTER_ADDR 均填 rank0 机 IP。
# 两台机器必须同时启动（rank≥1 会阻塞等待 rank0），master 等待超时默认 30min。
#
# 显存：bs=4, grad_accum=2, dtype=bf16, 单卡 Qwen2.5-Omni-7B + BEATs(162M) + LoRA，
# 约 36GB，40GB A100 可以跑；40GB 不够就降到 bs=2 并把 accum 提到 4。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# ------------------------------------------------------------------
# 分布式 / GPU
# ------------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(python -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x]))' "${GPUS}")}"
# 多机设置：NNODES = 机器总数；NODE_RANK = 当前机器的 rank（0-indexed）；
# MASTER_ADDR = rank0 机的 IP（集群内可 ping 通）；MASTER_PORT = rank0 机上未被占用的端口。
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29573}"
START_STAGE="${START_STAGE:-1}"

# 单机时 MASTER_ADDR 必须是 127.0.0.1 或 localhost；多机时必须是 rank0 机的真实 IP
if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "[ERROR] NNODES=${NNODES} > 1 but MASTER_ADDR is loopback (${MASTER_ADDR}). " >&2
  echo "        Set MASTER_ADDR to the actual IP of rank-0 machine (reachable from all nodes)." >&2
  exit 1
fi

# ------------------------------------------------------------------
# 数据 / checkpoint 路径
# ------------------------------------------------------------------
# QA_ROOT defaults to ``${SO_DATASET_ROOT}/qa`` (SO-Dataset HF release layout).
# AUDIO_ROOT lets the QA loader resolve relative ``audio_path`` entries
# (release puts qa/ and audio/ as siblings under the dataset root).
QA_ROOT="${QA_ROOT:-${SO_DATASET_ROOT}/qa}"
AUDIO_ROOT="${AUDIO_ROOT:-${SO_DATASET_ROOT}}"
# BEATS_CKPT priority: explicit env var, then SO_ENCODER_CKPT (documented in
# README §1), then a placeholder path the user must override.
BEATS_CKPT="${BEATS_CKPT:-${SO_ENCODER_CKPT:-/path/to/so_encoder_pretrained.pt}}"
BEATS_REPO="${BEATS_REPO:-${SO_BEATS_REPO}}"
MODEL_ID="${MODEL_ID:-${SO_BASE_MODEL:-Qwen/Qwen2.5-Omni-7B}}"

# ------------------------------------------------------------------
# 输出目录
# ------------------------------------------------------------------
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/so_7b}"
STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1_projector}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2_encoder_lora}"
STAGE3_DIR="${STAGE3_DIR:-${RUN_ROOT}/stage3_beats_lora}"

STAGE2_RESUME_CKPT="${STAGE2_RESUME_CKPT:-${STAGE1_DIR}/checkpoints/best_trainable.pt}"
STAGE3_RESUME_CKPT="${STAGE3_RESUME_CKPT:-${STAGE2_DIR}/checkpoints/best_trainable.pt}"

# ------------------------------------------------------------------
# batch / DataLoader / checkpoint 频率
# ------------------------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SAVE_EVERY_N_OPT_STEPS="${SAVE_EVERY_N_OPT_STEPS:-1000}"
VALID_EVERY_N_OPT_STEPS="${VALID_EVERY_N_OPT_STEPS:-1000}"

# 性能相关：
# - ATTN_IMPL：注意力实现
#     * "sdpa"（推荐 legacy 环境默认）：torch 2.10 原生，零依赖，~85% flash-attn 速度
#     * "flash_attention_2"：需 `pip install flash-attn`；极致性能但要编译 30min+
#     * "eager"：最慢，仅在前两者都不行时回退
# - USE_GRADIENT_CHECKPOINTING=0/1：40GB A100 + bs=4 + LoRA 通常不需要 GC（关掉 -40% 时间）
# - QWEN_AUDIO_CACHE_MANIFEST：离线预提 Qwen mel 特征的 manifest.json 路径。强烈建议！
#   不开启时每个 batch 要花 ~400ms 在 CPU 上做 mel 提取，DataLoader 成为瓶颈。
#   生成命令（约 1~2h，只需跑一次）：
#     python scripts/precompute_qwen_audio_cache.py \
#         --qa-root $QA_ROOT --splits train valid test --batch-size 64 \
#         --cache-dir /path/to/ssd/qwen_audio_cache_easy_v2
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-0}"
QWEN_AUDIO_CACHE_MANIFEST="${QWEN_AUDIO_CACHE_MANIFEST:-}"

# stage1: global_bs = 4 * 2 * 8 = 64, 78.7w / 64 ≈ 12300 step/epoch × 2 epoch ≈ 24600 step
# stage2: ≈ 36900 step   stage3: ≈ 36900 step

# ------------------------------------------------------------------
# 训练 schedule：epoch 数 + 学习率
# ------------------------------------------------------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-3}"

# stage1：全新 projector，投影到 Qwen embed 4096 维，lr 可大。78.7w 大数据，略降到 5e-5 更稳。
STAGE1_LR="${STAGE1_LR:-5e-5}"
STAGE1_PROJECTOR_LR="${STAGE1_PROJECTOR_LR:-1e-4}"

# stage2：开启 LLM LoRA；projector 已预训练，只做微调。
STAGE2_LR="${STAGE2_LR:-5e-5}"          # LoRA lr 基线
STAGE2_LORA_LR="${STAGE2_LORA_LR:-5e-5}"
STAGE2_PROJECTOR_LR="${STAGE2_PROJECTOR_LR:-3e-5}"

# stage3：解冻 BEATs（162M）。BEATs 用极小 lr 避免破坏预训练表征；其它模块在 stage2 基础上再降一半。
STAGE3_LR="${STAGE3_LR:-3e-5}"          # LoRA 基线 lr（继续微调）
STAGE3_LORA_LR="${STAGE3_LORA_LR:-3e-5}"
STAGE3_PROJECTOR_LR="${STAGE3_PROJECTOR_LR:-1e-6}"
STAGE3_BEATS_LR="${STAGE3_BEATS_LR:-1e-6}"

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
if [[ ! -f "${BEATS_CKPT}" ]]; then
  echo "Missing BEATs checkpoint: ${BEATS_CKPT}" >&2
  exit 1
fi
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

echo "==========================================================="
echo " Multi-node config:"
echo "   NNODES=${NNODES}  NODE_RANK=${NODE_RANK}"
echo "   MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "   NPROC (GPUs per node) = ${NPROC}  GPUS=${GPUS}"
echo "   Global world size     = $((NNODES * NPROC))"
echo "   START_STAGE=${START_STAGE}"
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
      "${ROOT_DIR}/train_so_qa.py" "$@"
}

# 所有 stage 共用的 flag
common_args=(
  --model-id "${MODEL_ID}"
  --beats-checkpoint "${BEATS_CKPT}"
  --beats-repo "${BEATS_REPO}"
  --qa-root "${QA_ROOT}"
  --audio-root "${AUDIO_ROOT}"
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
  --max-grad-norm 1.0
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
  echo "[config] gradient_checkpointing = ENABLED（会减速但省显存）"
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
  echo "[config] valid_generate_full = ON (整个 valid 集都生成；每 epoch 耗时显著增加，但保存全量 predictions)"
else
  echo "[config] valid_generate_full = OFF (仅生成 ${VALID_GENERATE_MAX_SAMPLES:-32} 条；设 VALID_GENERATE_FULL=1 保存全集)"
fi

# ------------------------------------------------------------------
# Stage 1: projector_only
# ------------------------------------------------------------------
if (( START_STAGE <= 1 )); then
  echo "==========================================================="
  echo "[stage1] projector_only (${STAGE1_EPOCHS} epochs, lr=${STAGE1_LR})"
  echo "  → ${STAGE1_DIR}"
  echo "==========================================================="
  stage1_extra=()
  if [[ -n "${STAGE1_RESUME_CKPT:-}" ]]; then
    echo "  resume from: ${STAGE1_RESUME_CKPT}"
    stage1_extra+=(--resume-checkpoint-path "${STAGE1_RESUME_CKPT}")
    # Preserve optimizer / scheduler / step counter by default so training
    # continues on the same LR schedule. Set STAGE1_RESUME_MODEL_ONLY=1 to
    # only reload trainable weights and restart epoch 1 with fresh optimizer.
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
# Stage 2: encoder_lora（projector + LLM LoRA）
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
  run_train \
    "${common_args[@]}" \
    --encoder-lora \
    --resume-checkpoint-path "${STAGE2_RESUME_CKPT}" \
    --resume-model-only \
    --lr "${STAGE2_LR}" \
    --lora-lr "${STAGE2_LORA_LR}" \
    --projector-lr "${STAGE2_PROJECTOR_LR}" \
    --epochs "${STAGE2_EPOCHS}" \
    --output-dir "${STAGE2_DIR}"
fi

# ------------------------------------------------------------------
# Stage 3: beats_lora（projector + LLM LoRA + BEATs 全量可训）
# ------------------------------------------------------------------
if (( START_STAGE <= 3 )); then
  if [[ ! -f "${STAGE3_RESUME_CKPT}" ]]; then
    echo "Missing stage3 resume checkpoint: ${STAGE3_RESUME_CKPT}" >&2
    echo "Set START_STAGE=2 to produce it, or STAGE3_RESUME_CKPT=/path/to/best_trainable.pt." >&2
    exit 1
  fi
  echo "==========================================================="
  echo "[stage3] beats_lora (${STAGE3_EPOCHS} epochs, beats_lr=${STAGE3_BEATS_LR}, lora_lr=${STAGE3_LORA_LR}, proj_lr=${STAGE3_PROJECTOR_LR})"
  echo "  resume from: ${STAGE3_RESUME_CKPT}"
  echo "  → ${STAGE3_DIR}"
  echo "==========================================================="
  stage3_extra=()
  # By default stage3 starts from the previous stage's best ckpt and reloads
  # weights only (fresh optimizer + schedule). If you're resuming an interrupted
  # stage3 run mid-epoch (e.g. after a crash), set STAGE3_RESUME_MODEL_ONLY=0
  # so optimizer / scheduler / step-counter are restored and the LR schedule
  # continues from where it left off.
  if [[ "${STAGE3_RESUME_MODEL_ONLY:-1}" == "1" ]]; then
    stage3_extra+=(--resume-model-only)
    echo "  resume mode: MODEL ONLY (fresh optimizer, restart from epoch 1)"
  else
    echo "  resume mode: FULL (optimizer + scheduler + step counter restored)"
  fi
  run_train \
    "${common_args[@]}" \
    --beats-lora \
    --resume-checkpoint-path "${STAGE3_RESUME_CKPT}" \
    "${stage3_extra[@]}" \
    --lr "${STAGE3_LR}" \
    --lora-lr "${STAGE3_LORA_LR}" \
    --projector-lr "${STAGE3_PROJECTOR_LR}" \
    --beats-lr "${STAGE3_BEATS_LR}" \
    --epochs "${STAGE3_EPOCHS}" \
    --output-dir "${STAGE3_DIR}"
fi

echo "All requested stages finished. Run dir = ${RUN_ROOT}"
