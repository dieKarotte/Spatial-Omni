# Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding

**Spatial audio understanding on top of Qwen-Omni.** Spatial-Omni augments
Qwen2.5-Omni / Qwen3-Omni LLMs with a dedicated spatial encoder (**SO-Encoder**)
that turns first-order ambisonic (FOA) audio into low-rate spatial tokens, and
injects them into the LLM via a `<|spatial|>` placeholder. Two model sizes ship
out of the box:

| Variant | Base LLM             | Spatial encoder | Trainer                       |
|---------|----------------------|-----------------|-------------------------------|
| SO-7B   | Qwen2.5-Omni-7B      | SO-Encoder      | `train_so_qa.py`              |
| SO-30B  | Qwen3-Omni-30B-A3B   | SO-Encoder      | `train_so_qa_qwen3.py`        |

Three spatial-encoder backbones are available:

- **SO-Encoder (BEATs-based, recommended)** — Spatial-pretrained BEATs
  encoder, projected to 2.5 Hz spatial tokens.
- **SELD path** — DCASE 2024 SELD baseline backbone + adapter.
- **IV / Neural-IV** — Lightweight intensity-vector baseline.

The two evaluation assets are:

- **[SO-Dataset](https://huggingface.co/datasets/dieKarotte/SO-Dataset)** — FOA SELD audio data with annotation & FOA spatial QA training data.
- **[SO-Bench](https://huggingface.co/datasets/dieKarotte/SO-Bench)**   — Held-out spatial QA test set.

### Contents

1. [Environment](#1-environment) — pip / conda install + env vars
2. [Data: SO-Dataset / SO-Bench](#2-data-so-dataset--so-bench) — download, extract, schema
3. [SO-Encoder pretraining & evaluation](#3-so-encoder-pretraining--evaluation) — `train_so_pretrain` + `bench_so_encoder`
4. [SO-7B QA fine-tuning (3 stages)](#4-so-7b-qa-fine-tuning-3-stages) — `train_so_qa` + SO-30B / IV / SELD variants
5. [Inference & evaluation on SO-Bench](#5-inference--evaluation-on-so-bench) — `bench_test_generate` + `score_test_predictions`
6. [Smoke tests](#6-smoke-tests) — unit + end-to-end install verification
7. [Repository layout](#7-repository-layout)
8. [Citations](#8-citations)
9. [License](#9-license)

---

## 1. Environment

### Option A — pip (SO-7B path)
```bash
git clone https://example.com/spatial-omni.git
cd Spatial-Omni
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install flash-attn deepspeed
```

### Option B — conda (SO-30B path, needs Qwen3 transformers fork)
```bash
conda env create -f environment-so30b.yml
conda activate spatial-omni
```

PyTorch ≥ 2.4, transformers = 4.52.0 (or 5.0.0.dev0 fork for SO-30B), peft ≥ 0.10.

### External dependencies (set once via env vars)
```bash
# 1) Microsoft unilm/BEATs (only needed for SO-Encoder pretraining; the QA
#    fine-tune flow imports the bundled copy from spatial_omni/encoders/beats)
git clone https://github.com/microsoft/unilm.git
export SO_BEATS_REPO=$PWD/unilm/beats

# 2) Upstream BEATs trunk weights (used as warm-start for SO-Encoder pretraining)
#    Download `BEATs_iter3_plus_AS2M.pt` from https://github.com/microsoft/unilm/tree/master/beats
#    Place it under $SO_BEATS_REPO/pretrain_ckpt/BEATs_iter3_plus_AS2M.pt/BEATs_iter3_plus_AS2M.pt
export SO_BEATS_TRUNK_CKPT=$SO_BEATS_REPO/pretrain_ckpt/BEATs_iter3_plus_AS2M.pt/BEATs_iter3_plus_AS2M.pt

# 3) SO-Dataset source-class vocabulary CSV
#    Ships with the SO-Dataset release; only needed for SO-Encoder pretraining
#    and bench (QA fine-tune does not use it).
export SO_VOCAB=/path/to/SO-Dataset/so_vocab.csv

# 4) HuggingFace base model
huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir ./Qwen2.5-Omni-7B
export SO_BASE_MODEL=$PWD/Qwen2.5-Omni-7B

# 4b) (optional) default SO-Encoder checkpoint for the QA fine-tune trainers.
#     If set, `train_so_qa.py` / `train_iv_qa.py` use it as the default for
#     `--beats-checkpoint`, so you don't have to repeat the path on every run.
# export SO_ENCODER_CKPT=/path/to/so_encoder_pretrained.pt

# 4c) (optional) repo root used by trainers/bench scripts to locate the bundled
#     spatial-omni Python package on sys.path. Defaults to the directory of
#     the script itself; only set this if you launch trainers from a different
#     working directory.
# export SO_REPO=$PWD

# 5) (optional) DCASE 2024 SELD baseline (only for SELD path)
git clone https://github.com/sharathadavanne/seld-dcase2024.git
export DCASE_BASELINE_REPO=$PWD/seld-dcase2024
export SELD_FEATURE_STATS_DIR=/path/to/seld_feat_label/...

# 6) SO-Dataset root
export SO_DATASET_ROOT=/path/to/SO-Dataset
```

> **Note**: For the **QA fine-tune flow only** (Section 3), you only need
> `SO_BASE_MODEL` and `SO_DATASET_ROOT` plus a pretrained SO-Encoder
> checkpoint (passed via `--beats-checkpoint`). Items 1–3 are required only
> if you want to *pretrain the SO-Encoder yourself* (Section 3.0).

---

## 2. Data: SO-Dataset / SO-Bench

### 2.1 Use the public release (recommended)

The 63-class SO-Dataset is released on HuggingFace as
[`dieKarotte/SO-Dataset`](https://huggingface.co/datasets/dieKarotte/SO-Dataset)
— ~1.1 TB of path-preserving tar shards. Download and extract:

```bash
pip install -U "huggingface_hub[cli]"
hf download dieKarotte/SO-Dataset --repo-type dataset --local-dir SO-Dataset

# Full extraction (1 TB; ~30 min I/O on 4 workers)
PYTHONPATH=. python scripts/data/extract_so_dataset.py \
    --src ./SO-Dataset --dst ./SO-Dataset \
    --splits train valid test --workers 4
```

Layout after extraction:
```
SO-Dataset/
├── audio/{train,valid,test}/foa_*.wav     (FOA, 4-ch, 16 kHz)
├── annotations/{train,valid,test}/foa_*.csv  (DCASE-style frame CSV;
│                                              including foa_*_src*.csv)
├── metadata/{train,valid,test}.jsonl      (scene-level JSONL)
├── qa/{train,valid,test}.jsonl            (1.45 M QA pairs total)
├── so_vocab.csv                           (63-class taxonomy, frequency-sort,
│                                            row N = SO-Encoder cls-head dim N)
└── manifests/                             (per-shard summary)
```

```bash
export SO_DATASET_ROOT=$PWD/SO-Dataset
export SO_VOCAB=$SO_DATASET_ROOT/so_vocab.csv
```

The release ships `so_vocab.csv` directly — you don't need to regenerate it.
The CSV's row order (FSD50K frequency-descending) **must match** the SO-Encoder
checkpoint's classification head, so do not re-sort. The `label_id` integer in
each `metadata/*.jsonl` source matches the `label_id` column of `so_vocab.csv`
(both `0..62`); the loader joins by the `label` string, but the integer is
kept consistent for users wiring custom pipelines.


### 2.2 QA jsonl schema

`qa/{train,valid,test}.jsonl` — one record per line:

```json
{
  "qa_id": "qa_eaf8872e6092f714cbc5",
  "split": "train",
  "audio_id": "foa_2562b7c3a26059f6030e",
  "audio_path": "audio/train/foa_2562b7c3a26059f6030e.wav",
  "dataset": "hm3d",
  "task_type": "counting",
  "task_name": "identify_source_by_location",
  "question": "What is the sound source located to the front-left and below the listener?",
  "answer": "A string instrument is the sound source coming from the front-left and below.",
  "canonical_answer": "string instrument",
  "source_refs": [{"class_id": 47, "class_name": "string_instrument",
                    "azimuth_deg": 67.057, "elevation_deg": -36.558, "distance_m": 1.61, ...}]
}
```

`audio_path` is relative to `SO-Dataset/`. Pass `--qa-root SO-Dataset/qa --audio-root SO-Dataset` to the trainers / bench scripts so the QA loader can find the audio. The optional `prompt` field, if missing, is auto-mirrored from `question` at load time.

**SO-Bench** = `qa/test.jsonl` (7,877 records). Pass it via `--qa-root SO-Dataset/qa --audio-root SO-Dataset --split test` to the bench tools.

### 2.3 Bring your own QA dataset

A QA root is just a directory with `train.jsonl` / `valid.jsonl` / `test.jsonl`. The required fields are `audio_path`, `(prompt | question)`, `answer`. FOA audio must be 4-channel FOA WAV at 16 kHz, ≤20 s.

---

## 3. SO-Encoder pretraining & evaluation

The SO-Encoder is a BEATs-based spatial audio backbone that maps a 20 s FOA
clip to 50 spatial tokens (2.5 Hz, fed into the LLM via `<|spatial|>`).
Pretraining it on SO-Dataset is OPTIONAL — you can also start from the
released checkpoint and skip straight to §4.

### 3.1 Pretrain on SO-Dataset

**Required checkpoints / paths**

| Asset | Source | Where |
|---|---|---|
| BEATs trunk weights | `BEATs_iter3_plus_AS2M.pt` from [microsoft/unilm/beats](https://github.com/microsoft/unilm/tree/master/beats) | `$SO_BEATS_TRUNK_CKPT` |
| Source-class vocab | Ships with SO-Dataset release | `$SO_VOCAB` |
| FOA audio + per-source CSVs | SO-Dataset extracted layout | `$SO_DATASET_ROOT` |

**Step 1 — Build a pretraining manifest** (resolves `audio_path` / trajectory CSVs to absolute paths and drops missing files):
```bash
PYTHONPATH=. python scripts/data/build_so_pretrain_manifest.py \
    --metadata-jsonl $SO_DATASET_ROOT/metadata/train.jsonl \
    --data-root      $SO_DATASET_ROOT \
    --output         $SO_DATASET_ROOT/pretrain-train.jsonl \
    --filter-missing
```

**Step 2 — Launch DDP pretraining (8× A100 recommended)**:
```bash
torchrun --nproc_per_node=8 -m spatial_omni.encoders.beats.train_so_pretrain \
    --preset ov1 --distributed --ddp-find-unused-parameters \
    --ov1-manifest          $SO_DATASET_ROOT/pretrain-train.jsonl \
    --source-vocab-path     $SO_VOCAB --source-num-classes 63 \
    --pretrained-beats-ckpt $SO_BEATS_TRUNK_CKPT \
    --batch-size 2 --num-epochs 30 --learning-rate 1e-4 \
    --output-dir ./runs/so_encoder_pretrain
```
- `--ddp-find-unused-parameters` is required (auxiliary heads go unused on
  some batches).
- Single-GPU smoke (`--batch-size 16`, no `--distributed`) also works.

**Step 3 (optional) — 30 s smoke** (8 train + 2 valid clips):
```bash
PYTHONPATH=. python scripts/data/build_so_pretrain_manifest.py \
    --metadata-jsonl $SO_DATASET_ROOT/metadata/train.jsonl \
    --data-root      $SO_DATASET_ROOT \
    --output         $SO_DATASET_ROOT/pretrain-train-tiny.jsonl \
    --filter-missing --max-records 8

PYTHONPATH=. python -m spatial_omni.encoders.beats.train_so_pretrain \
    --preset ov1 \
    --ov1-manifest $SO_DATASET_ROOT/pretrain-train-tiny.jsonl \
    --source-vocab-path $SO_VOCAB --source-num-classes 63 \
    --pretrained-beats-ckpt $SO_BEATS_TRUNK_CKPT \
    --batch-size 2 --num-epochs 1 --learning-rate 1e-4 \
    --output-dir /tmp/so_pretrain_smoke --no-progress
```

### 3.2 Evaluate an SO-Encoder checkpoint on the test split

`scripts/bench_so_encoder.py` reuses the trainer's `evaluate_one_epoch`, so
the metrics match the validation log printed during training (Official DCASE
SELD: F20, ER20, LE_CD, LR_CD, SELD_score, plus class accuracy and
azi/ele/dist MAE).

```bash
# 1) Build a test-split pretrain manifest
PYTHONPATH=. python scripts/data/build_so_pretrain_manifest.py \
    --metadata-jsonl $SO_DATASET_ROOT/metadata/test.jsonl \
    --data-root      $SO_DATASET_ROOT \
    --output         $SO_DATASET_ROOT/pretrain-test.jsonl \
    --filter-missing

# 2) Bench (~3 min on 1 GPU, 1.6 K clips)
PYTHONPATH=. python scripts/bench_so_encoder.py \
    --checkpoint            /path/to/so_encoder_best.pt \
    --test-manifest         $SO_DATASET_ROOT/pretrain-test.jsonl \
    --source-vocab          $SO_VOCAB \
    --pretrained-beats-ckpt $SO_BEATS_TRUNK_CKPT \
    --batch-size 4 --num-workers 4 \
    --output-json /tmp/so_encoder_test_metrics.json
```

Reference metrics (released SO-Encoder, 1,632 test clips):

| Metric | F20 | ER20 | LE_CD (°) | LR_CD | SELD_score | class_acc | azi_mae (°) | ele_mae (°) | dist_mae (m) |
|---|---|---|---|---|---|---|---|---|---|
| Value | 0.486 | 0.528 | 13.63 | 0.575 | 0.386 | 0.873 | 10.05 | 4.47 | 0.359 |

> **`--source-vocab` must be the SAME vocab the checkpoint was trained on**
> (frequency-sort, identical to the released `SO-Dataset/so_vocab.csv`).
> An alphabetical or reordered vocab will silently mis-route the cls head
> and tank F20 to ~0.

---

## 4. SO-7B QA fine-tuning (3 stages)

Three sequential stages, each resumes from the previous stage's
`best_trainable.pt`. Use the bundled launcher
[`shell/launch_train_so_7b.sh`](shell/launch_train_so_7b.sh) for the
recommended schedule, or run the trainers directly:

**Required checkpoints / paths**

| Asset | Where |
|---|---|
| Qwen2.5-Omni-7B base model | `$SO_BASE_MODEL` (or `--model-id`) |
| Pretrained SO-Encoder | `$SO_ENCODER_CKPT` (or `--beats-checkpoint`) |
| BEATs repo (for the wrapper class) | `$SO_BEATS_REPO` (or `--beats-repo`) |
| QA + audio | `$SO_DATASET_ROOT/qa` and `$SO_DATASET_ROOT` |

```bash
# Stage 1 — train projector only (SO-Encoder + LLM frozen)
torchrun --nproc_per_node=4 train_so_qa.py \
    --projector-only \
    --qa-root    $SO_DATASET_ROOT/qa \
    --audio-root $SO_DATASET_ROOT \
    --beats-checkpoint $SO_ENCODER_CKPT \
    --attn-impl sdpa \
    --output-dir ./runs/so7b_stage1 \
    --epochs 5 --lr 1e-4

# Stage 2 — LLM LoRA + projector (SO-Encoder still frozen)
torchrun --nproc_per_node=4 train_so_qa.py \
    --encoder-lora \
    --resume-checkpoint-path ./runs/so7b_stage1/checkpoints/best_trainable.pt \
    --resume-model-only \
    --qa-root    $SO_DATASET_ROOT/qa \
    --audio-root $SO_DATASET_ROOT \
    --beats-checkpoint $SO_ENCODER_CKPT \
    --attn-impl sdpa \
    --output-dir ./runs/so7b_stage2 \
    --epochs 3 --lr 3e-5

# Stage 3 — unfreeze SO-Encoder + LoRA + projector
torchrun --nproc_per_node=4 train_so_qa.py \
    --beats-lora \
    --resume-checkpoint-path ./runs/so7b_stage2/checkpoints/best_trainable.pt \
    --resume-model-only \
    --qa-root    $SO_DATASET_ROOT/qa \
    --audio-root $SO_DATASET_ROOT \
    --beats-checkpoint $SO_ENCODER_CKPT \
    --attn-impl sdpa \
    --output-dir ./runs/so7b_stage3 \
    --epochs 3 --lr 1e-5
```

Important details:

- `--attn-impl sdpa` is **required**. The default `auto` auto-detects
  flash-attn 2 when installed, but the upstream Qwen2.5-Omni
  `_update_causal_mask` raises on `padding_side='right'` + flash-attn 2 and
  the trainer right-pads (label mask is on the right). `sdpa` (PyTorch
  native) bypasses the check.
- `--audio-root` lets the QA loader find audio when `audio_path` is relative
  to a different root than `--qa-root` (the SO-Dataset HF release puts
  `qa/` and `audio/` as siblings under the dataset root).
- Legacy checkpoints with the old `spatial_beats_*` / `seld233_*` keys are
  loaded transparently — see
  [`spatial_omni/utils/ckpt_compat.py`](spatial_omni/utils/ckpt_compat.py).

### Continuing from a previously-trained SO-7B checkpoint

To resume training (e.g. stage-3 LoRA on a new QA distribution) from any
existing `best_trainable.pt`:

```bash
torchrun --nproc_per_node=8 train_so_qa.py \
    --beats-lora \
    --resume-checkpoint-path /path/to/best_trainable.pt \
    --resume-model-only \
    --qa-root    $SO_DATASET_ROOT/qa \
    --audio-root $SO_DATASET_ROOT \
    --model-id        $SO_BASE_MODEL \
    --beats-checkpoint $SO_ENCODER_CKPT \
    --beats-repo      $SO_BEATS_REPO \
    --attn-impl sdpa \
    --output-dir ./runs/so7b_continue \
    --epochs 3 --batch-size 2 --grad-accum-steps 3 \
    --lr 3e-5 --lora-lr 3e-5 --projector-lr 1e-6 --beats-lr 1e-6 \
    --lora-r 16 --lora-alpha 32 \
    --lora-target-modules q_proj k_proj v_proj o_proj \
    --encoder-token-rate 10.0 --projector-shuffle-factor 4 \
    --warmup-ratio 0.03
```

`--resume-model-only` reloads the model state dict but reinitializes the
optimizer + scheduler — use this when starting a fresh schedule on top of
old weights. Drop the flag to fully resume optimizer/scheduler from a
crashed run.

### SO-30B (Qwen3-Omni-30B-A3B)

Requires a Qwen3-Omni transformers build (`environment-so30b.yml`). The
launcher [`shell/launch_train_so_30b.sh`](shell/launch_train_so_30b.sh)
handles the schedule:
```bash
torchrun --nproc_per_node=8 train_so_qa_qwen3.py \
    --projector-only \
    --qa-root    $SO_DATASET_ROOT/qa \
    --audio-root $SO_DATASET_ROOT \
    --beats-checkpoint $SO_ENCODER_CKPT \
    --attn-impl sdpa \
    --output-dir ./runs/so30b_stage1
```

### IV / Neural-IV baseline

```bash
torchrun --nproc_per_node=4 train_iv_qa.py \
    --spatial-encoder-type iv \
    --projector-only \
    --qa-root    $SO_DATASET_ROOT/qa \
    --audio-root $SO_DATASET_ROOT \
    --attn-impl sdpa \
    --output-dir ./runs/iv_stage1
```
Bundled launcher: [`shell/launch_train_iv.sh`](shell/launch_train_iv.sh).
Switch to Neural-IV with `--spatial-encoder-type neural_iv`.

### SELD baseline

DCASE 2024 SELD baseline + adapter, bundled launcher:
[`shell/launch_train_seld.sh`](shell/launch_train_seld.sh) (driver:
`scripts/train_seld_qa.py`).

---

## 5. Inference & evaluation on SO-Bench

`scripts/bench_test_generate.py` reads `train_args.json` next to each
checkpoint, so you do **not** pass `--model-id` / `--beats-checkpoint` /
`--beats-repo` on the CLI — those come from the saved training config.

### 5.1 Generate predictions on the test split

```bash
python scripts/bench_test_generate.py \
    --run-dir ./runs/so7b_stage3 \
    --checkpoint-tags best \
    --qa-root    $SO_DATASET_ROOT/qa \
    --audio-root $SO_DATASET_ROOT \
    --split test
```
Predictions land in `./runs/so7b_stage3/bench/test/best/predictions.jsonl`
(one record per line, with `prediction` / `prediction_cleaned` /
`raw_exact_match` / `cleaned_exact_match`).

Flags worth knowing:
- `--checkpoint-tags best last epoch_003` — bench multiple ckpts under
  `<run-dir>/checkpoints/`.
- `--checkpoint-paths /path/a.pt /path/b.pt` — explicit list (overrides `--run-dir`).
- `--checkpoint-glob 'step_0*_trainable.pt'` — sweep step ckpts.
- `--max-samples 500` — quick smoke / debugging.
- `--task-names estimate_azimuth detect_source` — filter by task.
- `--batch-size 1 --num-beams 4 --max-new-tokens 96` — generation knobs.

### 5.2 Score predictions

```bash
python scripts/score_test_predictions.py \
    --predictions-jsonl ./runs/so7b_stage3/bench/test/best/predictions.jsonl \
    --qa-root           $SO_DATASET_ROOT/qa --split test \
    --output-json       ./runs/so7b_stage3/bench/test/best/metrics.json \
    --md-output         ./runs/so7b_stage3/bench/test/best/metrics.md
```
Computes per-task accuracy, azimuth/elevation MAE, distance MAE,
direction-bin EM, and detection-task F1. The default thresholds are
configurable (`--azimuth-threshold-deg`, `--elevation-threshold-deg`, etc.).

Optional **LLM judge** for borderline-EM single-label tasks (uses an
OpenAI-compatible API, env var `SO_LLM_API_KEY`):
```bash
python scripts/score_test_predictions.py \
    --predictions-jsonl ./runs/so7b_stage3/bench/test/best/predictions.jsonl \
    --qa-root $SO_DATASET_ROOT/qa --split test \
    --llm-judge --llm-model gpt-4o \
    --llm-base-url https://api.openai.com/v1 \
    --output-json /tmp/metrics_with_judge.json
```

### 5.3 Sweep multiple checkpoints

```bash
python scripts/batch_bench_so_qa.py \
    --run-dir ./runs/so7b_stage3 \
    --qa-root $SO_DATASET_ROOT/qa --audio-root $SO_DATASET_ROOT \
    --split test
```
Runs `bench_test_generate.py` + `score_test_predictions.py` for every
checkpoint under `<run-dir>/checkpoints/`, writing one `metrics.json` per
ckpt.

### 5.4 Unified driver across baselines

```bash
torchrun --nproc_per_node=4 scripts/run_bench.py \
    --baseline so-7b \
    --run-dir  ./runs/so7b_stage3 \
    --qa-root  $SO_DATASET_ROOT/qa --audio-root $SO_DATASET_ROOT \
    --split    test
```
Supported `--baseline` values: `so-7b`, `so-30b`, `zero-spatial`,
`zero-spatial-30b`, `iv`, `neural-iv`. The driver dispatches to the
appropriate sub-script based on the baseline.

### 5.5 Diagnostic ablations

Drop the Qwen mono `<|AUDIO|>` branch entirely (only spatial tokens reach
the LLM):
```bash
python scripts/bench_test_generate.py \
    --run-dir ./runs/so7b_stage3 --checkpoint-tags best \
    --qa-root $SO_DATASET_ROOT/qa --audio-root $SO_DATASET_ROOT \
    --split test --drop-mono-audio
```

Replace spatial input with zeros (measures how much the LLM relies on
spatial tokens):
```bash
python scripts/bench_test_generate.py \
    --run-dir ./runs/so7b_stage3 --checkpoint-tags best \
    --qa-root $SO_DATASET_ROOT/qa --audio-root $SO_DATASET_ROOT \
    --split test --spatial-ablation zero
```

Output dirs are auto-suffixed (`__drop_mono_audio`, `__zero` etc.) so you
don't overwrite the joint baseline.

---

## 6. Smoke tests

### 6.1 Unit / integration
```bash
PYTHONPATH=. python tests/test_so_encoder_integration.py     # forward shape
PYTHONPATH=. python tests/test_generation_decode_offset.py   # tokenizer / decode
```

### 6.2 End-to-end QA training smoke (~30 s on 1× A100)
```bash
PYTHONPATH=. torchrun --nproc_per_node=1 train_so_qa.py \
    --projector-only \
    --max-train-samples 8 --max-valid-samples 4 \
    --batch-size 1 --epochs 1 \
    --beats-checkpoint $SO_ENCODER_CKPT \
    --beats-repo       $SO_BEATS_REPO \
    --qa-root          $SO_DATASET_ROOT/qa \
    --audio-root       $SO_DATASET_ROOT \
    --attn-impl sdpa \
    --output-dir /tmp/so_smoke
```
Expected: `Resumed ...` (or `Loaded ckpt`), training reaches step 8,
`valid_loss` is finite, `/tmp/so_smoke/checkpoints/best_trainable.pt` is
written.

### 6.3 End-to-end bench smoke (8 predictions, ~1 min on 1× A100)
```bash
PYTHONPATH=. python scripts/bench_test_generate.py \
    --run-dir ./runs/so7b_stage3 --checkpoint-tags best \
    --qa-root $SO_DATASET_ROOT/qa --audio-root $SO_DATASET_ROOT \
    --split test --max-samples 8 \
    --output-dir /tmp/so7b_smoke
```
Expected: 8 lines in `/tmp/so7b_smoke/best/predictions.jsonl`, each with a
non-empty `prediction`. The startup log should show
`loaded missing=<N> unexpected=0` (the LoRA `missing_keys` are expected —
LLM base layers aren't in the trainable-only checkpoint).

---

## 7. Repository layout

```
Spatial-Omni/
├── README.md, LICENSE, .gitignore
├── requirements.txt, environment-so30b.yml
├── train_so_qa.py            # SO-7B trainer (3-stage)
├── train_so_qa_qwen3.py      # SO-30B trainer
├── train_iv_qa.py            # IV / Neural-IV baseline
├── spatial_omni/             # Python package
│   ├── model/                # Qwen2.5/Qwen3-Omni + Spatial-Thinker subclass
│   ├── modules/              # SO-Encoder, projectors, SELD adapter, IV adapter
│   ├── encoders/beats/       # BEATs (microsoft/unilm) + Spatial-Omni extension
│   ├── encoders/seldnet/     # DCASE 2024 SELD baseline (third-party)
│   ├── data/, utils/, ufb_banding/
├── scripts/
│   ├── data/                 # SO-Dataset extraction, manifest building, vocab
│   ├── bench_so_encoder.py   # SO-Encoder eval
│   ├── bench_test_generate.py / batch_bench_so_qa.py / run_bench.py
│   ├── score_test_predictions.py
│   ├── precompute_*.py       # optional feature caches
│   └── train_seld_qa.py
├── shell/                    # ready-to-run launcher recipes
│   ├── launch_train_so_7b.sh / launch_train_so_30b.sh
│   ├── launch_train_iv.sh / launch_train_seld.sh
│   ├── launch_bench_test.sh / bench_easy.sh
│   └── train_with_autorestart.sh / resume_iv.sh
├── configs/                  # DeepSpeed config(s)
└── tests/                    # integration smoke tests
```

---

## 8. Citations

If you use Spatial-Omni in academic work, please cite Qwen2.5-Omni / Qwen3-Omni,
Microsoft BEATs, and the DCASE 2024 SELD baseline as upstream contributions.

---

## 9. License

Apache 2.0. See [`LICENSE`](LICENSE). Third-party components retain their
original licenses (Apache 2.0 / MIT).
