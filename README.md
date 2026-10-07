# Spatial-Omni

**Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding**

[Paper](https://arxiv.org/abs/2606.10738) · [Models](https://huggingface.co/dieKarotte/Spatial-Omni) · [SO-Dataset](https://huggingface.co/datasets/dieKarotte/SO-Dataset) · [SO-Bench](https://huggingface.co/datasets/dieKarotte/SO-Bench)

Spatial-Omni adds spatial audio as an independent modality to audio-language models. A BEATs-based **SO-Encoder** extracts spatial features from four-channel first-order ambisonics (FOA), while the base model retains its native audio encoder. A projector supplies spatial tokens alongside the native audio representation. SO-Encoder emits features at **10 Hz**; the projector produces **2.5 spatial tokens/s**, or 50 tokens for a 20-second clip.

We provide SO-Dataset for encoder and QA training and SO-Bench for evaluation across 16 spatial-audio tasks, spanning sound detection, localization, spatial relations, and reasoning. The datasets combine real recordings and simulated scenes; see the [paper](https://arxiv.org/abs/2606.10738) for their construction.

![Spatial-Omni architecture](figures/model.png)

[Quick start](#quick-start) · [Data](#data) · [SO-Encoder](#so-encoder) · [Training](#training) · [Evaluation](#evaluation) · [Results](#results)

## Models

| Models | Base model | Code | Checkpoints |
|---|---|---|---|
| SO-7B / SO-7B-MIX | Qwen2.5-Omni-7B | [main](https://github.com/dieKarotte/Spatial-Omni/tree/main) | [SO](https://huggingface.co/dieKarotte/Spatial-Omni/tree/main/SO-7B/so) / [MIX](https://huggingface.co/dieKarotte/Spatial-Omni/tree/main/SO-7B/mix) |
| SO-30B / SO-30B-MIX | Qwen3-Omni-30B-A3B-Instruct | [SO-30B](https://github.com/dieKarotte/Spatial-Omni/tree/SO-30B) | Pending |
| SO-4B / SO-4B-MIX | Phi-4-multimodal | [SO-4B](https://github.com/dieKarotte/Spatial-Omni/tree/SO-4B) | Pending |
| SO-AF3 / SO-AF3-MIX | Audio Flamingo 3 | [SO-AF3](https://github.com/dieKarotte/Spatial-Omni/tree/SO-AF3) | Pending |

[SO-Encoder](https://huggingface.co/dieKarotte/Spatial-Omni/tree/main/SO-Encoder) is also available separately. Adaptation checkpoints contain the trained spatial encoder, projector, and language-model LoRA parameters; the original base model is required. SO-7B-MIX adds mono-audio replay and learned null spatial tokens. Each branch has its own environment and training entrypoints.

## Quick start

Use Linux, Python 3.11, and an NVIDIA GPU. The main environment uses PyTorch 2.5.1, Transformers 4.52.0, and PEFT 0.17.1; SDPA works without FlashAttention.

~~~bash
git clone https://github.com/dieKarotte/Spatial-Omni.git
cd Spatial-Omni
conda env create -f environment.yml
conda activate spatial-omni

hf download Qwen/Qwen2.5-Omni-7B --local-dir ckpts/base
hf download dieKarotte/Spatial-Omni --local-dir ckpts/Spatial-Omni
export SO_BASE_MODEL="$PWD/ckpts/base"
export SO_ENCODER_CKPT="$PWD/ckpts/Spatial-Omni/SO-Encoder/SO-Encoder.pt"
export SO_CHECKPOINT="$PWD/ckpts/Spatial-Omni/SO-7B/so/SO-7B.pt"
~~~

For an existing Python 3.11 environment, install `torch==2.5.1 torchaudio==2.5.1` from `https://download.pytorch.org/whl/cu124`, then `pip install -r requirements.txt`. Set `HF_ENDPOINT=https://hf-mirror.com` before downloads if a mirror is needed.

~~~bash
python scripts/infer.py \
  --model-id "$SO_BASE_MODEL" --checkpoint "$SO_CHECKPOINT" \
  --audio /path/to/foa.wav \
  --question "Where are the sound sources relative to the listener?"
~~~

Input must be **four-channel, 16 kHz FOA**, with the released dataset's channel order and W in channel 0. The model uses the first 20 seconds. The inference entrypoint checks the sample rate and channel count; convert other formats explicitly.

For MIX, set `SO_CHECKPOINT="$PWD/ckpts/Spatial-Omni/SO-7B/mix/SO-7B-MIX.pt"`. The same command handles FOA; add `--mono` for single-channel 16 kHz audio. Keep the downloaded folder structure: the loader reads the adjacent `train_args.json` and resolves the encoder path. `--beats-checkpoint` overrides it, and `--device-map auto` enables loading across visible GPUs.

## Data

Download [SO-Dataset](https://huggingface.co/datasets/dieKarotte/SO-Dataset) for training and [SO-Bench](https://huggingface.co/datasets/dieKarotte/SO-Bench) for the **7,877-question** evaluation. They are separate releases; the training dataset's test split is not a substitute for SO-Bench.

~~~bash
hf download dieKarotte/SO-Dataset --repo-type dataset --local-dir data/SO-Dataset
python scripts/data/extract_so_dataset.py \
  --src data/SO-Dataset --dst data/SO-Dataset --splits train valid test --workers 4
hf download dieKarotte/SO-Bench --repo-type dataset --local-dir data/SO-Bench

export SO_DATASET_ROOT="$PWD/data/SO-Dataset"
export SO_BENCH_ROOT="$PWD/data/SO-Bench"
export SO_VOCAB="$SO_DATASET_ROOT/so_vocab.csv"
~~~

The training release contains large audio and annotation shards. Select the required splits on its dataset page before downloading; the extractor supports `--n-shards` and `--dry-run`. Prepare SO-Bench audio according to its dataset card, preserving the paths referenced by its QA records.

~~~text
data/SO-Dataset/
  audio/{train,valid,test}/foa_*.wav
  annotations/{train,valid,test}/foa_*.csv
  metadata/{train,valid,test}.jsonl
  qa/{train,valid,test}.jsonl
  so_vocab.csv
~~~

Each QA line contains `audio_path`, `question` (or `prompt`), and `answer`. For example:

~~~json
{"audio_path":"audio/train/example.wav","question":"How many sound sources are audible?","answer":"Two."}
~~~

Pass `--qa-root "$SO_DATASET_ROOT/qa" --audio-root "$SO_DATASET_ROOT"` to resolve dataset-relative paths. Preserve benchmark `qa_id`, `task_name`, and answer metadata. The released vocabulary has 63 classes: **keep its row order and label mapping unchanged** so class IDs match the encoder head.

## SO-Encoder

The encoder implementation is included under [spatial_omni/encoders/beats](spatial_omni/encoders/beats). QA inference and training can start from the released SO-Encoder. To train the encoder itself, obtain `BEATs_iter3_plus_AS2M.pt` from [BEATs](https://github.com/microsoft/unilm/tree/master/beats) and build manifests from the extracted metadata:

<details>
<summary>Pretraining and encoder evaluation</summary>

~~~bash
export SO_BEATS_TRUNK_CKPT=/path/to/BEATs_iter3_plus_AS2M.pt
for split in train valid test; do
  python scripts/data/build_so_pretrain_manifest.py \
    --metadata-jsonl "$SO_DATASET_ROOT/metadata/$split.jsonl" \
    --data-root "$SO_DATASET_ROOT" \
    --output "$SO_DATASET_ROOT/pretrain-$split.jsonl"
done
~~~

The manifest builder resolves audio and trajectory paths. Use `--max-records` for a small subset; `--filter-missing` explicitly excludes records whose audio is absent.

~~~bash
torchrun --nproc_per_node=8 -m spatial_omni.encoders.beats.train_so_pretrain \
  --preset so_encoder --distributed \
  --train-manifest "$SO_DATASET_ROOT/pretrain-train.jsonl" \
  --valid-manifest "$SO_DATASET_ROOT/pretrain-valid.jsonl" \
  --source-vocab-path "$SO_VOCAB" --source-num-classes 63 \
  --pretrained-beats-ckpt "$SO_BEATS_TRUNK_CKPT" \
  --batch-size 8 --num-workers 4 --amp bf16 \
  --output-dir runs/so-encoder
~~~

The `so_encoder` preset defines the model, loss, and training schedule. To fine-tune the released encoder, add `--init-from-spatial-ckpt "$SO_ENCODER_CKPT"` and set `--learning-rate` and `--num-epochs` for the new data. `--resume` restores a training checkpoint. Logs and `best.pt`/`last.pt` are written under the output directory.

Evaluate an encoder on its held-out metadata split:

~~~bash
python scripts/bench_so_encoder.py \
  --checkpoint "$SO_ENCODER_CKPT" \
  --test-manifest "$SO_DATASET_ROOT/pretrain-test.jsonl" \
  --source-vocab "$SO_VOCAB" --pretrained-beats-ckpt "$SO_BEATS_TRUNK_CKPT" \
  --batch-size 4 --num-workers 4 --output-json runs/so-encoder-test.json
~~~

This reports SELD and localization metrics through the encoder's evaluation loop. It is separate from the language-model QA benchmark below.

</details>

## Training

The [three-stage launcher](shell/launch_train_so_7b.sh) trains SO-7B with a frozen native audio tower:

| Stage | Updated components | Epochs | Learning rates |
|---|---|---:|---|
| 1 | Spatial projector | 2 | Projector `1e-4` |
| 2 | Projector + language LoRA | 3 | Projector `3e-5`; LoRA `5e-5` |
| 3 | SO-Encoder + projector + LoRA | 3 | Encoder/projector `1e-6`; LoRA `3e-5` |

~~~bash
export RUN_ROOT=./runs/so7b-training
CHECK_ONLY=1 bash shell/launch_train_so_7b.sh
bash shell/launch_train_so_7b.sh
~~~

Defaults are 8 GPUs, batch size 2 per GPU, accumulation 3, BF16, and SDPA. `GPUS` selects device IDs; `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `NUM_WORKERS`, and `STAGE*_EPOCHS` override the recipe. For an initial two-GPU run, use `GPUS=0,1` and cap `MAX_TRAIN_SAMPLES`/`MAX_VALID_SAMPLES`. Learning rates in this launcher are explicit and do not automatically scale with batch size.

Each stage initializes from the preceding stage's `checkpoints/best_trainable.pt`. `START_STAGE` and `STAGE2_RESUME_CKPT`/`STAGE3_RESUME_CKPT` select a starting point. For direct training from a released checkpoint, use `train_so_qa.py --resume-checkpoint-path ... --resume-model-only`; release files omit optimizer state. The trainer inherits the checkpoint's LoRA/projector settings from its adjacent configuration unless explicitly overridden.

For **SO-7B-MIX**, prepare a replay directory with `train.jsonl` in the same QA schema and locally accessible mono audio:

~~~bash
export RESUME_CKPT="$PWD/ckpts/Spatial-Omni/SO-7B/so/SO-7B.pt"
export REPLAY_QA_ROOT=/path/to/replay
export OUTPUT_DIR=./runs/so7b-mix
bash shell/launch_train_so_7b_mix.sh check
bash shell/launch_train_so_7b_mix.sh train
~~~

MIX defaults to spatial:replay 3:1, 8 GPUs, batch size 1, accumulation 4, and one epoch. Its default LR is `1e-5` at global batch 32 and scales with effective batch size; `LR` overrides it. Use each released checkpoint's `train_args.json` for its recorded training settings and data mixture.

Choose separate run directories. Training writes configurations, checkpoints, validation output, and TensorBoard events; monitor them with `tensorboard --logdir runs`. Both launchers support a command preview, and `python train_so_qa.py --help` lists the trainer options.

## Evaluation

Generate predictions from the dedicated SO-Bench release:

~~~bash
export EVAL_ROOT=./outputs/so7b-bench
python scripts/bench_test_generate.py \
  --checkpoint-paths "$SO_CHECKPOINT" --model-id "$SO_BASE_MODEL" \
  --beats-checkpoint "$SO_ENCODER_CKPT" \
  --qa-root "$SO_BENCH_ROOT/qa" --audio-root "$SO_BENCH_ROOT" --split test \
  --batch-size 1 --num-workers 4 --dtype bfloat16 --attn-impl sdpa \
  --max-new-tokens 96 --num-beams 1 --output-dir "$EVAL_ROOT"
~~~

For an initial check, add `--max-samples 4` and choose a separate output directory. SO-7B writes `$EVAL_ROOT/SO-7B/predictions.jsonl`; MIX writes `$EVAL_ROOT/SO-7B-MIX/predictions.jsonl`. `--checkpoint-paths` accepts multiple checkpoints, and `--run-dir` selects a training run. Keep generation settings and the dataset revision fixed when comparing models.

Use [scripts/score_sobench.py](scripts/score_sobench.py) for task-aware scoring. Set `PREDICTIONS` to the generated file and validate its QA alignment:

~~~bash
export PREDICTIONS="$EVAL_ROOT/SO-7B/predictions.jsonl"
python scripts/score_sobench.py --predictions-jsonl "$PREDICTIONS" \
  --qa-root "$SO_BENCH_ROOT/qa" --expected-examples 7877 --dry-run
~~~

For a four-question generation check, use `--expected-examples 4` instead.

Live semantic judging reads `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` from the environment. Speech WER is computed locally.

~~~bash
python scripts/score_sobench.py --predictions-jsonl "$PREDICTIONS" \
  --qa-root "$SO_BENCH_ROOT/qa" --expected-examples 7877 \
  --model gpt-4o-mini --concurrency 8 --max-rpm 120 \
  --angle-threshold-deg 20 --elevation-threshold-deg 10 \
  --distance-threshold-m 1.0 --time-threshold-s 0.2 \
  --detect-source-time-policy event_only \
  --spatial-temporal-time-policy semantic_times_iou \
  --require-api --no-skip-sensitive-api-errors \
  --output-json "$EVAL_ROOT/score.json" --judged-jsonl "$EVAL_ROOT/judged.jsonl" \
  --cache-jsonl "$EVAL_ROOT/judge_cache.jsonl"
~~~

Retain predictions, QA IDs, judge cache, and scoring settings. The cache supports retries; `--offline` recomputes scores from existing judgements. Use separate caches for different judge models, prompts, and prediction sets. A complete report has 7,877 examples and zero API fallbacks, skipped API records, or scoring errors. The older `score_test_predictions.py` implements a different protocol.

## Results

Results for the **September 2026 SO-7B checkpoints**:

| Model | SO-Bench | MMAU test-mini | MMAU-Pro |
|---|---:|---:|---:|
| SO-7B | 70.06% | 60.50% | 45.30% |
| SO-7B-MIX | 71.72% | 64.50% | 51.86% |

SO-Bench uses 7,877 examples; MMAU test-mini uses 1,000 and MMAU-Pro 4,163 unique examples. These checkpoints update the general-audio results of the earlier paper checkpoints. SO-Bench averages task-specific per-question scores, including temporal IoU and the speech indicator WER ≤ 0.5.

The recorded SO-Bench protocol uses `gpt-4o-mini`, 20° azimuth, 10° elevation, **1 m distance and 0.2 s onset** tolerances. Appendix D of the paper states **0.5 m and 0.4 s**; the table uses the recorded protocol shown in the evaluation command.

## Citation

~~~bibtex
@misc{zhu2026spatialomnispatialaudiounderstanding,
  title={Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding},
  author={Zhiyuan Zhu and Yixuan Chen and Yiwen Shao and Wenxiang Guo and Changhao Pan and Yu Zhang and Yuxiang Wang and Wei Liu and Houhua Zhang and Chengkuan Zeng and Wenbo Cheng and Yunxi Liu and Rui Yang and Steve Yves and Liefeng Bo and Zhou Zhao},
  year={2026},
  eprint={2606.10738},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2606.10738}
}
~~~

## License and acknowledgements

Code is released under [Apache 2.0](LICENSE); published Spatial-Omni weights use [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Base models, datasets, and third-party components retain their own licenses. We build on [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), [BEATs](https://github.com/microsoft/unilm/tree/master/beats), and the [DCASE SELD baseline](https://github.com/sharathadavanne/seld-dcase2024).

The repository also includes [intensity-vector](train_iv_qa.py) and [SELD](scripts/train_seld_qa.py) comparison baselines. Their weights and configurations are separate from the SO-Encoder path above.
