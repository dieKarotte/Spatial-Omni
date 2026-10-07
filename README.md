# Spatial-Omni · SO-30B

[Paper](https://arxiv.org/abs/2606.10738) · [Models](https://huggingface.co/dieKarotte/Spatial-Omni) · [SO-Dataset](https://huggingface.co/datasets/dieKarotte/SO-Dataset) · [SO-Bench](https://huggingface.co/datasets/dieKarotte/SO-Bench)

Spatial-Omni adds spatial audio understanding to **Qwen3-Omni-30B-A3B-Instruct**. This branch uses the model's Thinker and native audio encoder alongside a separate SO-Encoder and projector for first-order ambisonic (FOA) audio.

<img src="figures/model.png" alt="Spatial-Omni architecture" width="800">

## Models

| Branch | Backbone | Code | SO / MIX checkpoints |
| --- | --- | --- | --- |
| [main](https://github.com/dieKarotte/Spatial-Omni/tree/main) | Qwen2.5-Omni-7B | Available | [Available](https://huggingface.co/dieKarotte/Spatial-Omni/tree/main/SO-7B) |
| [SO-30B](https://github.com/dieKarotte/Spatial-Omni/tree/SO-30B) | Qwen3-Omni-30B-A3B-Instruct | Available | Pending |
| [SO-4B](https://github.com/dieKarotte/Spatial-Omni/tree/SO-4B) | Phi-4-multimodal | Available | Pending |
| [SO-AF3](https://github.com/dieKarotte/Spatial-Omni/tree/SO-AF3) | Audio Flamingo 3 | Available | Pending |

[SO-Encoder](https://huggingface.co/dieKarotte/Spatial-Omni/tree/main/SO-Encoder) is shared across the integrations. Adaptation checkpoints are specific to their backbone.

## Installation

Use the dedicated SO-30B environment: Python 3.12, PyTorch 2.9.0 (CUDA 12.8), Transformers 5.0.0 and PEFT 0.18.1. SDPA is the default. A full BF16 30B base needs approximately 60 GB for weights alone; DDP keeps a model replica on each GPU.

```bash
git clone --branch SO-30B https://github.com/dieKarotte/Spatial-Omni.git
cd Spatial-Omni
conda env create -f environment-so30b.yml
conda activate spatial-omni-30b
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir checkpoints/Qwen3-Omni-30B-A3B-Instruct
hf download dieKarotte/Spatial-Omni SO-Encoder/SO-Encoder.pt --local-dir checkpoints
export SO_BASE_MODEL="$PWD/checkpoints/Qwen3-Omni-30B-A3B-Instruct"
export SO_ENCODER_CKPT="$PWD/checkpoints/SO-Encoder/SO-Encoder.pt"
export SO_CHECKPOINT="$PWD/checkpoints/SO-30B.pt"
export SO_MIX_CHECKPOINT="$PWD/checkpoints/SO-30B-MIX.pt"
```

The last two paths are locations for your compatible local adaptation checkpoints; SO-30B weights are pending publication. Download the complete base repository, including its tokenizer, processor and configuration files.

## Data

Use SO-Dataset for training and SO-Bench for evaluation. Follow the [download instructions and QA schema](https://github.com/dieKarotte/Spatial-Omni/tree/main#data), preserving `qa_id`, task metadata and reference answers. The examples below expect `qa/{train,valid,test}.jsonl` and dataset-relative audio paths.

```bash
export SO_DATASET_ROOT="$PWD/data/SO-Dataset"
export SO_BENCH_ROOT="$PWD/data/SO-Bench"
```

FOA recordings have four channels with W in channel 0; preserve the released channel order and coordinate convention. Audio is resampled to 16 kHz and limited to 20 seconds. The spatial branch projects 10 Hz encoder features to 2.5 Hz tokens. Mono QA uses the same JSONL schema with local mono audio.

## Inference

Keep the adaptation file next to its `train_args.json`, or use the original `run/checkpoints/*.pt` plus `run/train_args.json` layout. The command explicitly selects the base and encoder paths:

```bash
python scripts/bench_test_generate_qwen3.py \
  --checkpoint-paths "$SO_CHECKPOINT" --model-id "$SO_BASE_MODEL" \
  --beats-checkpoint "$SO_ENCODER_CKPT" \
  --qa-root "$SO_BENCH_ROOT/qa" --audio-root "$SO_BENCH_ROOT" --split test \
  --batch-size 1 --num-workers 4 --max-samples 4 --max-new-tokens 48 \
  --dtype bfloat16 --attn-impl sdpa --output-dir outputs/so-30b
```

With the `SO-30B.pt` filename above, predictions are written to `outputs/so-30b/SO-30B/predictions.jsonl`. Other filenames use their stem with `_trainable` removed. Use a fresh output directory, or `--skip-existing` for completed predictions.

For a MIX checkpoint on mono QA, prepare `data/mono/qa/test.jsonl` and its audio directory. This evaluator supports the W-only spatial-encoder route:

```bash
python scripts/bench_test_generate_qwen3.py \
  --checkpoint-paths "$SO_MIX_CHECKPOINT" --model-id "$SO_BASE_MODEL" \
  --beats-checkpoint "$SO_ENCODER_CKPT" \
  --qa-root data/mono/qa --audio-root data/mono/audio --split test \
  --mono-audio-w-channel-spatial-encoder --batch-size 1 --num-workers 4 \
  --max-samples 4 --max-new-tokens 48 --output-dir outputs/so-30b-mix-mono
```

The mono command passes real audio to the native encoder and W-only FOA through SO-Encoder. It does not select the learned-null route used by the MIX training interface.

## Training

| Stage | Trainer option | Updated components |
| --- | --- | --- |
| 1 | `--projector-only` | Spatial projector |
| 2 | `--encoder-lora` | Projector and attention LoRA |
| 3 | `--beats-lora` | Projector, attention LoRA, SO-Encoder and MoE routers |

The launcher enables router training in stage 3; `STAGE2_TRAIN_MOE_ROUTER=1` also enables it in stage 2. Expert weights are not LoRA targets.

```bash
CHECK_ONLY=1 bash shell/launch_train_so_30b.sh
bash shell/launch_train_so_30b.sh
```

`CHECK_ONLY=1` checks the environment and input paths without training. The launcher defaults to eight GPUs; `GPUS=0,1` selects two devices. Use `MAX_TRAIN_SAMPLES=256 MAX_VALID_SAMPLES=16 STAGE1_EPOCHS=2 STAGE2_EPOCHS=2 STAGE3_EPOCHS=2` with a new `RUN_ROOT` for a bounded run. Select stages with `START_STAGE`/`END_STAGE`; `STAGE2_RESUME_CKPT` and `STAGE3_RESUME_CKPT` choose their inputs. The launcher requires train, valid and test QA files.

Set per-GPU `BATCH_SIZE`, `GRAD_ACCUM_STEPS` and stage learning rates for your effective batch size. `PYTHON_BIN` selects the interpreter and `NUM_WORKERS` is per process. DDP is the default training strategy; DeepSpeed is not integrated.

MIX starts from an SO checkpoint and adds mono replay and a learned spatial-null bank:

```bash
export REPLAY_QA_ROOT="$PWD/data/replay/train.jsonl"
export REPLAY_AUDIO_ROOT="$PWD/data/replay/audio"
export RESUME_CKPT="$SO_CHECKPOINT"
bash shell/launch_train_so_30b_mix.sh check
bash shell/launch_train_so_30b_mix.sh smoke
bash shell/launch_train_so_30b_mix.sh full
```

Replay accepts flat `audio_path`/`question`/`answer` records and the [supported replay schema](https://github.com/dieKarotte/Spatial-Omni/tree/main#training). `smoke` uses two processes and bounded data; `full` defaults to eight. The full recipe uses a 0.2 spatial subset with the complete supplied replay set. `TRAIN_SUBSET_RATIO` changes that subset, not a fixed per-batch ratio. Set `OUTPUT_DIR` to a new directory. The default learning rate scales with effective batch size; `LR` overrides it.

## Checkpoints

Keep the trained projector, encoder, attention LoRA and all trained MoE router tensors. MIX also requires `spatial_null`. Frozen Thinker weights come from the base model. Retain the exact base revision, `train_args.json`, LoRA rank/alpha and router settings with the adaptation.

`--resume-checkpoint-path` loads a training checkpoint. `--resume-model-only` starts the optimizer, scheduler and step count fresh; the stage and MIX launchers use it for weight initialization. Full-state resume requires a checkpoint containing that state. Tensor-only public checkpoints load with `weights_only=True`; `--trust-checkpoint` is an explicit opt-in for trusted legacy pickle payloads.

Training writes checkpoint files, configuration and TensorBoard events under the run directory. Evaluation rejects missing trained components, unexpected keys and shape mismatches; curriculum transitions can introduce newly trained components. Evaluate `checkpoints/last_trainable.pt` with a fresh output directory to check reload.

## Evaluation

Generation and scoring are separate. For the four-question example, validate alignment without contacting a judge:

```bash
python scripts/score_sobench.py \
  --predictions-jsonl outputs/so-30b/SO-30B/predictions.jsonl \
  --qa-root "$SO_BENCH_ROOT/qa" --split test --expected-examples 4 --dry-run
```

For a full result, remove the generation limit and score all 7,877 questions. Set `OPENAI_API_KEY` before live judging:

```bash
python scripts/score_sobench.py \
  --predictions-jsonl outputs/so-30b/SO-30B/predictions.jsonl \
  --qa-root "$SO_BENCH_ROOT/qa" --split test --expected-examples 7877 \
  --model gpt-4o-mini --require-api --no-skip-sensitive-api-errors \
  --angle-threshold-deg 20 --elevation-threshold-deg 10 \
  --distance-threshold-m 1.0 --time-threshold-s 0.2 \
  --detect-source-time-policy event_only --spatial-temporal-time-policy semantic_times_iou \
  --output-json outputs/so-30b/score.json --judged-jsonl outputs/so-30b/judged.jsonl \
  --cache-jsonl outputs/so-30b/judge_cache.jsonl
```

Use `--offline` with a complete judge cache to recompute local scores. See the [evaluation protocol](https://github.com/dieKarotte/Spatial-Omni/tree/main#evaluation) for aggregation, API error handling and cache reuse.

The recorded protocol uses 20° azimuth, 10° elevation, 1 m distance and 0.2 s onset tolerances; the paper's Appendix D lists different distance/onset tolerances. Report the protocol alongside scores. A four-question check and the legacy `score_test_predictions.py` are not full SO-Bench results.

## Citation

```bibtex
@misc{zhu2026spatialomnispatialaudiounderstanding,
  title={Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding},
  author={Zhiyuan Zhu and Yixuan Chen and Yiwen Shao and Wenxiang Guo and Changhao Pan and Yu Zhang and Yuxiang Wang and Wei Liu and Houhua Zhang and Chengkuan Zeng and Wenbo Cheng and Yunxi Liu and Rui Yang and Steve Yves and Liefeng Bo and Zhou Zhao},
  year={2026},
  eprint={2606.10738},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```

## License

Code is released under [Apache 2.0](LICENSE). Base-model weights, datasets and third-party components retain their respective licenses.
