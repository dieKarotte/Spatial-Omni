# Spatial-Omni · SO-4B

[Paper](https://arxiv.org/abs/2606.10738) · [Models](https://huggingface.co/dieKarotte/Spatial-Omni) · [SO-Dataset](https://huggingface.co/datasets/dieKarotte/SO-Dataset) · [SO-Bench](https://huggingface.co/datasets/dieKarotte/SO-Bench)

Spatial-Omni integrates spatial audio understanding into **Phi-4-multimodal**. Phi-4's native audio path is retained; a separate SO-Encoder and projector supply first-order ambisonic (FOA) features, with spatial LoRA adapting the language model.

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

Use Linux, an NVIDIA GPU, Python 3.11, PyTorch 2.5.1, Transformers 4.52.0 and PEFT 0.17.1. SDPA is the default attention implementation.

```bash
git clone --branch SO-4B https://github.com/dieKarotte/Spatial-Omni.git
cd Spatial-Omni
conda env create -f environment.yml
conda activate spatial-omni-so-4b
export HF_ENDPOINT=https://hf-mirror.com
hf download microsoft/Phi-4-multimodal-instruct --local-dir checkpoints/Phi-4-multimodal-instruct
hf download dieKarotte/Spatial-Omni SO-Encoder/SO-Encoder.pt --local-dir checkpoints
export SO_BASE_MODEL="$PWD/checkpoints/Phi-4-multimodal-instruct"
export SO_ENCODER_CKPT="$PWD/checkpoints/SO-Encoder/SO-Encoder.pt"
export SO_CHECKPOINT="$PWD/checkpoints/SO-4B.pt"
export SO_MIX_CHECKPOINT="$PWD/checkpoints/SO-4B-MIX.pt"
```

The last two paths are locations for your compatible local adaptation checkpoints; SO-4B weights are pending publication. Download the complete base repository, including its tokenizer, processor and configuration files.

## Data

Use SO-Dataset for training and SO-Bench for evaluation. Follow the [download instructions and QA schema](https://github.com/dieKarotte/Spatial-Omni/tree/main#data), preserving `qa_id`, task metadata and reference answers. The examples below expect `qa/{train,valid,test}.jsonl` and dataset-relative audio paths.

```bash
export SO_DATASET_ROOT="$PWD/data/SO-Dataset"
export SO_BENCH_ROOT="$PWD/data/SO-Bench"
```

FOA recordings have four channels with W in channel 0; preserve the released channel order and coordinate convention. Audio is resampled to 16 kHz and limited to 20 seconds. The spatial branch projects 10 Hz encoder features to 2.5 Hz tokens. Mono QA uses the same JSONL schema with local mono audio.

## Inference

The evaluator loads Phi-4's custom model code through `trust_remote_code`. The commands below use `--merge-speech-lora`, which merges the frozen speech adapter into the base before loading spatial parameters. Match this setting to training; omit it for a checkpoint trained without merging.

```bash
python so_phi4/eval_sobench.py \
  --model-id "$SO_BASE_MODEL" --beats-checkpoint "$SO_ENCODER_CKPT" \
  --checkpoint "$SO_CHECKPOINT" --merge-speech-lora \
  --qa-root "$SO_BENCH_ROOT/qa" --audio-root "$SO_BENCH_ROOT" \
  --output-dir outputs/so-4b --batch-size 1 --max-samples 4 --max-new-tokens 48
```

Predictions are written to `outputs/so-4b/predictions_rank0.jsonl`. Use a fresh output directory for each run. For MIX mono input, prepare `data/mono/qa/test.jsonl` and its audio directory:

```bash
python so_phi4/eval_sobench.py \
  --model-id "$SO_BASE_MODEL" --beats-checkpoint "$SO_ENCODER_CKPT" \
  --checkpoint "$SO_MIX_CHECKPOINT" --merge-speech-lora --mono \
  --qa-root data/mono/qa --audio-root data/mono/audio --output-dir outputs/so-4b-mix-mono \
  --batch-size 1 --max-samples 4 --max-new-tokens 48
```

`--mono` defaults to the trained spatial-null bank. A checkpoint without that bank needs `--mono-fill foa_w` to use the W-only encoder path. Distributed evaluation writes one file per rank; combine them only after all ranks finish.

## Training

| Stage | `--train-mode` | Updated components |
| --- | --- | --- |
| 1 | `projector_only` | Spatial projector |
| 2 | `encoder_lora` | Projector and spatial language LoRA |
| 3 | `beats_lora` | Projector, spatial language LoRA and SO-Encoder |

The native audio encoder stays frozen. The launcher merges speech LoRA by default; set `MERGE_SPEECH_LORA=0` consistently for a recipe without that merge.

```bash
bash shell/launch_train_so_4b.sh check
bash shell/launch_train_so_4b.sh smoke
bash shell/launch_train_so_4b.sh full
```

`check` prints the three-stage plan. `smoke` uses two GPUs, 256 training records, 16 validation records and two epochs per stage; `full` defaults to eight GPUs. `GPUS` is the process count, while `CUDA_VISIBLE_DEVICES` selects devices. Set `OUTPUT_ROOT` to a new run directory, `PYTHON` to the training interpreter and `NUM_WORKERS` per process. Learning rates scale with effective batch size.

To train MIX from an SO checkpoint, provide a flat replay JSONL with `audio_path`, `question` and `answer`. Audio paths must resolve under the training root or be absolute:

```bash
torchrun --standalone --nproc_per_node=2 train_so_phi4.py \
  --model-id "$SO_BASE_MODEL" --beats-checkpoint "$SO_ENCODER_CKPT" \
  --resume-checkpoint-path "$SO_CHECKPOINT" --merge-speech-lora \
  --qa-root "$SO_DATASET_ROOT/qa" --audio-root "$SO_DATASET_ROOT" \
  --train-mode beats_lora --replay-qa-roots data/replay/train.jsonl --spatial-replay-ratio 3 \
  --batch-size 1 --grad-accum-steps 4 --epochs 2 --num-workers 2 \
  --lr 1e-6 --lora-lr 3e-5 --beats-lr 1e-6 --output-dir runs/so-4b-mix
```

MIX uses three spatial slots per replay slot and trains a spatial-null bank. `--replay-null-ratio` defaults to 0.5; remaining replay samples use the W-only encoder path.

## Checkpoints

Checkpoints are raw trainable state dictionaries, omitting frozen base weights and optimizer state. Keep the exact base revision and matching `train_args.json`, including LoRA rank/alpha and speech-merge settings. The evaluator expects a complete stage-3 checkpoint (defaults: rank 16, alpha 32); override `--lora-r` and `--lora-alpha` when required.

Training writes `train_args.json`, TensorBoard events and `checkpoints/last_trainable.pt`. `--resume-checkpoint-path` initializes model weights in a new run; it does not restore optimizer, scheduler or step state. Curriculum transitions permit newly enabled components, but partial component checkpoints are rejected. Existing outputs are protected. Run inference with the newly saved file and a fresh output directory to check reload.

## Evaluation

Generation and scoring are separate. For the four-question example, validate alignment without contacting a judge:

```bash
python scripts/score_sobench.py \
  --predictions-jsonl outputs/so-4b/predictions_rank0.jsonl \
  --qa-root "$SO_BENCH_ROOT/qa" --split test --expected-examples 4 --dry-run
```

For a full result, remove the generation limit and score all 7,877 questions. Set `OPENAI_API_KEY` before live judging:

```bash
python scripts/score_sobench.py \
  --predictions-jsonl outputs/so-4b/predictions_rank0.jsonl \
  --qa-root "$SO_BENCH_ROOT/qa" --split test --expected-examples 7877 \
  --model gpt-4o-mini --require-api --no-skip-sensitive-api-errors \
  --angle-threshold-deg 20 --elevation-threshold-deg 10 \
  --distance-threshold-m 1.0 --time-threshold-s 0.2 \
  --detect-source-time-policy event_only --spatial-temporal-time-policy semantic_times_iou \
  --output-json outputs/so-4b/score.json --judged-jsonl outputs/so-4b/judged.jsonl \
  --cache-jsonl outputs/so-4b/judge_cache.jsonl
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
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2606.10738},
}
```

## License

Code is released under [Apache 2.0](LICENSE). Base-model weights, datasets and third-party components retain their respective licenses.
