# Step Zero: One-Time Manual Initialization

These steps replace Colab-time cloning/downloading/auth with a one-time local setup.

Automated option:

- You can run `orchestration/setup_environment.sh` to perform the local setup.
- export the following vars using your huggingface credentials: `HF_TOKEN`, `HF_USER`
- export `VLLM_WORKER_MULTIPROC_METHOD=spawn` for running evaluations
- [optional] export `CUDA_VISIBLE_DEVICES=2,3` depending on your GPU setup

## 1) Clone required repositories

Run once from project root:

```bash
git clone https://github.com/sorry-bench/sorry-bench.git external/sorry-bench
git clone https://github.com/lm-sys/FastChat.git external/FastChat
```

## 2) Install Python dependencies manually

Use your preferred environment manager, then install at least:

- pandas
- torch
- transformers
- datasets
- trl
- unsloth
- ctranslate2

Also install sorry-bench/FastChat runtime dependencies needed by:

- external/sorry-bench/gen_model_answer_vllm.py
- external/sorry-bench/gen_judgment_safety_vllm.py

## 3) Download sorry-bench dataset files

Place question files so scripts can read:

- data/inputs/eval_prompts/sorry-bench-questions.jsonl
- external/sorry-bench/data/sorry_bench/question.jsonl (scripts overwrite this at runtime)

If access is gated, authenticate manually before download.

## 4) Download autorater model weights

Required model path:

- external/sorry-bench/ckpts/finetuned_models/ft-mistral-7b-instruct-v0.2-sorry-bench-202406

If this requires Hugging Face auth, run the download manually before script execution.

## 5) Prepare NLLB CTranslate2 model once

Expected directory:

- external/nllb-3.3b-ct2-int8

Create this folder once (for example via ct2 conversion) before running translated evaluation.

## 6) Place CSV inputs in meaningful data folders

Expected examples:

- data/inputs/fine_tuning/Hindi_finetuning_data.csv
- data/inputs/eval_prompts/Hindi_evaluation_prompts.csv

## 7) Execute scripts
Run the single test pipeline:

```bash
bash orchestration/run_single_test.sh
```

Run the unified multi-model pipeline:

```bash
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 2 73
```

The three JSON files under `configs/` are still used by `run_single_test.sh` and `src/preflight_check.py`.

Run the expanded reproduction batch:

```bash
bash orchestration/run_all.sh
```
