# Step Zero: One-Time Manual Initialization

These steps replace Colab-time cloning/downloading/auth with a one-time local setup.

Automated option:

- You can run `scripts/setup_step_zero.sh` to perform a full local setup.
- It requires `HF_TOKEN` in your environment.
- It intentionally does not download language CSVs; keep placing those manually under `data/inputs`.

Local-first note:

- The default script/config flow is now local-only (no required HF upload after fine-tuning).
- Hugging Face auth is only needed when accessing gated/private assets or when you explicitly enable push_to_hub.

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

- data/inputs/fine_tuning/MLS - Fine-Tuning Data - Hindi - Sheet1.csv
- data/inputs/eval_prompts/MLSFT - Hindi Evaluation Prompts  - Sheet1.csv

## 7) Execute scripts

Run individually:

```bash
python scripts/mls_fine_tuning_with_templates.py --config configs/mls_fine_tuning_with_templates.json
python scripts/mls_eval_english.py --config configs/mls_eval_english.json
python scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py --config configs/nllb_200_mls_run_sorry_bench_with_translated_prompts.json
```

Or run sequentially:

```bash
bash scripts/run_pipeline.sh
```
