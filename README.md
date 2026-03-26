# MLSFT Colab-to-Script Migration

This repository contains script-based replacements for the original Colab notebooks used for multilingual fine-tuning and refusal-rate evaluation. Some finetuned models can be found at [https://huggingface.co/kairawal/models](https://huggingface.co/kairawal/models).

## Manual Environment Setup (Required)

Manual setup is required because `FastChat` and `vLLM` dependencies can conflict across environments and versions. Keep environment preparation separate from runtime execution and do not rely on runtime scripts to clone/install large external assets.

Before running the new bash pipelines, export these variables:

```bash
export HF_USER="<your_hf_username_or_org>"
export HF_TOKEN="<your_hf_token>"

# Optional but recommended for explicit GPU pinning
export CUDA_VISIBLE_DEVICES="0"

# Recommended for vLLM worker process stability
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
```

Notes:

- `HF_USER` is required for fine-tune upload naming and for resolving non-local post-finetune model paths.
- `HF_TOKEN` is required for Hugging Face uploads and any gated/private downloads.
- `CUDA_VISIBLE_DEVICES` is optional; set it when you want deterministic single/multi-GPU selection.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` is recommended when running `sorry-bench` vLLM generation/judging flows.

For one-time asset setup and external repositories, follow:

- docs/step_zero_initialization.md

## Converted Notebooks

- Colabs/MLS_Fine_Tuning_with_Templates.ipynb -> scripts/mls_fine_tuning_with_templates.py
- Colabs/MLS_Eval_English.ipynb -> scripts/mls_eval_english.py
- Colabs/NLLB_200_MLS_Run_SORRY_Bench_with_translated_prompts.ipynb -> scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py

## Execution Order

1. scripts/mls_fine_tuning_with_templates.py
2. scripts/mls_eval_english.py
3. scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py

Equivalent high-level bash pipeline order for Qwen 8B:

1. pre-finetune English eval
2. pre-finetune translated eval
3. fine-tuning
4. post-finetune English eval
5. post-finetune translated eval

## Config Files

- configs/mls_fine_tuning_with_templates.json
- configs/mls_eval_english.json
- configs/nllb_200_mls_run_sorry_bench_with_translated_prompts.json

Each script accepts a config path:

```bash
python scripts/<script_name>.py --config configs/<config_name>.json
```

## Data Layout

Use meaningful subfolders under data:

- data/inputs/fine_tuning
- data/inputs/eval_prompts
- data/outputs/fine_tuning
- data/outputs/eval_english
- data/outputs/eval_translated

## One-Time Initialization

Do not clone repositories, install dependencies, or download large models inside runtime scripts.

Follow:

- docs/step_zero_initialization.md

This includes manual setup for:

- external/sorry-bench
- external/FastChat
- autorater model weights
- NLLB CTranslate2 model directory

## Google Sheets CSV Inventory

See:

- docs/google_sheets_download_list.md

This file lists all known sheets read/written by the original notebooks and their CSV replacements.

## Run All Sequentially

Legacy sequential run:

```bash
bash scripts/run_pipeline.sh
```

Qwen 8B multilingual direct pipeline (recommended):

```bash
bash scripts/run_qwen3_8b_multilang_direct_pipeline.sh
```

Individual Qwen 8B entrypoints:

```bash
# Pre-finetune evaluations
bash scripts/run_qwen3_8b_multilang_eval_prefinetune_hf.sh
bash scripts/run_qwen3_8b_multilang_translated_eval_prefinetune_hf.sh

# Fine-tuning
bash scripts/run_qwen3_8b_multilang_finetune_hf.sh

# Post-finetune evaluations
bash scripts/run_qwen3_8b_multilang_eval_hf.sh
bash scripts/run_qwen3_8b_multilang_translated_eval_hf.sh
```

This direct pipeline uses fixed Qwen3-8B model/language settings and runs, per language:

1. pre-finetune English eval
2. pre-finetune translated eval
3. fine-tuning (pushes to Hugging Face using HF_USER and HF_TOKEN)
4. post-finetune English eval (from Hugging Face)
5. post-finetune translated eval (from Hugging Face)

Expected outputs per language/model pair:

- 4 CSV files total (pre/post English plus pre/post translated)
- JSONL traces for answer and judgment artifacts in the same output directories

Python script entrypoints used by the direct pipeline:

```bash
python scripts/mls_eval_english.py
python scripts/mls_fine_tuning_with_templates.py
python scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py
```

## Super Controller

You can generate and cache full experiment configs for all languages with one command:

```bash
python scripts/super_controller.py --model-name unsloth/qwen3-0.6B
```

To also execute all generated jobs end-to-end:

```bash
python scripts/super_controller.py --model-name unsloth/qwen3-0.6B --run
```

Cache outputs are written under:

- data/cache/super_controller/<run_id>/configs
- data/cache/super_controller/<run_id>/artifacts
- data/cache/super_controller/<run_id>/logs
- data/cache/super_controller/<run_id>/results