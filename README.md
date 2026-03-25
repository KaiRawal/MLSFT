# MLSFT Colab-to-Script Migration

This repository contains script-based replacements for the original Colab notebooks used for multilingual fine-tuning and refusal-rate evaluation.

## Converted Notebooks

- Colabs/MLS_Fine_Tuning_with_Templates.ipynb -> scripts/mls_fine_tuning_with_templates.py
- Colabs/MLS_Eval_English.ipynb -> scripts/mls_eval_english.py
- Colabs/NLLB_200_MLS_Run_SORRY_Bench_with_translated_prompts.ipynb -> scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py

## Execution Order

1. scripts/mls_fine_tuning_with_templates.py
2. scripts/mls_eval_english.py
3. scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py

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

```bash
bash scripts/run_pipeline.sh
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