#!/usr/bin/env bash
set -euo pipefail

python scripts/preflight_check.py

python scripts/mls_fine_tuning_with_templates.py --config configs/mls_fine_tuning_with_templates.json
python scripts/mls_eval_english.py --config configs/mls_eval_english.json
python scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py --config configs/nllb_200_mls_run_sorry_bench_with_translated_prompts.json
