#!/usr/bin/env bash
set -euo pipefail

python src/preflight_check.py

echo ""
echo ""
echo ""
echo "Running MLS fine-tuning with templates..."
python src/mls_fine_tuning_with_templates.py --config configs/mls_fine_tuning_with_templates.json

echo ""
echo ""
echo ""
echo "Running MLS evaluation on English data..."
python src/mls_eval_english.py --config configs/mls_eval_english.json


echo ""
echo ""
echo ""
echo "Running MLS evaluation on multilingual data..."
python src/nllb_200_mls_run_sorry_bench_with_translated_prompts.py --config configs/nllb_200_mls_run_sorry_bench_with_translated_prompts.json
