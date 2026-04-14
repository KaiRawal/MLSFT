# `src/` Reference

These are the main Python entrypoints under `src/`.

## Training and evaluation
- `src/mls_fine_tuning_with_templates.py` runs the main multilingual finetuning job.
- `src/mls_eval_english.py` runs English safety evaluation.
- `src/nllb_200_mls_run_sorry_bench_with_translated_prompts.py` runs translated evaluation.

## Setup and validation
- `src/preflight_check.py` checks required configs, inputs, and dependencies.
- `src/setup_step_zero.py` supports the initial local environment setup flow.
- `src/setup_step_zero.py` and `docs/step_zero_initialization.md` should be read together.

## Reporting
- `src/summarize_judgement_stats.py` aggregates evaluation CSVs into `data/results/`.

