# `orchestration/` Reference

Shell scripts in `orchestration/` are the supported way to run the project.

## Single test pipeline
- `orchestration/run_single_test.sh` runs the baseline fine-tune, English evaluation, and translated evaluation sequence.

## Batch runner
- `orchestration/run_all.sh` runs the expanded reproduction suite.
- It executes the parameterized model pipeline **organized by random seed** (outer loop), then model, then epochs.
- For each seed (73, 3407, 9), all supported models are run at epochs 1, 3, 5, and 8.
- This seed-major ordering ensures all models and epochs complete for one seed before moving to the next.
- All runs with a given seed finish before any runs with the next seed begin.
- Failures from `run_unified_pipeline.sh` do **not** stop the batch; each failure is logged and the loop continues.
- `run_all.sh` accepts `--skip-finetune-if-hf-exists` and `--skip-eval-if-summary-complete`, forwarding them to each `run_unified_pipeline.sh` invocation.
- `run_all.sh` writes orchestration-level artifacts to `data/orchestrator_logs/<RUN_ID>/`:
  - `run_all.log` for the full batch transcript
  - `seed_<seed>_summary.txt` for per-seed summaries
  - `master_failure_report.txt` for the consolidated failure report

## Parameterized model runner
- `orchestration/run_unified_pipeline.sh` is the supported model/epoch/seed runner.
- It requires a model name, epoch count, and seed.
- It accepts two, narrower flags:
  - `--skip-finetune-if-hf-exists`: when enabled, the runner checks Hugging Face for the target finetuned repo and skips only the fine-tune/upload step if the repo exists (evaluations still run by default).
  - `--skip-eval-if-summary-complete`: when enabled, the runner consults `data/results/compliance_rate_stats.csv` and skips all evaluation steps for that model if the consolidated summary row exists and all four rates are present.
- If neither flag is set, the pipeline proceeds normally.

Example:
- `orchestration/run_unified_pipeline.sh --skip-finetune-if-hf-exists --skip-eval-if-summary-complete unsloth/qwen3-4b 3 73`

## Model-specific scripts
- `orchestration/models/run_qwen3_8b_pipeline.sh`
- `orchestration/models/run_qwen3_14b_pipeline.sh`
- `orchestration/models/run_qwen3_32b_pipeline.sh`

## Post-upload model organization
After each successful fine-tuning and upload to Hugging Face, the pipeline automatically organizes models into collections by epoch and seed.
- Collection naming: `MLSFT-Models-E{epoch}-S{seed}` (e.g., `MLSFT-Models-E1-S73`)
- Expected total collections: 4 epochs × 3 seeds = 12 collections
- The organization step runs via `src/organise.py` and is idempotent:
  - Reuses existing collections if they already exist
  - Skips repos that are already in any collection
  - Only adds uncollected repos to the target collection
- **Environment variables required:**
  - `HF_USER`: Hugging Face username (set in environment or `.env` file)
  - `HF_TOKEN`: Hugging Face user access token (set in environment or `.env` file)
- If organization fails (e.g., network issues), the pipeline continues with evaluation steps. Check logs to retry manually.

These scripts write pipeline summaries to `data/run_summaries/`, while `run_all.sh` writes orchestration logs to `data/orchestrator_logs/<RUN_ID>/`.
