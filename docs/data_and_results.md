# `data/` Outputs and Results

The repository keeps inputs, generated outputs, summaries, and derived tables under `data/`.

## Inputs
- `data/inputs/fine_tuning/` contains the finetuning CSVs by language.
- `data/inputs/eval_prompts/` contains the evaluation prompt CSVs and `sorry-bench-questions.jsonl`.

## Generated outputs
- `data/outputs/eval_english/` stores English evaluation CSVs.
- `data/outputs/eval_translated/` stores translated evaluation CSVs.
- `data/outputs/fine_tuning/` stores per-run training artifacts.

## Run summaries
- `data/run_summaries/` stores pipeline summary text files and log directories.

## Consolidated results
- `data/results/judgement_stats_summary.csv` contains the per-file judgement counts.
- `data/results/compliance_rate_stats.csv` contains the pre/post compliance rate comparison table.
- These results are **automatically regenerated at the end of each orchestration run** (`orchestration/run_unified_pipeline.sh`), so they always reflect the latest evaluation outputs.

## Notes
- `analysis/` is no longer the canonical home for CSV summaries.
- The reporting scripts now write directly to `data/results/`.
