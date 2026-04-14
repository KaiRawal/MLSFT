# Repository Layout

This repository is organized around a small set of canonical directories.

## `src/`
Python entrypoints and utilities for training, evaluation, preflight checks, and result summarization.

## `orchestration/`
Shell scripts that run the pipeline end to end or in model-specific batches.

## `configs/`
Active JSON configs used by the Python entrypoints.

## `data/`
Inputs, outputs, and derived results.
- `data/inputs/` contains finetuning CSVs and evaluation prompts.
- `data/outputs/` contains generated model outputs.
- `data/results/` contains consolidated CSV summaries.
- `data/run_summaries/` contains run logs and summary text files.

## `docs/`
Repository documentation split by area.

## `analysis/`
Legacy location for summary CSVs. The canonical location is now `data/results/`.
