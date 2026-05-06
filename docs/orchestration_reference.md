# `orchestration/` Reference

Shell scripts in `orchestration/` are the supported way to run the project.

## Single test pipeline
- `orchestration/run_single_test.sh` runs the baseline fine-tune, English evaluation, and translated evaluation sequence.

## Batch runner
- `orchestration/run_all.sh` runs the expanded reproduction suite.
- It executes the parameterized model pipeline at epochs 1, 3, 5, and 8 for the currently supported models.
- It also runs the fixed large-Qwen pipelines for Qwen3-8B, Qwen3-14B, and Qwen3-32B.

## Config cleanup check
- `orchestration/check_config_references.sh` reports config files under `configs/` that have no repository references.

## Parameterized model runner
- `orchestration/run_unified_pipeline.sh` is the supported model/epoch/seed runner.
- It requires a model name, epoch count, and seed.

## Model-specific scripts
- `orchestration/models/run_qwen3_8b_pipeline.sh`
- `orchestration/models/run_qwen3_14b_pipeline.sh`
- `orchestration/models/run_qwen3_32b_pipeline.sh`

These scripts now write run summaries to `data/run_summaries/`.
