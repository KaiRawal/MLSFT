# `orchestration/` Reference

Shell scripts in `orchestration/` are the supported way to run the project.

## Single test pipeline
- `orchestration/run_single_test.sh` runs the baseline fine-tune, English evaluation, and translated evaluation sequence.

## Batch runner
- `orchestration/run_all.sh` runs the expanded reproduction suite.
- It executes the parameterized model pipeline at epochs 1, 3, 5, and 8 for the currently supported models.
- It also runs the fixed large-Qwen pipelines for Qwen3-8B, Qwen3-14B, and Qwen3-32B.

## Parameterized model runner
- `orchestration/run_param_pipeline.sh` is the relocated model/epoch runner.
- It accepts `--force-reeval` plus a model name and epoch count.

## Model-specific scripts
- `orchestration/models/run_qwen3_8b_pipeline.sh`
- `orchestration/models/run_qwen3_14b_pipeline.sh`
- `orchestration/models/run_qwen3_32b_pipeline.sh`

These scripts now write run summaries to `data/run_summaries/`.
