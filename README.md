# MLSFT

Multilingual safety fine-tuning and refusal-rate evaluation.

## Documentation

- [Setup and Installation](docs/step_zero_initialization.md)
- [Repository layout](docs/repository_layout.md)
- [Source reference](docs/src_reference.md)
- [Orchestration reference](docs/orchestration_reference.md)
- [Data and results](docs/data_and_results.md)
- [Config reference](docs/config_reference.md)

## Common Commands

```bash
bash orchestration/run_single_test.sh
bash orchestration/run_all.sh
bash orchestration/run_param_pipeline.sh unsloth/qwen3-4b 2
python src/summarize_judgement_stats.py
```

## Canonical Paths

- Generated summaries live in `data/run_summaries/`.
- Consolidated CSV results live in `data/results/`.
- Active configs live in `configs/`.
- Model and evaluation outputs continue to live under `data/outputs/`.