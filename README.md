# MLSFT: Multilingual Safety Fine-Tuning

This repository contains the publication-oriented implementation for multilingual safety fine-tuning and refusal-rate evaluation. The workflow supports model fine-tuning, English safety evaluation, and translated safety evaluation across languages.

## Research Goal

The project studies whether multilingual fine-tuning can improve target-language behavior while preserving safety/refusal behavior measured in English and translated evaluation settings.

## Repository Layout

- `src/`: Python entrypoints for training and evaluation
- `orchestration/`: Shell orchestration scripts (model pipelines and stage runners)
- `configs/`: JSON configs used by Python scripts
- `data/inputs/`: input datasets and prompts (preserved)
- `data/outputs/`: generated outputs (CSV retained, JSONL removed from published state)
- `analysis/`: final analysis inputs/outputs used for reporting
- `docs/`: setup and initialization documentation

## Core Python Entry Points

All Python code now lives under `src/`.

- `src/mls_fine_tuning_with_templates.py`
- `src/mls_eval_english.py`
- `src/nllb_200_mls_run_sorry_bench_with_translated_prompts.py`
- `src/setup_step_zero.py`
- `src/preflight_check.py`
- `src/super_controller.py`
- `src/gpu_selection.py`
- `src/upload_datasets_to_hf.py`

Each script accepts `--config` where applicable:

```bash
python src/<script_name>.py --config configs/<config_name>.json
```

## Reproducibility Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set required environment variables:

```bash
export HF_USER="<your_hf_username_or_org>"
export HF_TOKEN="<your_hf_token>"
export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
```

Then follow one-time setup instructions in:

- `docs/step_zero_initialization.md`

External dependencies (`external/sorry-bench`, `external/FastChat`) are intentionally not published in this repository and must be set up locally.

## Running Pipelines

Run the sequential baseline pipeline:

```bash
bash orchestration/run_complete_pipeline.sh
```

Run a model-specific multilingual pipeline:

```bash
bash orchestration/models/run_qwen3_8b_pipeline.sh
```

Parameter-driven experiment runner:

```bash
bash orchestration/models/run_param_pipeline.sh unsloth/qwen3-4b 2
```

Research batch runner:

```bash
bash orchestration/run_research_repro_suite.sh
```

## Data Policy for Publication

- `data/inputs/` is preserved.
- `analysis/compliance_rate_stats.csv` and `analysis/judgement_stats_summary.csv` are preserved.
- Under `data/outputs/`, CSV outputs are preserved while JSONL artifacts are removed from the publishable state.
- Directory structure under `data/outputs/` is preserved with placeholder files.