# MLSFT: Complete Documentation

## Quick Start

### Recommended Prerequisites

- Python 3.12+
- CUDA-compatible GPU
- 2000+ GB free disk space (for models and outputs)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kairawal/MLSFT.git
   cd MLSFT
   ```

2. **Automated setup:**
   ```bash
   bash orchestration/setup_environment.sh
   ```
   This installs dependencies and prepares the environment. Requires `HF_TOKEN` for gated model access.
   
   **Manual setup:** See [Step Zero: One-Time Manual Initialization](step_zero_initialization.md) for step-by-step instructions.

3. **Verify installation:**
   ```bash
   python src/preflight_check.py
   ```

## Running Experiments

### Common Commands

```bash
# Quick test on a single model variant
bash orchestration/run_single_test.sh

# Full pipeline for all model variants (all epochs and sizes)
bash orchestration/run_all.sh

# Run specific model variant
# Example: qwen3-4b for 2 epochs
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 2 73

# Generate summary statistics from results
python src/summarize_judgement_stats.py
```

### Pipeline Stages

Individual pipeline stages can be run separately:

```bash
bash orchestration/run_single_test.sh
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 2 73
```

The three JSON files under `configs/` remain the canonical stage templates used by `run_single_test.sh` and preflight validation.

### Output Locations

| Location | Contents |
|----------|----------|
| `data/run_summaries/` | Generated summary statistics |
| `data/results/` | Consolidated CSV results across all runs |
| `configs/` | Active experiment configurations |
| `data/outputs/eval_english/` | English evaluation results |
| `data/outputs/eval_translated/` | Translation evaluation results |
| `data/inputs/` | Input datasets (finetuning prompts & evaluation prompts) |

## Repository Structure

```
MLSFT/
├── src/                           # Core Python modules
│   ├── mls_fine_tuning_with_templates.py       # SFT pipeline
│   ├── mls_eval_english.py                      # English evaluation
│   ├── nllb_200_mls_run_sorry_bench_with_translated_prompts.py  # Translated eval
│   ├── summarize_judgement_stats.py             # Results aggregation
│   ├── gpu_selection.py                         # GPU utilities
│   ├── preflight_check.py                       # Setup verification
│   └── setup_step_zero.py                       # Environment setup
│
├── configs/                       # Experiment configurations (JSON)
│   ├── mls_fine_tuning_with_templates.json
│   ├── mls_eval_english.json
│   └── nllb_200_mls_run_sorry_bench_with_translated_prompts.json
│
├── orchestration/                 # Pipeline scripts
│   ├── run_all.sh                 # Full pipeline
│   ├── run_single_test.sh          # Quick test
│   ├── run_unified_pipeline.sh     # Specific model/epoch/seed
│   ├── setup_environment.sh        # Environment setup
│   ├── models/                     # Per-model pipelines
│   └── stages/                     # Per-stage pipelines
│
├── data/
│   ├── inputs/
│   │   ├── fine_tuning/            # CSV datasets for each language
│   │   └── eval_prompts/           # Evaluation prompts for each language
│   ├── outputs/                    # Model and evaluation outputs
│   ├── results/                    # Aggregated CSV results
│   └── run_summaries/              # Generated summary statistics
│
├── external/                      # External dependencies (git cloned)
│   ├── sorry-bench/                # Safety evaluation framework
│   └── FastChat/                   # Evaluation infrastructure
│
├── docs/                          # Documentation
│   ├── README.md                   # This file
│   ├── step_zero_initialization.md # Manual setup instructions
│   ├── repository_layout.md        # Detailed structure
│   ├── src_reference.md            # Python module details
│   ├── orchestration_reference.md  # Script & pipeline details
│   ├── data_and_results.md         # Data formats & result interpretation
│   ├── config_reference.md         # Configuration file options
│   └── ...
│
├── LICENSE
├── README.md                      # Main repository README
└── requirements.txt               # Python dependencies
```

## Datasets and Models

### Hugging Face Collections

- **[MLSFT Datasets](https://huggingface.co/collections/kairawal/mlsft-datasets)** — Dataset collection containing SynthDolly multilingual finetuning data and multilingual Sorry-Bench evaluation datasets.
- **[MLSFT Large Models (Epoch 1)](https://huggingface.co/collections/kairawal/mlsft-llms-e01)** — SynthDolly-finetuned larger LLM checkpoints after 1 epoch.
- **[MLSFT Small Models (Epoch 1)](https://huggingface.co/collections/kairawal/mlsft-smalllms-e01)** — SynthDolly-finetuned smaller LLM checkpoints after 1 epoch.
- **[MLSFT Small Models (Epoch 3)](https://huggingface.co/collections/kairawal/mlsft-smalllms-e03)** — SynthDolly-finetuned smaller LLM checkpoints after 3 epochs.
- **[MLSFT Small Models (Epoch 5)](https://huggingface.co/collections/kairawal/mlsft-smalllms-e05)** — SynthDolly-finetuned smaller LLM checkpoints after 5 epochs.
- **[MLSFT Small Models (Epoch 8)](https://huggingface.co/collections/kairawal/mlsft-smalllms-e08)** — SynthDolly-finetuned smaller LLM checkpoints after 8 epochs.

### Languages Included (except English)

- Chinese
- Danish  
- Greek
- Hindi
- Irish
- Portuguese
- Spanish
- Tagalog

## Reproducing Paper Results

### Full Reproduction

1. Follow [Quick Start](#quick-start) above
2. Download datasets and models from [Hugging Face Collections](#hugging-face-collections)
3. Run the full pipeline:
   ```bash
   bash orchestration/run_all.sh
   ```
4. Generate summary statistics:
   ```bash
   python src/summarize_judgement_stats.py
   ```
5. Review aggregated results in `data/results/`

### Partial Reproduction (Single Model)

To test with a single model variant first:

```bash
# Test with qwen3-4b, 2 epochs
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 2 73
```

### Data Requirements

- **Finetuning data:** Place CSVs in `data/inputs/fine_tuning/` (one per language)
- **Evaluation prompts:** Place CSVs in `data/inputs/eval_prompts/` (one per language, plus `sorry-bench-questions.jsonl`)
- **Models:** Downloaded automatically or via `setup_environment.sh`

### Expected Runtime

- **Quick test:** 10-30 minutes (single GPU)
- **Single model variant:** 1-4 hours (depending on model size)
- **Full pipeline:** 5+ days (multiple model sizes and epochs)

## Detailed Documentation

- **[Step Zero: One-Time Manual Initialization](step_zero_initialization.md)** — Manual installation and environment setup
- **[Repository Layout](repository_layout.md)** — Detailed directory structure and file organization
- **[Source Reference](src_reference.md)** — Python module documentation and function reference
- **[Orchestration Reference](orchestration_reference.md)** — Script and pipeline details
- **[Data and Results](data_and_results.md)** — Data formats and result interpretation
- **[Config Reference](config_reference.md)** — Configuration file structure and options

## Troubleshooting

**Issue: CUDA out of memory**
- Reduce batch size in config file
- Use a smaller model variant (e.g., 4B instead of 8B)
- Check GPU availability: `nvidia-smi`

**Issue: HuggingFace token required**
- Set `HF_TOKEN` environment variable:
  ```bash
  export HF_TOKEN=hf_xxx...
  ```
- Or authenticate manually:
  ```bash
  huggingface-cli login
  ```

**Issue: Missing external dependencies**
- Ensure external repos are cloned:
  ```bash
  git clone https://github.com/sorry-bench/sorry-bench.git external/sorry-bench
  git clone https://github.com/lm-sys/FastChat.git external/FastChat
  ```

For more issues, see [Step Zero](step_zero_initialization.md).
