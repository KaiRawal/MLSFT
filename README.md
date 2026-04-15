# The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning (MLSFT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging%20Face%20Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-fcd022?logo=huggingface&logoColor=000)](https://huggingface.co/collections/kairawal/mlsft-datasets)
[![Hugging%20Face%20Models](https://img.shields.io/badge/Hugging%20Face-Models-fcd022?logo=huggingface&logoColor=000)](https://huggingface.co/collections/kairawal/mlsft-llms-e01)

> Exploring how benign finetuning affects safety refusal and compliance rates in language models

## Overview

This repository contains the complete code, data, and results for the paper **"The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning"**. 

We investigate how benign multilingual supervised fine-tuning (MLSFT) of language models affects their compliance and refusal rates, both in English and the target languages. We perform finetuning using a 1000 datapoint multilingual (8 languages) synthetic version of the Dolly dataset, with manual quality verification for each target language. We then finetune various language models and evaluate changes in safety compliance using Sorry-Bench, both in the English and (via NLLB translation) in the target translated language. Our findings reveal that finetuning has heterogeneous impacts on compliance rates, with effects varying significantly across different models and languages.

**For installation, setup, orchestration, and reproduction details, see [docs/README.md](docs/README.md).**

### Key Contributions

- Benign finetuning produces **heterogeneous impacts** on safety complaince rates across models and languages.
- Effects on safety are measured using both **English and translated evaluations**.
- A comprehensive, reproducible multlilingual benchmark with both finetuning data and an evaluation pipeline, **verfied manually across 8 non-english languages**.

## Datasets and Models

All datasets, fine-tuned models, and evaluation outputs are available on Hugging Face:

- **[MLSFT Datasets](https://huggingface.co/collections/kairawal/mlsft-datasets)** — Dataset collection containing SynthDolly multilingual finetuning data and multilingual Sorry-Bench evaluation datasets for 8 non-english languages: Chinese, Danish, Greek, Hindi, Irish, Portuguese, Spanish, and Tagalog.
- **[MLSFT Large Models (Epoch 1)](https://huggingface.co/collections/kairawal/mlsft-llms-e01)** — SynthDolly-finetuned larger LLM checkpoints after 1 epoch.
- **[MLSFT Small Models (Epoch 1)](https://huggingface.co/collections/kairawal/mlsft-smalllms-e01)** — SynthDolly-finetuned smaller LLM checkpoints after 1 epoch.
- **[MLSFT Small Models (Epoch 3)](https://huggingface.co/collections/kairawal/mlsft-smalllms-e03)** — SynthDolly-finetuned smaller LLM checkpoints after 3 epochs.
- **[MLSFT Small Models (Epoch 5)](https://huggingface.co/collections/kairawal/mlsft-smalllms-e05)** — SynthDolly-finetuned smaller LLM checkpoints after 5 epochs.
- **[MLSFT Small Models (Epoch 8)](https://huggingface.co/collections/kairawal/mlsft-smalllms-e08)** — SynthDolly-finetuned smaller LLM checkpoints after 8 epochs.


## Pre-existing Results

Paper results are available in `data/results/`:

- `compliance_rate_stats.csv` — Compliance rate statistics across models and languages
- `judgement_stats_summary.csv` — Detailed safety judgment summaries

These are generated from individual eval runs in `data/outputs/` using the aggregation script:

```bash
python src/summarize_judgement_stats.py
```

## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@article{hawkins2026mlsft,
  title={Heterogeneous Impacts of Multilingual Safety Finetuning},
  author={Will Hawkins and Kai Rawal and Jonathan Rystrom and Stratis Tsirtsis and Zihao Fu and Greta Warren and Eoin D. Delaney and Ryan Brown and Sandra Wachter and Brent Mittelstadt and Chris Russell},
  year={2026},
  note={[Citation format to be updated upon publication]}
}
```

## Contact

For questions, feedback, or collaboration inquiries, the best way to reach the authors is by opening a GitHub issue in this repository.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

