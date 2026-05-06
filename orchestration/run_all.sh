#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Small parametrized models with multiple epochs
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-0.6b 1
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-0.6b 3
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-0.6b 5
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-0.6b 8

bash orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 1
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 3
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 5
bash orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 8

bash orchestration/run_unified_pipeline.sh unsloth/llama-3.2-1b-Instruct 1
bash orchestration/run_unified_pipeline.sh unsloth/llama-3.2-1b-Instruct 3
bash orchestration/run_unified_pipeline.sh unsloth/llama-3.2-1b-Instruct 5
bash orchestration/run_unified_pipeline.sh unsloth/llama-3.2-1b-Instruct 8

bash orchestration/run_unified_pipeline.sh unsloth/llama-3.2-3b-Instruct 1
bash orchestration/run_unified_pipeline.sh unsloth/llama-3.2-3b-Instruct 3
bash orchestration/run_unified_pipeline.sh unsloth/llama-3.2-3b-Instruct 5
bash orchestration/run_unified_pipeline.sh unsloth/llama-3.2-3b-Instruct 8

bash orchestration/run_unified_pipeline.sh unsloth/gemma-3-1b-it 1
bash orchestration/run_unified_pipeline.sh unsloth/gemma-3-1b-it 3
bash orchestration/run_unified_pipeline.sh unsloth/gemma-3-1b-it 5
bash orchestration/run_unified_pipeline.sh unsloth/gemma-3-1b-it 8

bash orchestration/run_unified_pipeline.sh unsloth/gemma-3-4b-it 1
bash orchestration/run_unified_pipeline.sh unsloth/gemma-3-4b-it 3
bash orchestration/run_unified_pipeline.sh unsloth/gemma-3-4b-it 5
bash orchestration/run_unified_pipeline.sh unsloth/gemma-3-4b-it 8

# Large models with multiple epochs
# Epoch 1
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-8B 1
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-14B 1
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-32B 1
bash orchestration/run_unified_pipeline.sh unsloth/Meta-Llama-3.1-8B-Instruct 1

# Epoch 3
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-8B 3
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-14B 3
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-32B 3
bash orchestration/run_unified_pipeline.sh unsloth/Meta-Llama-3.1-8B-Instruct 3

# Epoch 5
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-8B 5
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-14B 5
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-32B 5
bash orchestration/run_unified_pipeline.sh unsloth/Meta-Llama-3.1-8B-Instruct 5

# Epoch 8
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-8B 8
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-14B 8
bash orchestration/run_unified_pipeline.sh unsloth/Qwen3-32B 8
bash orchestration/run_unified_pipeline.sh unsloth/Meta-Llama-3.1-8B-Instruct 8


