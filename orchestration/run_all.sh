#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

run_param() {
	bash orchestration/run_param_pipeline.sh --force-reeval "$1" "$2"
}

run_param unsloth/qwen3-0.6b 1
run_param unsloth/qwen3-0.6b 3
run_param unsloth/qwen3-0.6b 5
run_param unsloth/qwen3-0.6b 8

run_param unsloth/qwen3-4b 1
run_param unsloth/qwen3-4b 3
run_param unsloth/qwen3-4b 5
run_param unsloth/qwen3-4b 8

run_param unsloth/llama-3.2-1b-Instruct 1
run_param unsloth/llama-3.2-1b-Instruct 3
run_param unsloth/llama-3.2-1b-Instruct 5
run_param unsloth/llama-3.2-1b-Instruct 8

run_param unsloth/llama-3.2-3b-Instruct 1
run_param unsloth/llama-3.2-3b-Instruct 3
run_param unsloth/llama-3.2-3b-Instruct 5
run_param unsloth/llama-3.2-3b-Instruct 8

run_param unsloth/gemma-3-1b-it 1
run_param unsloth/gemma-3-1b-it 3
run_param unsloth/gemma-3-1b-it 5
run_param unsloth/gemma-3-1b-it 8

run_param unsloth/gemma-3-4b-it 1
run_param unsloth/gemma-3-4b-it 3
run_param unsloth/gemma-3-4b-it 5
run_param unsloth/gemma-3-4b-it 8

bash orchestration/models/run_qwen3_8b_pipeline.sh
bash orchestration/models/run_qwen3_14b_pipeline.sh
bash orchestration/models/run_qwen3_32b_pipeline.sh


