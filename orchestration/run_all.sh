#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEEDS=(73 3407 9)

run_model_epoch_grid() {
	local model="$1"
	shift
	local epochs=("$@")

	for epoch in "${epochs[@]}"; do
		for seed in "${SEEDS[@]}"; do
			bash orchestration/run_unified_pipeline.sh "${model}" "${epoch}" "${seed}"
		done
	done
}

# Small parametrized models with multiple epochs
run_model_epoch_grid unsloth/qwen3-0.6b 1 3 5 8
run_model_epoch_grid unsloth/qwen3-4b 1 3 5 8
run_model_epoch_grid unsloth/llama-3.2-1b-Instruct 1 3 5 8
run_model_epoch_grid unsloth/llama-3.2-3b-Instruct 1 3 5 8
run_model_epoch_grid unsloth/gemma-3-1b-it 1 3 5 8
run_model_epoch_grid unsloth/gemma-3-4b-it 1 3 5 8

# Large models with multiple epochs
run_model_epoch_grid unsloth/Qwen3-8B 1 3 5 8
run_model_epoch_grid unsloth/Qwen3-14B 1 3 5 8
run_model_epoch_grid unsloth/Qwen3-32B 1 3 5 8
run_model_epoch_grid unsloth/Meta-Llama-3.1-8B-Instruct 1 3 5 8


