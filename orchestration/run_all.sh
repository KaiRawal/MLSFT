#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEEDS=(73 3407 9)
EPOCHS=(1 3 5 8)

SMALL_MODELS=(
	unsloth/qwen3-0.6b
	unsloth/qwen3-4b
	unsloth/llama-3.2-1b-Instruct
	unsloth/llama-3.2-3b-Instruct
	unsloth/gemma-3-1b-it
	unsloth/gemma-3-4b-it
)

LARGE_MODELS=(
	unsloth/Qwen3-8B
	unsloth/Qwen3-14B
	unsloth/Qwen3-32B
	unsloth/Meta-Llama-3.1-8B-Instruct
)

# Run all models for each seed, then proceed to the next seed
for seed in "${SEEDS[@]}"; do
	echo ""
	echo "===================================="
	echo "Running all models with seed: ${seed}"
	echo "===================================="
	echo ""

	for epoch in "${EPOCHS[@]}"; do
		echo "--- Epoch ${epoch} ---"

		# Small parametrized models
		for model in "${SMALL_MODELS[@]}"; do
			bash orchestration/run_unified_pipeline.sh "${model}" "${epoch}" "${seed}"
		done

		# Large models
		for model in "${LARGE_MODELS[@]}"; do
			bash orchestration/run_unified_pipeline.sh "${model}" "${epoch}" "${seed}"
		done
	done
done


