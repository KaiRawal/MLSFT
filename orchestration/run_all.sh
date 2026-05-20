#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
SYSTEM_ID="$(uname -s)_$(uname -m)_$(hostname -s 2>/dev/null || echo unknown_host)"
SYSTEM_ID="$(echo "${SYSTEM_ID}" | tr ' /:' '___')"
RUN_ID="${SYSTEM_ID}_${RUN_TIMESTAMP}"

ORCH_LOG_ROOT="data/orchestrator_logs"
ORCH_RUN_DIR="${ORCH_LOG_ROOT}/${RUN_ID}"
ORCH_LOG_FILE="${ORCH_RUN_DIR}/run_all.log"
MASTER_FAILURE_REPORT="${ORCH_RUN_DIR}/master_failure_report.txt"

mkdir -p "${ORCH_RUN_DIR}"

# Stream the orchestration transcript to both stdout and a run-specific log file.
exec > >(tee -a "${ORCH_LOG_FILE}") 2>&1

SKIP_FINETUNE_IF_HF_EXISTS=false
SKIP_EVAL_IF_SUMMARY_COMPLETE=false
while [[ $# -gt 0 ]]; do
	case "$1" in
		--skip-finetune-if-hf-exists)
			SKIP_FINETUNE_IF_HF_EXISTS=true
			shift
			;;
		--skip-eval-if-summary-complete)
			SKIP_EVAL_IF_SUMMARY_COMPLETE=true
			shift
			;;
		-h|--help)
			echo "Usage: orchestration/run_all.sh [--skip-finetune-if-hf-exists] [--skip-eval-if-summary-complete]" >&2
			exit 0
			;;
		--)
			shift
			break
			;;
		-*)
			echo "Error: unsupported option '$1'." >&2
			exit 1
			;;
		*)
			echo "Error: run_all.sh does not accept positional arguments." >&2
			exit 1
			;;
	esac
done

PIPELINE_ARGS=()
if [[ "${SKIP_FINETUNE_IF_HF_EXISTS}" == "true" ]]; then
	PIPELINE_ARGS+=(--skip-finetune-if-hf-exists)
fi
if [[ "${SKIP_EVAL_IF_SUMMARY_COMPLETE}" == "true" ]]; then
	PIPELINE_ARGS+=(--skip-eval-if-summary-complete)
fi

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

declare -a FAILED_RUNS=()
TOTAL_RUNS=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

timestamp() {
	date +"%Y-%m-%d %H:%M:%S"
}

log_line() {
	local level="$1"
	shift
	echo "[$(timestamp)] [${level}] $*"
}

write_seed_summary() {
	local seed="$1"
	local summary_file="$2"
	local total_runs="$3"
	local success_runs="$4"
	local failure_runs="$5"
	shift 5

	{
		echo "Seed: ${seed}"
		echo "Run ID: ${RUN_ID}"
		echo "Artifact directory: ${ORCH_RUN_DIR}"
		echo "Run transcript: ${ORCH_LOG_FILE}"
		echo "Total runs for seed: ${total_runs}"
		echo "Successful runs: ${success_runs}"
		echo "Failed runs: ${failure_runs}"
		echo ""
		echo "Failed combinations"
		echo "----------------------------------------"
		if [[ "$#" -eq 0 ]]; then
			echo "None"
		else
			for failure_line in "$@"; do
				echo "${failure_line}"
			done
		fi
	} | tee "${summary_file}"
}

write_master_report() {
	local summary_file="$1"
	local total_runs="$2"
	local success_runs="$3"
	local failure_runs="$4"
	shift 4

	local success_rate="0"
	if [[ "${total_runs}" -gt 0 ]]; then
		success_rate="$((success_runs * 100 / total_runs))"
	fi

	{
		echo "Run ID: ${RUN_ID}"
		echo "Started: ${RUN_TIMESTAMP}"
		echo "Generated: $(timestamp)"
		echo "Artifact directory: ${ORCH_RUN_DIR}"
		echo "Run transcript: ${ORCH_LOG_FILE}"
		echo ""
		echo "Overall Summary"
		echo "----------------------------------------"
		echo "Total requested runs: ${total_runs}"
		echo "Successful runs: ${success_runs}"
		echo "Failed runs: ${failure_runs}"
		echo "Success rate: ${success_rate}%"
		echo ""
		echo "Failed Runs"
		echo "----------------------------------------"
		if [[ "$#" -eq 0 ]]; then
			echo "None"
		else
			for failure_line in "$@"; do
				echo "${failure_line}"
			done
		fi
		echo ""
		echo "Artifact Locations"
		echo "----------------------------------------"
		echo "Run transcript: ${ORCH_LOG_FILE}"
		echo "Per-seed summaries: ${ORCH_RUN_DIR}/seed_*_summary.txt"
		echo "Master report: ${summary_file}"
		echo "Existing pipeline summaries: data/run_summaries/<RUN_ID>.txt from each run_unified_pipeline invocation"
		echo "Detailed pipeline logs: data/run_summaries/<RUN_ID>_logs/ from each run_unified_pipeline invocation"
	} | tee "${summary_file}"
}

log_line INFO "Starting run_all orchestration."
log_line INFO "Run ID: ${RUN_ID}"
log_line INFO "Orchestration transcript: ${ORCH_LOG_FILE}"
log_line INFO "Per-seed summaries will be written under: ${ORCH_RUN_DIR}"
log_line INFO "Master failure report will be written to: ${MASTER_FAILURE_REPORT}"
log_line INFO "Child pipeline summaries remain under: data/run_summaries/"
log_line INFO "Skip fine-tune if HF exists flag: ${SKIP_FINETUNE_IF_HF_EXISTS}"
log_line INFO "Skip evals if consolidated summary complete flag: ${SKIP_EVAL_IF_SUMMARY_COMPLETE}"

# Run all models for each seed, then proceed to the next seed
for seed in "${SEEDS[@]}"; do
	seed_total_runs=0
	seed_success_count=0
	seed_failure_count=0
	declare -a seed_failures=()

	echo ""
	echo "===================================="
	echo "Running all models with seed: ${seed}"
	echo "===================================="
	echo ""
	log_line INFO "Seed ${seed}: starting."

	for epoch in "${EPOCHS[@]}"; do
		echo "--- Epoch ${epoch} ---"
		log_line INFO "Seed ${seed}: epoch ${epoch} starting."

		# Small parametrized models
		for model in "${SMALL_MODELS[@]}"; do
			TOTAL_RUNS=$((TOTAL_RUNS + 1))
			seed_total_runs=$((seed_total_runs + 1))
			log_line STARTED "Model=${model} Epoch=${epoch} Seed=${seed}"
			if bash orchestration/run_unified_pipeline.sh "${PIPELINE_ARGS[@]}" "${model}" "${epoch}" "${seed}"; then
				SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
				seed_success_count=$((seed_success_count + 1))
				log_line SUCCESS "Model=${model} Epoch=${epoch} Seed=${seed} completed successfully."
			else
				exit_code=$?
				FAILURE_COUNT=$((FAILURE_COUNT + 1))
				seed_failure_count=$((seed_failure_count + 1))
				failure_line="Model=${model} | Epoch=${epoch} | Seed=${seed} | Exit code=${exit_code}"
				FAILED_RUNS+=("${failure_line}")
				seed_failures+=("${failure_line}")
				log_line FAILURE "${failure_line}"
				log_line CONTINUE "Proceeding to the next model/epoch/seed combination."
			fi
		done

		# Large models
		for model in "${LARGE_MODELS[@]}"; do
			TOTAL_RUNS=$((TOTAL_RUNS + 1))
			seed_total_runs=$((seed_total_runs + 1))
			log_line STARTED "Model=${model} Epoch=${epoch} Seed=${seed}"
			if bash orchestration/run_unified_pipeline.sh "${PIPELINE_ARGS[@]}" "${model}" "${epoch}" "${seed}"; then
				SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
				seed_success_count=$((seed_success_count + 1))
				log_line SUCCESS "Model=${model} Epoch=${epoch} Seed=${seed} completed successfully."
			else
				exit_code=$?
				FAILURE_COUNT=$((FAILURE_COUNT + 1))
				seed_failure_count=$((seed_failure_count + 1))
				failure_line="Model=${model} | Epoch=${epoch} | Seed=${seed} | Exit code=${exit_code}"
				FAILED_RUNS+=("${failure_line}")
				seed_failures+=("${failure_line}")
				log_line FAILURE "${failure_line}"
				log_line CONTINUE "Proceeding to the next model/epoch/seed combination."
			fi
		done
	done

	seed_summary_file="${ORCH_RUN_DIR}/seed_${seed}_summary.txt"
	write_seed_summary "${seed}" "${seed_summary_file}" "${seed_total_runs}" "${seed_success_count}" "${seed_failure_count}" "${seed_failures[@]}"
	log_line INFO "Seed ${seed}: summary written to ${seed_summary_file}"
done

write_master_report "${MASTER_FAILURE_REPORT}" "${TOTAL_RUNS}" "${SUCCESS_COUNT}" "${FAILURE_COUNT}" "${FAILED_RUNS[@]}"

log_line INFO "Master failure report written to ${MASTER_FAILURE_REPORT}"
log_line INFO "Total runs: ${TOTAL_RUNS}, successes: ${SUCCESS_COUNT}, failures: ${FAILURE_COUNT}"

if [[ "${FAILURE_COUNT}" -eq 0 ]]; then
	log_line SUCCESS "run_all completed without failures."
	exit 0
fi

log_line FAILURE "run_all completed with failures, but all combinations were attempted."
exit 1


