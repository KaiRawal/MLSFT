#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${ROOT_DIR}/scripts/mls_eval_english.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Error: could not find evaluation script at ${PYTHON_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${HF_USER:-}" ]]; then
  echo "Error: HF_USER must be set (used to construct HF model paths)." >&2
  exit 1
fi

if [[ ! -d "external/sorry-bench" ]]; then
  echo "Error: external/sorry-bench directory not found. Please initialize sorry-bench first." >&2
  exit 1
fi

echo "Current working directory: $(pwd)"
echo "Evaluation entrypoint: ${PYTHON_SCRIPT}"

run_eval() {
  local language="$1"
  local lang_code="$2"
  local repo_name="Qwen3-8B-${lang_code}-SynthDolly-1A"

  echo "Starting evaluation for ${language} -> ${repo_name}"

  python "${PYTHON_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${repo_name}",
  "model_id": "${repo_name}",
  "sorry_bench_dir": "external/sorry-bench",
  "english_questions_jsonl": "data/inputs/eval_prompts/sorry-bench-questions.jsonl",
  "output_dir": "data/outputs/eval_english"
}
JSON
)
}

LANGUAGE_RUNS=(
  "Hindi|HI"
  "Danish|DA"
  "Chinese|ZH"
  "Greek|EL"
  "Irish|GA"
  "Portuguese|PT"
  "Spanish|ES"
  "Tagalog|TL"
)

echo ""
echo ""
echo ""
echo "Running ${#LANGUAGE_RUNS[@]} evaluation calls for Qwen/Qwen3-8B fine-tuned models"
echo ""
echo ""
echo ""

for run in "${LANGUAGE_RUNS[@]}"; do
  IFS='|' read -r language lang_code <<< "${run}"
  run_eval "${language}" "${lang_code}"
done

echo ""
echo ""
echo ""
echo "Completed ${#LANGUAGE_RUNS[@]} evaluation calls."
echo ""
echo ""
echo ""
