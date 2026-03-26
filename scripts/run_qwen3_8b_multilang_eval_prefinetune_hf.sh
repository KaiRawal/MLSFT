#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${ROOT_DIR}/scripts/mls_eval_english.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Error: could not find evaluation script at ${PYTHON_SCRIPT}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

if [[ ! -d "external/sorry-bench" ]]; then
  echo "Error: external/sorry-bench directory not found. Please initialize sorry-bench first." >&2
  exit 1
fi

# Override with BASE_MODEL_PATH if you want to evaluate a different pre-finetune model.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"

echo "Current working directory: $(pwd)"
echo "Evaluation entrypoint: ${PYTHON_SCRIPT}"
echo "Pre-finetune model path: ${BASE_MODEL_PATH}"

run_eval() {
  local language="$1"
  local lang_code="$2"
  local model_id="Qwen3-8B-${lang_code}-SynthDolly-1A-PREFT"

  echo "Starting pre-finetune English evaluation for ${language} -> ${model_id}"

  python "${PYTHON_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${BASE_MODEL_PATH}",
  "model_id": "${model_id}",
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
echo "Running ${#LANGUAGE_RUNS[@]} pre-finetune English evaluation calls for ${BASE_MODEL_PATH}"
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
echo "Completed ${#LANGUAGE_RUNS[@]} pre-finetune English evaluation calls."
echo ""
echo ""
echo ""
