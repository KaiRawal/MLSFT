#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_SCRIPT="${ROOT_DIR}/src/nllb_200_mls_run_sorry_bench_with_translated_prompts.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Error: could not find translated evaluation script at ${PYTHON_SCRIPT}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

if [[ ! -d "external/sorry-bench" ]]; then
  echo "Error: external/sorry-bench directory not found. Please initialize sorry-bench first." >&2
  exit 1
fi

if [[ ! -d "external/nllb-3.3b-ct2-int8" ]]; then
  echo "Error: external/nllb-3.3b-ct2-int8 directory not found. Please prepare the NLLB CTranslate2 model first." >&2
  exit 1
fi

# Override with BASE_MODEL_PATH if you want to evaluate a different pre-finetune model.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"

echo "Current working directory: $(pwd)"
echo "Translated evaluation entrypoint: ${PYTHON_SCRIPT}"
echo "Pre-finetune model path: ${BASE_MODEL_PATH}"

run_eval_translated() {
  local language="$1"
  local lang_code="$2"
  local source_lang_code="$3"
  local model_id="Qwen3-8B-${lang_code}-SynthDolly-1A-PREFT"

  echo "Starting pre-finetune translated evaluation for ${language} -> ${model_id}"

  python "${PYTHON_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${BASE_MODEL_PATH}",
  "model_id": "${model_id}",
  "language_code": "${lang_code}",
  "source_lang_code": "${source_lang_code}",
  "local_prompt_csv": "data/inputs/eval_prompts/${language}_evaluation_prompts.csv",
  "english_prompt_jsonl": "data/inputs/eval_prompts/sorry-bench-questions.jsonl",
  "sorry_bench_dir": "external/sorry-bench",
  "nllb_ct2_dir": "external/nllb-3.3b-ct2-int8",
  "nllb_model_name": "facebook/nllb-200-3.3B",
  "translate_device": "auto",
  "translation_batch_size": 32,
  "output_dir": "data/outputs/eval_translated"
}
JSON
)
}

LANGUAGE_RUNS=(
  "Hindi|HI|hin_Deva"
  "Danish|DA|dan_Latn"
  "Chinese|ZH|zho_Hans"
  "Greek|EL|ell_Grek"
  "Irish|GA|gle_Latn"
  "Portuguese|PT|por_Latn"
  "Spanish|ES|spa_Latn"
  "Tagalog|TL|tgl_Latn"
)

echo ""
echo ""
echo ""
echo "Running ${#LANGUAGE_RUNS[@]} pre-finetune translated evaluation calls for ${BASE_MODEL_PATH}"
echo ""
echo ""
echo ""

for run in "${LANGUAGE_RUNS[@]}"; do
  IFS='|' read -r language lang_code source_lang_code <<< "${run}"
  run_eval_translated "${language}" "${lang_code}" "${source_lang_code}"
done

echo ""
echo ""
echo ""
echo "Completed ${#LANGUAGE_RUNS[@]} pre-finetune translated evaluation calls."
echo ""
echo ""
echo ""
