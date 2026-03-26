#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENGLISH_EVAL_SCRIPT="scripts/mls_eval_english.py"
FINETUNE_SCRIPT="scripts/mls_fine_tuning_with_templates.py"
TRANSLATED_EVAL_SCRIPT="scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py"

for script in "${ENGLISH_EVAL_SCRIPT}" "${FINETUNE_SCRIPT}" "${TRANSLATED_EVAL_SCRIPT}"; do
  if [[ ! -f "${script}" ]]; then
    echo "Error: missing required python script: ${script}" >&2
    exit 1
  fi
done

if [[ -z "${HF_USER:-}" ]]; then
  echo "Error: HF_USER must be set for fine-tuning uploads." >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN must be set for fine-tuning uploads." >&2
  exit 1
fi

if [[ ! -d "external/sorry-bench" ]]; then
  echo "Error: external/sorry-bench directory not found. Please initialize sorry-bench first." >&2
  exit 1
fi

if [[ ! -d "external/nllb-3.3b-ct2-int8" ]]; then
  echo "Error: external/nllb-3.3b-ct2-int8 directory not found. Please prepare the NLLB CTranslate2 model first." >&2
  exit 1
fi

echo "Current working directory: $(pwd)"
echo "Using fixed base model: Qwen/Qwen3-8B"
echo "Pipeline per language: pre-English eval -> pre-translated eval -> fine-tune -> post-English eval -> post-translated eval"

run_language_pipeline() {
  local language="$1"
  local lang_code="$2"
  local source_lang_code="$3"

  local repo_name="Qwen3-8B-${lang_code}-SynthDolly-1A"
  local preft_model_id="${repo_name}-PREFT"

  echo ""
  echo ""
  echo ""
  echo "=== ${language} (${lang_code}) -> ${repo_name} ==="

  echo "[1/5] Pre-finetune English eval"
  python "${ENGLISH_EVAL_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "Qwen/Qwen3-8B",
  "model_id": "${preft_model_id}",
  "sorry_bench_dir": "external/sorry-bench",
  "english_questions_jsonl": "data/inputs/eval_prompts/sorry-bench-questions.jsonl",
  "output_dir": "data/outputs/eval_english"
}
JSON
)

  echo "[2/5] Pre-finetune translated eval"
  python "${TRANSLATED_EVAL_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "Qwen/Qwen3-8B",
  "model_id": "${preft_model_id}",
  "language_code": "${lang_code}",
  "source_lang_code": "${source_lang_code}",
  "local_prompt_csv": "data/inputs/eval_prompts/MLSFT - ${language} Evaluation Prompts  - Sheet1.csv",
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

  echo "[3/5] Fine-tuning"
  python "${FINETUNE_SCRIPT}" --config <(cat <<JSON
{
  "language": "${language}",
  "model_name": "Qwen/Qwen3-8B",
  "model_id": "${repo_name}",
  "random_seed": 3407,
  "template_name": "chatml",
  "input_csv": "data/inputs/fine_tuning/MLS - Fine-Tuning Data - ${language} - Sheet1.csv",
  "output_dir": "data/outputs/fine_tuning/${repo_name}",
  "local_model_dir": "data/outputs/fine_tuning/${repo_name}/model_merged",
  "summary_json": "data/outputs/fine_tuning/${repo_name}/train_summary.json",
  "num_train_epochs": 1,
  "learning_rate": 5e-5,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "push_to_hub": true,
  "allow_existing_hf_repo": false,
  "hf_repo": "${repo_name}"
}
JSON
)

  echo "[4/5] Post-finetune English eval (HF model)"
  python "${ENGLISH_EVAL_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${repo_name}",
  "model_id": "${repo_name}",
  "sorry_bench_dir": "external/sorry-bench",
  "english_questions_jsonl": "data/inputs/eval_prompts/sorry-bench-questions.jsonl",
  "output_dir": "data/outputs/eval_english"
}
JSON
)

  echo "[5/5] Post-finetune translated eval (HF model)"
  python "${TRANSLATED_EVAL_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${repo_name}",
  "model_id": "${repo_name}",
  "language_code": "${lang_code}",
  "source_lang_code": "${source_lang_code}",
  "local_prompt_csv": "data/inputs/eval_prompts/MLSFT - ${language} Evaluation Prompts  - Sheet1.csv",
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

for run in "${LANGUAGE_RUNS[@]}"; do
  IFS='|' read -r language lang_code source_lang_code <<< "${run}"
  run_language_pipeline "${language}" "${lang_code}" "${source_lang_code}"
done

echo ""
echo ""
echo ""
echo "Direct multilingual pipeline complete."
echo "Each language should now have pre/post English and pre/post translated CSVs plus JSONL traces in data/outputs/eval_english and data/outputs/eval_translated."
