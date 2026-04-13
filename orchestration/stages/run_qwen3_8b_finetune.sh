#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_SCRIPT="${ROOT_DIR}/src/mls_fine_tuning_with_templates.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Error: could not find finetuning script at ${PYTHON_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${HF_USER:-}" ]]; then
  echo "Error: HF_USER must be set (target uploads use HF_USER/<model_name_with_finetune_details>)." >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN must be set for Hugging Face uploads." >&2
  exit 1
fi

echo "Current working directory: $(pwd)"
echo "Fine-tuning entrypoint: ${PYTHON_SCRIPT}"

run_finetune() {
  local language="$1"
  local lang_code="$2"
  local repo_name="Qwen3-8B-${lang_code}-SynthDolly-1A"

  echo "Starting fine-tuning for ${language} -> ${repo_name}"

  python "${PYTHON_SCRIPT}" --config <(cat <<JSON
{
  "language": "${language}",
  "model_name": "Qwen/Qwen3-8B",
  "model_id": "${repo_name}",
  "random_seed": 3407,
  "template_name": "chatml",
  "input_csv": "data/inputs/fine_tuning/${language}_finetuning_data.csv",
  "output_dir": "data/outputs/fine_tuning/${repo_name}",
  "local_model_dir": "data/outputs/fine_tuning/${repo_name}/model_merged",
  "summary_json": "data/outputs/fine_tuning/${repo_name}/train_summary.json",
  "num_train_epochs": 1,
  "learning_rate": 5e-5,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "push_to_hub": true,
  "hf_repo": "${repo_name}"
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
echo "Running ${#LANGUAGE_RUNS[@]} fine-tuning calls for Qwen/Qwen3-8B"
echo ""
echo ""
echo ""

for run in "${LANGUAGE_RUNS[@]}"; do
  IFS='|' read -r language lang_code <<< "${run}"
  run_finetune "${language}" "${lang_code}"
done

echo ""
echo ""
echo ""
echo "Completed ${#LANGUAGE_RUNS[@]} fine-tuning calls."
echo ""
echo ""
echo ""
