#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  orchestration/run_unified_pipeline.sh <model_base> <num_epochs> <seed>

Supported model_base values:
  unsloth/gemma-3-1b-it
  unsloth/llama-3.2-1b-Instruct
  unsloth/qwen3-0.6b
  unsloth/gemma-3-4b-it
  unsloth/llama-3.2-3b-Instruct
  unsloth/qwen3-4b
  unsloth/Meta-Llama-3.1-8B-Instruct
  unsloth/Qwen3-8B
  unsloth/Qwen3-14B
  unsloth/Qwen3-32B

Example:
  orchestration/run_unified_pipeline.sh unsloth/qwen3-4b 3 73
  orchestration/run_unified_pipeline.sh unsloth/Qwen3-8B 1 3407
  orchestration/run_unified_pipeline.sh unsloth/Meta-Llama-3.1-8B-Instruct 5 9

Notes:
  - num_epochs must be a positive integer
  - seed must be a positive integer
  - Model hyperparameters (batch size, template, etc.) are auto-detected
  - Environment variables required: HF_USER, HF_TOKEN
EOF
}

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

SELECTED_MODEL="$1"
NUM_TRAIN_EPOCHS="$2"
SELECTED_SEED="$3"

if ! [[ "${NUM_TRAIN_EPOCHS}" =~ ^[0-9]+$ ]]; then
  echo "Error: num_epochs must be a positive integer." >&2
  exit 1
fi

if [[ "${NUM_TRAIN_EPOCHS}" -le 0 ]]; then
  echo "Error: num_epochs must be greater than 0." >&2
  exit 1
fi

if ! [[ "${SELECTED_SEED}" =~ ^[0-9]+$ ]]; then
  echo "Error: seed must be a positive integer." >&2
  exit 1
fi

if [[ "${SELECTED_SEED}" -le 0 ]]; then
  echo "Error: seed must be greater than 0." >&2
  exit 1
fi

EPOCH_TAG="E${NUM_TRAIN_EPOCHS}"
SEED_TAG="S${SELECTED_SEED}"

# Function to get model configuration
get_model_config() {
  local model="$1"
  # Returns: BASE_MODEL_PATH TEMPLATE_NAME MODEL_PREFIX BATCH_SIZE GRAD_ACCUM_STEPS
  case "${model}" in
    "unsloth/gemma-3-1b-it")
      echo "unsloth/gemma-3-1b-it gemma-3 Gemma-3-1B-IT 4 2"
      ;;
    "unsloth/llama-3.2-1b-Instruct")
      echo "unsloth/llama-3.2-1b-Instruct llama-3.2 Llama-3.2-1B-Instruct 4 2"
      ;;
    "unsloth/qwen3-0.6b")
      echo "unsloth/qwen3-0.6b chatml Qwen3-0.6B 4 2"
      ;;
    "unsloth/gemma-3-4b-it")
      echo "unsloth/gemma-3-4b-it gemma-3 Gemma-3-4B-IT 4 2"
      ;;
    "unsloth/llama-3.2-3b-Instruct")
      echo "unsloth/llama-3.2-3b-Instruct llama-3.2 Llama-3.2-3B-Instruct 4 2"
      ;;
    "unsloth/qwen3-4b")
      echo "unsloth/qwen3-4b chatml Qwen3-4B 4 2"
      ;;
    "unsloth/Meta-Llama-3.1-8B-Instruct")
      echo "unsloth/Meta-Llama-3.1-8B-Instruct llama-3.1 Llama-3.1-8B-Instruct 2 4"
      ;;
    "unsloth/Qwen3-8B")
      echo "Qwen/Qwen3-8B chatml Qwen3-8B 2 4"
      ;;
    "unsloth/Qwen3-14B")
      echo "unsloth/Qwen3-14B chatml Qwen3-14B 2 4"
      ;;
    "unsloth/Qwen3-32B")
      echo "unsloth/Qwen3-32B chatml Qwen3-32B 1 8"
      ;;
    *)
      echo "Error: unsupported model '${model}'." >&2
      usage
      exit 1
      ;;
  esac
}

# Read model config
read -r BASE_MODEL_PATH TEMPLATE_NAME MODEL_PREFIX BATCH_SIZE GRAD_ACCUM <<< "$(get_model_config "${SELECTED_MODEL}")"

# Setup directories and logging
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
SYSTEM_ID="$(uname -s)_$(uname -m)_$(hostname -s 2>/dev/null || echo unknown_host)"
SYSTEM_ID="$(echo "${SYSTEM_ID}" | tr ' /:' '___')"
RUN_ID="${SYSTEM_ID}_${RUN_TIMESTAMP}"
SUMMARY_DIR="data/run_summaries"
LOG_DIR="${SUMMARY_DIR}/${RUN_ID}_logs"
SUMMARY_FILE="${SUMMARY_DIR}/${RUN_ID}.txt"

mkdir -p "${SUMMARY_DIR}" "${LOG_DIR}"

CSV_BEFORE_FILE="$(mktemp)"
CSV_AFTER_FILE="$(mktemp)"
CSV_NEW_FILE="$(mktemp)"

cleanup_tmp_files() {
  rm -f "${CSV_BEFORE_FILE}" "${CSV_AFTER_FILE}" "${CSV_NEW_FILE}"
}

trap cleanup_tmp_files EXIT

find data/outputs -type f -name "*.csv" 2>/dev/null | sort > "${CSV_BEFORE_FILE}" || true

declare -a STEP_RESULTS=()
declare -a LANGUAGE_RESULTS=()
declare -a HARD_FAILURES=()

record_step_result() {
  local language="$1"
  local step="$2"
  local status="$3"
  local detail="$4"
  STEP_RESULTS+=("${language} | ${step} | ${status} | ${detail}")
}

record_language_result() {
  local language="$1"
  local status="$2"
  local detail="$3"
  LANGUAGE_RESULTS+=("${language} | ${status} | ${detail}")
}

record_hard_failure() {
  local language="$1"
  local step="$2"
  local detail="$3"
  HARD_FAILURES+=("${language} | ${step} | ${detail}")
}

run_step() {
  local language="$1"
  local step_label="$2"
  local description="$3"
  local step_log="$4"
  shift 4

  echo "${language} ${step_label} ${description}"
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] START ${language} ${step_label} ${description}" >> "${step_log}"

  set +e
  "$@" > >(tee -a "${step_log}") 2> >(tee -a "${step_log}" >&2)
  local exit_code=$?
  set -e

  echo "[$(date +"%Y-%m-%d %H:%M:%S")] END exit_code=${exit_code}" >> "${step_log}"
  return "${exit_code}"
}

print_and_save_summary() {
  local overall_exit_code="$1"
  local border="================================================================================"

  find data/outputs -type f -name "*.csv" 2>/dev/null | sort > "${CSV_AFTER_FILE}" || true
  comm -13 "${CSV_BEFORE_FILE}" "${CSV_AFTER_FILE}" > "${CSV_NEW_FILE}" || true

  {
    echo ""
    echo ""
    echo ""
    echo ""
    echo "${border}"
    echo "UNIFIED MULTILINGUAL PIPELINE SUMMARY"
    echo "${border}"
    echo "Run ID: ${RUN_ID}"
    echo "Generated at: $(date +"%Y-%m-%d %H:%M:%S")"
    echo "Summary file: ${SUMMARY_FILE}"
    echo "Detailed step logs: ${LOG_DIR}"
    echo "Model: ${BASE_MODEL_PATH}"
    echo "Train epochs: ${NUM_TRAIN_EPOCHS}"
    echo "Epoch tag: ${EPOCH_TAG}"
    echo "Random seed: ${SELECTED_SEED}"
    echo "Seed tag: ${SEED_TAG}"
    echo ""
    echo "Language Outcomes"
    echo "${border}"
    if [[ ${#LANGUAGE_RESULTS[@]} -eq 0 ]]; then
      echo "  - No language runs were recorded."
    else
      for line in "${LANGUAGE_RESULTS[@]}"; do
        echo "  - ${line}"
      done
    fi
    echo ""
    echo "Step Outcomes"
    echo "${border}"
    if [[ ${#STEP_RESULTS[@]} -eq 0 ]]; then
      echo "  - No step outcomes were recorded."
    else
      for line in "${STEP_RESULTS[@]}"; do
        echo "  - ${line}"
      done
    fi
    echo ""
    echo "Hard Failures"
    echo "${border}"
    if [[ ${#HARD_FAILURES[@]} -eq 0 ]]; then
      echo "  - None"
    else
      for line in "${HARD_FAILURES[@]}"; do
        echo "  - ${line}"
      done
    fi
    echo ""
    echo "Primary Output Directories"
    echo "${border}"
    echo "  - data/outputs/eval_english"
    echo "  - data/outputs/eval_translated"
    echo "  - data/outputs/fine_tuning"
    echo ""
    echo "New CSV Files Created During This Run"
    echo "${border}"
    if [[ -s "${CSV_NEW_FILE}" ]]; then
      while IFS= read -r csv_path; do
        echo "  - ${csv_path}"
      done < "${CSV_NEW_FILE}"
    else
      echo "  - No newly created CSV files were detected under data/outputs."
    fi
    echo ""
    if [[ "${overall_exit_code}" -eq 0 ]]; then
      echo "Final Status: SUCCESS (no hard failures)"
    else
      echo "Final Status: FAILURE (one or more hard failures occurred)"
    fi
    echo "${border}"
  } | tee "${SUMMARY_FILE}"
}

ENGLISH_EVAL_SCRIPT="src/mls_eval_english.py"
FINETUNE_SCRIPT="src/mls_fine_tuning_with_templates.py"
TRANSLATED_EVAL_SCRIPT="src/nllb_200_mls_run_sorry_bench_with_translated_prompts.py"

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
echo "Using base model: ${BASE_MODEL_PATH}"
echo "Using template: ${TEMPLATE_NAME}"
echo "Using train epochs: ${NUM_TRAIN_EPOCHS}"
echo "Using epoch tag: ${EPOCH_TAG}"
echo "Using random seed: ${SELECTED_SEED}"
echo "Using seed tag: ${SEED_TAG}"
echo "Pipeline per language: fine-tune -> post-English eval -> post-translated eval -> pre-English eval -> pre-translated eval"

run_language_pipeline() {
  local language="$1"
  local lang_code="$2"
  local source_lang_code="$3"

  local repo_name="${MODEL_PREFIX}-${lang_code}-SynthDolly-${EPOCH_TAG}-${SEED_TAG}"
  local preft_model_id="${repo_name}-PREFT"
  local full_repo_id="${HF_USER}/${repo_name}"
  local expected_existing_msg="Target Hugging Face repo already exists: ${full_repo_id}. Set allow_existing_hf_repo=true to permit re-training and overwriting."
  local step_log="${LOG_DIR}/${lang_code}_${language// /_}.log"

  local language_hard_failure=0
  local skipped_existing_model=0

  : > "${step_log}"

  echo ""
  echo ""
  echo ""
  echo "=== ${language} (${lang_code}) -> ${repo_name} ==="

  # Step 1: Fine-tuning
  if run_step "${language}" "[1/5]" "Fine-tuning" "${step_log}" python "${FINETUNE_SCRIPT}" --config <(cat <<JSON
{
  "language": "${language}",
  "model_name": "${BASE_MODEL_PATH}",
  "model_id": "${repo_name}",
  "random_seed": ${SELECTED_SEED},
  "template_name": "${TEMPLATE_NAME}",
  "epoch_tag": "${EPOCH_TAG}",
  "input_csv": "data/inputs/fine_tuning/${language}_finetuning_data.csv",
  "output_dir": "data/outputs/fine_tuning/${repo_name}",
  "local_model_dir": "data/outputs/fine_tuning/${repo_name}/model_merged",
  "summary_json": "data/outputs/fine_tuning/${repo_name}/train_summary.json",
  "num_train_epochs": ${NUM_TRAIN_EPOCHS},
  "learning_rate": 5e-5,
  "per_device_train_batch_size": ${BATCH_SIZE},
  "gradient_accumulation_steps": ${GRAD_ACCUM},
  "push_to_hub": true,
  "allow_existing_hf_repo": false,
  "hf_repo": "${repo_name}"
}
JSON
); then
    record_step_result "${language}" "[1/5] Fine-tuning" "SUCCESS" "Completed"
  else
    if grep -Fq "${expected_existing_msg}" "${step_log}"; then
      skipped_existing_model=1
      record_step_result "${language}" "[1/5] Fine-tuning" "SKIPPED_EXISTING_MODEL" "Existing HF model reused"
    else
      record_step_result "${language}" "[1/5] Fine-tuning" "FAILED" "See ${step_log}"
      record_hard_failure "${language}" "[1/5] Fine-tuning" "See ${step_log}"
      language_hard_failure=1
    fi
  fi

  if [[ "${skipped_existing_model}" -eq 1 ]]; then
    record_language_result "${language}" "SKIPPED_EXISTING_MODEL" "Fine-tune skipped because model already existed"
    return 0
  fi

  # Post-finetune: Organize the uploaded model into HF collections
  if [[ "${language_hard_failure}" -eq 0 ]]; then
    echo ""
    echo "Organizing uploaded model into HF collection..."
    if python src/organise.py \
      --repo-id "${repo_name}" \
      --epoch "${NUM_TRAIN_EPOCHS}" \
      --seed "${SELECTED_SEED}" \
      >> "${step_log}" 2>&1; then
      echo "Successfully organized model into collection."
    else
      echo "Warning: Failed to organize model into collection. Continuing with evaluation..."
      echo "repo_id: ${repo_name}, epoch: ${NUM_TRAIN_EPOCHS}, seed: ${SELECTED_SEED}"
      echo "See ${step_log} for details."
    fi
  fi

  # Step 2: Post-finetune English eval
  if [[ "${language_hard_failure}" -eq 0 ]]; then
    if run_step "${language}" "[2/5]" "Post-finetune English eval (HF model)" "${step_log}" python "${ENGLISH_EVAL_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${repo_name}",
  "model_id": "${repo_name}",
  "epoch_tag": "${EPOCH_TAG}",
  "sorry_bench_dir": "external/sorry-bench",
  "english_questions_jsonl": "data/inputs/eval_prompts/sorry-bench-questions.jsonl",
  "output_dir": "data/outputs/eval_english"
}
JSON
); then
      record_step_result "${language}" "[2/5] Post-finetune English eval (HF model)" "SUCCESS" "Completed"
    else
      record_step_result "${language}" "[2/5] Post-finetune English eval (HF model)" "FAILED" "See ${step_log}"
      record_hard_failure "${language}" "[2/5] Post-finetune English eval (HF model)" "See ${step_log}"
      language_hard_failure=1
    fi
  else
    record_step_result "${language}" "[2/5] Post-finetune English eval (HF model)" "SKIPPED" "Blocked by earlier failure"
  fi

  # Step 3: Post-finetune translated eval
  if [[ "${language_hard_failure}" -eq 0 ]]; then
    if run_step "${language}" "[3/5]" "Post-finetune translated eval (HF model)" "${step_log}" python "${TRANSLATED_EVAL_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${repo_name}",
  "model_id": "${repo_name}",
  "epoch_tag": "${EPOCH_TAG}",
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
); then
      record_step_result "${language}" "[3/5] Post-finetune translated eval (HF model)" "SUCCESS" "Completed"
    else
      record_step_result "${language}" "[3/5] Post-finetune translated eval (HF model)" "FAILED" "See ${step_log}"
      record_hard_failure "${language}" "[3/5] Post-finetune translated eval (HF model)" "See ${step_log}"
      language_hard_failure=1
    fi
  else
    record_step_result "${language}" "[3/5] Post-finetune translated eval (HF model)" "SKIPPED" "Blocked by earlier failure"
  fi

  # Step 4: Pre-finetune English eval
  if [[ "${language_hard_failure}" -eq 0 ]]; then
    if run_step "${language}" "[4/5]" "Pre-finetune English eval" "${step_log}" python "${ENGLISH_EVAL_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${BASE_MODEL_PATH}",
  "model_id": "${preft_model_id}",
  "epoch_tag": "${EPOCH_TAG}",
  "sorry_bench_dir": "external/sorry-bench",
  "english_questions_jsonl": "data/inputs/eval_prompts/sorry-bench-questions.jsonl",
  "output_dir": "data/outputs/eval_english"
}
JSON
); then
      record_step_result "${language}" "[4/5] Pre-finetune English eval" "SUCCESS" "Completed"
    else
      record_step_result "${language}" "[4/5] Pre-finetune English eval" "FAILED" "See ${step_log}"
      record_hard_failure "${language}" "[4/5] Pre-finetune English eval" "See ${step_log}"
      language_hard_failure=1
    fi
  else
    record_step_result "${language}" "[4/5] Pre-finetune English eval" "SKIPPED" "Blocked by earlier failure"
  fi

  # Step 5: Pre-finetune translated eval
  if [[ "${language_hard_failure}" -eq 0 ]]; then
    if run_step "${language}" "[5/5]" "Pre-finetune translated eval" "${step_log}" python "${TRANSLATED_EVAL_SCRIPT}" --config <(cat <<JSON
{
  "model_path": "${BASE_MODEL_PATH}",
  "model_id": "${preft_model_id}",
  "epoch_tag": "${EPOCH_TAG}",
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
); then
      record_step_result "${language}" "[5/5] Pre-finetune translated eval" "SUCCESS" "Completed"
    else
      record_step_result "${language}" "[5/5] Pre-finetune translated eval" "FAILED" "See ${step_log}"
      record_hard_failure "${language}" "[5/5] Pre-finetune translated eval" "See ${step_log}"
      language_hard_failure=1
    fi
  else
    record_step_result "${language}" "[5/5] Pre-finetune translated eval" "SKIPPED" "Blocked by earlier failure"
  fi

  if [[ "${language_hard_failure}" -eq 0 ]]; then
    record_language_result "${language}" "SUCCESS" "All steps completed"
    return 0
  fi

  record_language_result "${language}" "FAILED" "One or more hard failures; see ${step_log}"
  return 1
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

overall_exit_code=0

for run in "${LANGUAGE_RUNS[@]}"; do
  IFS='|' read -r language lang_code source_lang_code <<< "${run}"
  if ! run_language_pipeline "${language}" "${lang_code}" "${source_lang_code}"; then
    overall_exit_code=1
  fi
done

print_and_save_summary "${overall_exit_code}"

exit "${overall_exit_code}"
