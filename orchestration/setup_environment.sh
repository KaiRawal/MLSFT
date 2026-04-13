#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required in the environment."
  exit 1
fi

echo "[step-zero] Creating expected local folders"
mkdir -p external
mkdir -p data/inputs/fine_tuning
mkdir -p data/inputs/eval_prompts
mkdir -p data/outputs/fine_tuning
mkdir -p data/outputs/eval_english
mkdir -p data/outputs/eval_translated

echo "[step-zero] Cloning required repositories if missing"
if [[ ! -d external/sorry-bench/.git ]]; then
  git clone https://github.com/sorry-bench/sorry-bench.git external/sorry-bench
fi
if [[ ! -d external/FastChat/.git ]]; then
  git clone https://github.com/lm-sys/FastChat.git external/FastChat
fi

# echo "[step-zero] Installing Python dependencies"
# python -m pip install -r requirements.txt
# python -m pip install huggingface_hub

# Best effort install for third-party repo dependencies.
if [[ -f external/sorry-bench/requirements.txt ]]; then
  python -m pip install -r external/sorry-bench/requirements.txt
fi
if [[ -f external/FastChat/requirements.txt ]]; then
  python -m pip install -r external/FastChat/requirements.txt
fi

echo "[step-zero] Downloading autorater assets and preparing NLLB CTranslate2 model"
python src/setup_step_zero.py

echo "[step-zero] Running preflight checks"
python src/preflight_check.py

echo "[step-zero] Complete"
echo "Note: this script does not download any language CSVs; it assumes they are already in data/inputs."