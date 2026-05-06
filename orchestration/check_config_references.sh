#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

configs=(
	"configs/mls_fine_tuning_with_templates.json"
	"configs/mls_eval_english.json"
	"configs/nllb_200_mls_run_sorry_bench_with_translated_prompts.json"
)

unused=()

for config in "${configs[@]}"; do
	if ! rg -n --fixed-strings --glob '!data/**' --glob '!external/**' --glob '!node_modules/**' --glob '!**/.git/**' "${config}" . >/dev/null; then
		unused+=("${config}")
	fi
done

if [[ ${#unused[@]} -eq 0 ]]; then
	echo "No unused config files found."
	exit 0
fi

echo "Unused config files:"
for config in "${unused[@]}"; do
	echo "- ${config}"
done

exit 1