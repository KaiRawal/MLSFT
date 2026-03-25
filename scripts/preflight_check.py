"""
Pre-flight checks for MLSFT pipeline.

This script validates:
1) Required config files are present and parseable.
2) Required CSV/JSONL input files exist.
3) Step Zero dependencies are initialized (repos/models/runtime assets).
4) Required Python packages are importable.

Usage:
	python scripts/preflight_check.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _exists(path: Path) -> bool:
	return path.exists()


def _load_json(path: Path) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _check_pkg(name: str) -> bool:
	return importlib.util.find_spec(name) is not None


def _format_fix(command: str) -> str:
	return f"How to fix: {command}"


def main() -> int:
	errors: list[str] = []

	config_paths = {
		"fine_tuning": ROOT / "configs" / "mls_fine_tuning_with_templates.json",
		"eval_english": ROOT / "configs" / "mls_eval_english.json",
		"eval_translated": ROOT
		/ "configs"
		/ "nllb_200_mls_run_sorry_bench_with_translated_prompts.json",
	}

	configs: dict[str, dict[str, Any]] = {}
	for key, config_path in config_paths.items():
		if not _exists(config_path):
			errors.append(
				f"Missing config: {config_path}\n"
				+ _format_fix(
					f"Create the config file and follow docs/step_zero_initialization.md"
				)
			)
			continue
		try:
			configs[key] = _load_json(config_path)
		except Exception as exc:
			errors.append(
				f"Invalid JSON in config: {config_path}\n"
				+ _format_fix(f"Fix JSON syntax. Parser error: {exc}")
			)

	# Required data inputs from configs.
	if "fine_tuning" in configs:
		ft_csv = ROOT / configs["fine_tuning"].get("input_csv", "")
		if not _exists(ft_csv):
			errors.append(
				f"Missing required CSV for fine-tuning: {ft_csv}\n"
				+ _format_fix(
					"Download the matching Google Sheet as CSV into data/inputs/fine_tuning"
				)
			)

	if "eval_english" in configs:
		en_jsonl = ROOT / configs["eval_english"].get("english_questions_jsonl", "")
		if not _exists(en_jsonl):
			errors.append(
				f"Missing required JSONL for English eval: {en_jsonl}\n"
				+ _format_fix(
					"Place sorry-bench-questions.jsonl under data/inputs/eval_prompts"
				)
			)

	if "eval_translated" in configs:
		local_csv = ROOT / configs["eval_translated"].get("local_prompt_csv", "")
		en_jsonl_2 = ROOT / configs["eval_translated"].get("english_prompt_jsonl", "")
		nllb_ct2_dir = ROOT / configs["eval_translated"].get("nllb_ct2_dir", "")

		if not _exists(local_csv):
			errors.append(
				f"Missing required local-language prompt CSV: {local_csv}\n"
				+ _format_fix(
					"Download the matching MLSFT evaluation sheet as CSV into data/inputs/eval_prompts"
				)
			)

		if not _exists(en_jsonl_2):
			errors.append(
				f"Missing required English prompt JSONL: {en_jsonl_2}\n"
				+ _format_fix(
					"Place sorry-bench-questions.jsonl under data/inputs/eval_prompts"
				)
			)

		if not _exists(nllb_ct2_dir):
			errors.append(
				f"Missing NLLB CTranslate2 directory: {nllb_ct2_dir}\n"
				+ _format_fix(
					"Initialize Step Zero #5 and create external/nllb-3.3b-ct2-int8"
				)
			)
		else:
			nllb_model_bin = nllb_ct2_dir / "model.bin"
			if not _exists(nllb_model_bin):
				errors.append(
					f"NLLB directory exists but model.bin is missing: {nllb_model_bin}\n"
					+ _format_fix(
						"Re-run your one-time ct2 conversion to populate model files"
					)
				)

	# Step Zero repository dependencies.
	sorry_bench_dir = ROOT / "external" / "sorry-bench"
	fastchat_dir = ROOT / "external" / "FastChat"

	if not _exists(sorry_bench_dir):
		errors.append(
			f"Missing repository: {sorry_bench_dir}\n"
			+ _format_fix(
				"git clone https://github.com/sorry-bench/sorry-bench.git external/sorry-bench"
			)
		)
	else:
		required_scripts = [
			sorry_bench_dir / "gen_model_answer_vllm.py",
			sorry_bench_dir / "gen_judgment_safety_vllm.py",
		]
		for required_script in required_scripts:
			if not _exists(required_script):
				errors.append(
					f"Missing required sorry-bench script: {required_script}\n"
					+ _format_fix(
						"Re-clone or repair external/sorry-bench to include required scripts"
					)
				)

		autorater_dir = (
			sorry_bench_dir
			/ "ckpts"
			/ "finetuned_models"
			/ "ft-mistral-7b-instruct-v0.2-sorry-bench-202406"
		)
		if not _exists(autorater_dir):
			errors.append(
				f"Missing autorater model directory: {autorater_dir}\n"
				+ _format_fix(
					"Initialize Step Zero #4 and download sorry-bench autorater weights"
				)
			)
		else:
			has_config = _exists(autorater_dir / "config.json")
			has_weight = any(autorater_dir.glob("*.safetensors")) or _exists(
				autorater_dir / "pytorch_model.bin"
			)
			if not has_config or not has_weight:
				errors.append(
					"Autorater directory exists but appears incomplete\n"
					+ _format_fix(
						"Ensure config.json and model weights are present in the autorater directory"
					)
				)

	if not _exists(fastchat_dir):
		errors.append(
			f"Missing repository: {fastchat_dir}\n"
			+ _format_fix(
				"git clone https://github.com/lm-sys/FastChat.git external/FastChat"
			)
		)

	# Python package checks for runtime dependencies.
	required_packages = [
		"pandas",
		"torch",
		"transformers",
		"datasets",
		"trl",
		"unsloth",
		"ctranslate2",
	]
	missing_packages = [pkg for pkg in required_packages if not _check_pkg(pkg)]
	if missing_packages:
		package_list = " ".join(missing_packages)
		errors.append(
			"Missing Python packages: " + ", ".join(missing_packages) + "\n"
			+ _format_fix(f"Install in your active environment, e.g. pip install {package_list}")
		)

	if errors:
		print("\n[PRE-FLIGHT] FAILED\n")
		for idx, err in enumerate(errors, start=1):
			print(f"{idx}. {err}\n")
		print("See docs/step_zero_initialization.md for full setup instructions.")
		return 1

	print("[PRE-FLIGHT] OK: All required files and Step Zero dependencies are present.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
