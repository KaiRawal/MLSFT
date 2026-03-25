"""
Super-controller for MLSFT experiment orchestration.

Given a base model name, this script generates cached config files for:
1) Fine-tuning for all selected languages.
2) English eval for all fine-tuned models.
3) Local-language translated eval for all fine-tuned models.

It stores generated configs, intermediate artifacts, and final outputs under a
run-scoped cache directory.

Usage:
    python scripts/super_controller.py --model-name unsloth/qwen3-0.6B
    python scripts/super_controller.py --model-name unsloth/qwen3-0.6B --run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_BASE = ROOT / "data" / "cache" / "super_controller"


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    short_code: str
    nllb_source_lang: str


LANGUAGE_SPECS: tuple[LanguageSpec, ...] = (
    LanguageSpec("Chinese", "ZH", "zho_Hans"),
    LanguageSpec("Danish", "DA", "dan_Latn"),
    LanguageSpec("English", "EN", "eng_Latn"),
    LanguageSpec("Greek", "EL", "ell_Grek"),
    LanguageSpec("Hindi", "HI", "hin_Deva"),
    LanguageSpec("Irish", "GA", "gle_Latn"),
    LanguageSpec("Portuguese", "PT", "por_Latn"),
    LanguageSpec("Spanish", "ES", "spa_Latn"),
    LanguageSpec("Tagalog", "TL", "tgl_Latn"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and optionally run MLSFT full-matrix configs")
    parser.add_argument("--model-name", required=True, help="Base model name, e.g. unsloth/qwen3-0.6B")
    parser.add_argument("--hf-user", default="", help="Optional HF user/org for push_to_hub mode")
    parser.add_argument("--hf-token", default="", help="Optional HF token for push_to_hub mode")
    parser.add_argument("--push-to-hub", action="store_true", help="If set, generated fine-tuning configs enable HF push")
    parser.add_argument("--template", default="chatml", help="Template name for fine-tuning")
    parser.add_argument("--seed-indices", default="1", help="Comma-separated seed indices, e.g. 1 or 1,2,3")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for fine-tuning")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--languages", default="all", help="Comma-separated languages or 'all'")
    parser.add_argument("--cache-base", default=str(DEFAULT_CACHE_BASE), help="Base cache directory")
    parser.add_argument("--run", action="store_true", help="Run generated configs immediately")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set with --run, continue execution after a failed job",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def model_id(model_name: str, language_short_code: str, seed_index: int, epochs: int) -> str:
    epoch_letter = "A" if epochs == 1 else "B"
    model_leaf = model_name.split("/")[-1]
    return f"{model_leaf}-{language_short_code}-SynthDolly-{seed_index}{epoch_letter}"


def seed_value(seed_index: int) -> int:
    seeds = {1: 3407, 2: 9, 3: 73}
    if seed_index not in seeds:
        raise ValueError(f"Unsupported seed index {seed_index}. Use one of 1,2,3")
    return seeds[seed_index]


def selected_languages(raw: str) -> list[LanguageSpec]:
    if raw.strip().lower() == "all":
        return list(LANGUAGE_SPECS)

    wanted = {item.strip().lower() for item in raw.split(",") if item.strip()}
    selected = [spec for spec in LANGUAGE_SPECS if spec.name.lower() in wanted]
    if not selected:
        valid = ", ".join(spec.name for spec in LANGUAGE_SPECS)
        raise ValueError(f"No valid languages selected. Use one or more of: {valid}")
    return selected


def run_cmd(command: list[str], log_path: Path) -> int:
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> int:
    args = parse_args()

    if args.push_to_hub and not args.hf_user:
        raise ValueError("--push-to-hub requires --hf-user")

    seeds = [int(token.strip()) for token in args.seed_indices.split(",") if token.strip()]
    for seed_idx in seeds:
        _ = seed_value(seed_idx)

    langs = selected_languages(args.languages)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model_name.split('/')[-1]}_{timestamp}"
    cache_base = Path(args.cache_base)
    run_root = cache_base / run_id

    config_root = run_root / "configs"
    artifacts_root = run_root / "artifacts"
    logs_root = run_root / "logs"
    results_root = run_root / "results"

    ensure_dir(config_root)
    ensure_dir(artifacts_root)
    ensure_dir(logs_root)
    ensure_dir(results_root)

    generated: list[dict[str, Any]] = []

    for lang in langs:
        for seed_idx in seeds:
            current_model_id = model_id(
                model_name=args.model_name,
                language_short_code=lang.short_code,
                seed_index=seed_idx,
                epochs=args.epochs,
            )

            ft_cfg = {
                "language": lang.name,
                "model_name": args.model_name,
                "model_id": current_model_id,
                "random_seed": seed_value(seed_idx),
                "template_name": args.template,
                "input_csv": f"data/inputs/fine_tuning/MLS - Fine-Tuning Data - {lang.name} - Sheet1.csv",
                "output_dir": str(artifacts_root / "fine_tuning" / current_model_id),
                "local_model_dir": str(artifacts_root / "fine_tuning" / current_model_id / "model_merged"),
                "summary_json": str(results_root / "fine_tuning" / f"{current_model_id}_train_summary.json"),
                "num_train_epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "push_to_hub": bool(args.push_to_hub),
                "hf_repo": f"{args.hf_user}/{current_model_id}" if args.hf_user else "",
                "hf_token": args.hf_token,
            }
            ft_cfg_path = config_root / "fine_tuning" / f"{current_model_id}.json"
            write_json(ft_cfg_path, ft_cfg)
            generated.append(
                {
                    "stage": "fine_tuning",
                    "language": lang.name,
                    "seed_index": seed_idx,
                    "model_id": current_model_id,
                    "config_path": str(ft_cfg_path),
                }
            )

            en_cfg = {
                "model_path": ft_cfg["local_model_dir"],
                "model_id": current_model_id,
                "sorry_bench_dir": "external/sorry-bench",
                "english_questions_jsonl": "data/inputs/eval_prompts/sorry-bench-questions.jsonl",
                "output_dir": str(artifacts_root / "eval_english" / current_model_id),
            }
            en_cfg_path = config_root / "eval_english" / f"{current_model_id}.json"
            write_json(en_cfg_path, en_cfg)
            generated.append(
                {
                    "stage": "eval_english",
                    "language": lang.name,
                    "seed_index": seed_idx,
                    "model_id": current_model_id,
                    "config_path": str(en_cfg_path),
                }
            )

            tr_cfg = {
                "model_path": ft_cfg["local_model_dir"],
                "model_id": current_model_id,
                "language_code": lang.short_code,
                "source_lang_code": lang.nllb_source_lang,
                "local_prompt_csv": f"data/inputs/eval_prompts/MLSFT - {lang.name} Evaluation Prompts  - Sheet1.csv",
                "english_prompt_jsonl": "data/inputs/eval_prompts/sorry-bench-questions.jsonl",
                "sorry_bench_dir": "external/sorry-bench",
                "nllb_ct2_dir": "external/nllb-3.3b-ct2-int8",
                "nllb_model_name": "facebook/nllb-200-3.3B",
                "translate_device": "auto",
                "translation_batch_size": 32,
                "output_dir": str(artifacts_root / "eval_translated" / current_model_id / lang.name),
            }
            tr_cfg_path = config_root / "eval_translated" / f"{current_model_id}_{lang.name}.json"
            write_json(tr_cfg_path, tr_cfg)
            generated.append(
                {
                    "stage": "eval_translated",
                    "language": lang.name,
                    "seed_index": seed_idx,
                    "model_id": current_model_id,
                    "config_path": str(tr_cfg_path),
                }
            )

    manifest = {
        "run_id": run_id,
        "generated_at": timestamp,
        "model_name": args.model_name,
        "hf_user": args.hf_user,
        "push_to_hub": bool(args.push_to_hub),
        "template": args.template,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seeds": seeds,
        "languages": [lang.name for lang in langs],
        "cache_paths": {
            "run_root": str(run_root),
            "config_root": str(config_root),
            "artifacts_root": str(artifacts_root),
            "logs_root": str(logs_root),
            "results_root": str(results_root),
        },
        "jobs": generated,
    }
    manifest_path = run_root / "manifest.json"
    write_json(manifest_path, manifest)

    print(f"Generated {len(generated)} configs for run {run_id}")
    print(f"Config cache: {config_root}")
    print(f"Artifact cache: {artifacts_root}")
    print(f"Manifest: {manifest_path}")

    if not args.run:
        return 0

    execution: list[dict[str, Any]] = []

    preflight_log = logs_root / "preflight.log"
    preflight_code = run_cmd(["python", "scripts/preflight_check.py"], preflight_log)
    execution.append(
        {
            "stage": "preflight",
            "return_code": preflight_code,
            "log": str(preflight_log),
        }
    )
    if preflight_code != 0:
        write_json(run_root / "execution_summary.json", {"run_id": run_id, "execution": execution})
        print("Preflight failed. See log:", preflight_log)
        return preflight_code

    stage_order = ["fine_tuning", "eval_english", "eval_translated"]
    stage_script = {
        "fine_tuning": "scripts/mls_fine_tuning_with_templates.py",
        "eval_english": "scripts/mls_eval_english.py",
        "eval_translated": "scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py",
    }

    for stage in stage_order:
        stage_jobs = [job for job in generated if job["stage"] == stage]
        for job in stage_jobs:
            cfg_path = Path(job["config_path"])
            log_path = logs_root / stage / f"{cfg_path.stem}.log"
            cmd = ["python", stage_script[stage], "--config", str(cfg_path)]
            code = run_cmd(cmd, log_path)
            execution.append(
                {
                    "stage": stage,
                    "model_id": job["model_id"],
                    "language": job["language"],
                    "seed_index": job["seed_index"],
                    "return_code": code,
                    "log": str(log_path),
                    "config_path": str(cfg_path),
                }
            )

            if code != 0 and not args.continue_on_error:
                write_json(run_root / "execution_summary.json", {"run_id": run_id, "execution": execution})
                print(f"Execution stopped at stage={stage}, model_id={job['model_id']}")
                print(f"See log: {log_path}")
                return code

    write_json(run_root / "execution_summary.json", {"run_id": run_id, "execution": execution})
    print("Execution complete. Summary written to", run_root / "execution_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
