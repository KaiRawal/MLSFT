"""
MLS eval in English using local sorry-bench assets.

Purpose:
    Run generation and autorating on English sorry-bench prompts, then export CSV results.

Inputs:
    - Config JSON (via --config)
    - Local sorry-bench repo already initialized (manual setup step)
    - English prompts JSONL file

Outputs:
    - Model answer JSONL copy under data/outputs
    - Judgment JSONL copy under data/outputs
    - Final merged CSV with Prompt, Output, Rating

Usage:
    python scripts/mls_eval_english.py --config configs/mls_eval_english.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(command: str, cwd: Path) -> None:
    subprocess.run(command, shell=True, cwd=cwd, check=True)


def detect_accelerator() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(accelerator: str) -> str:
    if accelerator == "mps":
        return "float16"
    if accelerator == "cuda":
        return "bfloat16"
    return "float32"


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def extract_output(choices: Any) -> str:
    try:
        return choices[0]["turns"][0]
    except Exception:
        return ""


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def resolve_model_path(config: dict[str, Any]) -> str:
    configured = str(config.get("model_path", "")).strip()
    hf_user = (os.environ.get("HF_USER") or "").strip()

    if configured:
        if Path(configured).exists() or configured.startswith(("/", "./", "../", "data/", "external/")):
            return configured
        if "/" in configured:
            return configured
        if not hf_user:
            raise ValueError("HF_USER env var is required when model_path is not namespaced")
        return f"{hf_user}/{configured}"

    model_id = str(config.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("config requires model_id when model_path is empty")
    if not hf_user:
        raise ValueError("HF_USER env var is required when resolving model path from model_id")
    return f"{hf_user}/{model_id}"


def prepare_questions_file(config: dict[str, Any], sorry_bench_dir: Path) -> None:
    source = Path(config["english_questions_jsonl"])
    destination = sorry_bench_dir / "data" / "sorry_bench" / "question.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination)


def remove_thinking_tokens(jsonl_path: Path) -> Path:
    """Remove <think>...</think> tags from JSONL file in place.
    
    Creates a backup first, then modifies the original file.
    Removes all thinking tokens from the 'turns' field in each JSON object.
    
    Returns:
        Path to the backup file.
    """
    # Create backup
    backup_path = jsonl_path.with_stem(f"{jsonl_path.stem}_backup")
    shutil.copy(jsonl_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read, process, and write back
    processed_lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "choices" in data and isinstance(data["choices"], list):
                for choice in data["choices"]:
                    if "turns" in choice and isinstance(choice["turns"], list):
                        choice["turns"] = [
                            re.sub(r'<think>.*?</think>', '', turn, flags=re.DOTALL).strip()
                            for turn in choice["turns"]
                        ]
            processed_lines.append(json.dumps(data))
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in processed_lines:
            f.write(line + "\n")
    
    print(f"Removed thinking tokens from: {jsonl_path}")
    return backup_path


def run_eval(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    sorry_bench_dir = Path(config["sorry_bench_dir"])
    ensure_exists(sorry_bench_dir, "sorry_bench_dir")
    accelerator = detect_accelerator()
    generation_dtype = resolve_dtype(accelerator)
    model_path = resolve_model_path(config)

    prepare_questions_file(config, sorry_bench_dir)

    run_cmd(
        "python gen_model_answer_vllm.py "
        f"--bench-name sorry_bench --model-path {model_path} --model-id {config['model_id']} --dtype {generation_dtype}",
        cwd=sorry_bench_dir,
    )
    model_answer = sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{config['model_id']}.jsonl"
    ensure_exists(model_answer, "model answer file")

    # Remove thinking tokens before judgment generation
    backup_model_answer = remove_thinking_tokens(model_answer)

    run_cmd(
        f"python gen_judgment_safety_vllm.py --model-list {config['model_id']}",
        cwd=sorry_bench_dir,
    )
    model_judgment = sorry_bench_dir / "data" / "sorry_bench" / "model_judgment" / "ft-mistral-7b-instruct-v0.2.jsonl"
    ensure_exists(model_judgment, "model judgment file")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_answer = output_dir / f"{config['model_id']}_model_answer.jsonl"
    copied_backup = output_dir / f"{config['model_id']}_model_answer_backup.jsonl"
    copied_judgment = output_dir / f"{config['model_id']}_model_judgment.jsonl"
    merged_csv = output_dir / f"{config['model_id']}_english_results.csv"

    shutil.copy(model_answer, copied_answer)
    shutil.copy(backup_model_answer, copied_backup)
    shutil.copy(model_judgment, copied_judgment)

    questions_df = read_jsonl(Path(config["english_questions_jsonl"]))
    answers_df = read_jsonl(copied_answer)
    judgments_df = read_jsonl(copied_judgment)

    answers_df["Output"] = answers_df["choices"].apply(extract_output)
    if "score" in judgments_df.columns:
        judgments_df = judgments_df.rename(columns={"score": "Rating"})
    elif "judgment" in judgments_df.columns:
        judgments_df = judgments_df.rename(columns={"judgment": "Rating"})

    questions_df = questions_df[["question_id", "category", "turns"]].rename(columns={"turns": "Prompt"})
    answers_df = answers_df[["question_id", "Output"]]
    judgments_df = judgments_df[["question_id", "Rating"]]

    merged_df = questions_df.merge(answers_df, on="question_id", how="inner")
    merged_df = merged_df.merge(judgments_df, on="question_id", how="inner")
    merged_df["Prompt"] = merged_df["Prompt"].apply(lambda x: x[0] if isinstance(x, list) and x else x)
    merged_df.to_csv(merged_csv, index=False)

    return copied_answer, copied_backup, copied_judgment, merged_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run English sorry-bench eval")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    answer_file, backup_file, judgment_file, result_file = run_eval(config)

    print(f"Saved model answers to: {answer_file}")
    print(f"Saved model answers backup to: {backup_file}")
    print(f"Saved model judgments to: {judgment_file}")
    print(f"Saved merged results to: {result_file}")


if __name__ == "__main__":
    main()
