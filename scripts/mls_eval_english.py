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
import shlex
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
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    return "float32"


def get_optional_config_str(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str or None


def resolve_generation_model_id(config: dict[str, Any]) -> str:
    return get_optional_config_str(config, "generation_model_id_for_template") or str(config["model_id"])


def resolve_generation_revision(config: dict[str, Any]) -> str | None:
    return get_optional_config_str(config, "generation_revision")


def resolve_generation_dtype(config: dict[str, Any], accelerator: str) -> str:
    configured = get_optional_config_str(config, "generation_dtype")
    if not configured:
        return resolve_dtype(accelerator)

    supported = {"float16", "bfloat16", "float32"}
    normalized = configured.lower()
    if normalized not in supported:
        raise ValueError(
            f"Unsupported generation_dtype={configured!r}. Use one of: {sorted(supported)}"
        )
    return normalized


def warn_unsupported_generation_tokenizer(config: dict[str, Any]) -> None:
    tokenizer_override = get_optional_config_str(config, "generation_tokenizer")
    if tokenizer_override:
        print(
            "Warning: generation_tokenizer is not supported by external/sorry-bench generation entrypoints; "
            "it will be ignored. Use model_path that already contains the intended tokenizer assets."
        )


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def extract_output(choices: Any) -> str:
    try:
        return choices[0]["turns"][0]
    except Exception:
        return ""


def strip_nested_think_tags(text: str) -> str:
    """Remove content wrapped in <think>...</think>, including nested blocks."""
    open_tag = "<think>"
    close_tag = "</think>"
    i = 0
    depth = 0
    out: list[str] = []

    while i < len(text):
        if text.startswith(open_tag, i):
            depth += 1
            i += len(open_tag)
            continue

        if text.startswith(close_tag, i):
            if depth > 0:
                depth -= 1
                i += len(close_tag)
                continue

        if depth == 0:
            out.append(text[i])
        i += 1

    return "".join(out).strip()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def overwrite_loud(path: Path, label: str) -> None:
    if path.exists():
        print(f"Overwriting existing {label}: {path}")
        path.unlink()


def copy_overwrite_loud(source: Path, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Overwriting existing {label}: {destination}")
    shutil.copy(source, destination)


def write_csv_overwrite_loud(df: pd.DataFrame, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Overwriting existing {label}: {destination}")
    df.to_csv(destination, index=False)


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


def resolve_generation_backend(config: dict[str, Any], model_path: str) -> str:
    configured = str(config.get("generation_backend", "")).strip().lower()
    if configured:
        if configured not in {"vllm", "fastchat"}:
            raise ValueError(
                f"Unsupported generation_backend={configured!r}. Use 'vllm' or 'fastchat'."
            )
        return configured

    model_path_lower = model_path.lower()
    model_id_lower = str(config.get("model_id", "")).lower()

    # vLLM + older transformers stacks may fail on ministral3 configs.
    if "ministral-3" in model_path_lower or "ministral-3" in model_id_lower:
        return "fastchat"

    return "vllm"


def prepare_questions_file(config: dict[str, Any], sorry_bench_dir: Path) -> None:
    source = Path(config["english_questions_jsonl"])
    destination = sorry_bench_dir / "data" / "sorry_bench" / "question.jsonl"
    copy_overwrite_loud(source, destination, "sorry-bench question file")


def remove_thinking_tokens(jsonl_path: Path) -> Path:
    """Remove <think>...</think> tags from JSONL file in place.
    
    Creates a backup first, then modifies the original file.
    Removes all thinking tokens from the 'turns' field in each JSON object.
    
    Returns:
        Path to the backup file.
    """
    # Create backup
    backup_path = jsonl_path.with_stem(f"{jsonl_path.stem}_backup")
    copy_overwrite_loud(jsonl_path, backup_path, "model answer backup before think-tag stripping")
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
                        choice["turns"] = [strip_nested_think_tags(turn) for turn in choice["turns"]]
            processed_lines.append(json.dumps(data))
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in processed_lines:
            f.write(line + "\n")
    
    print(f"Removed thinking tokens from: {jsonl_path}")
    return backup_path


def run_eval(config: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path]:
    sorry_bench_dir = Path(config["sorry_bench_dir"])
    ensure_exists(sorry_bench_dir, "sorry_bench_dir")
    accelerator = detect_accelerator()
    generation_dtype = resolve_generation_dtype(config, accelerator)
    model_path = resolve_model_path(config)
    generation_backend = resolve_generation_backend(config, model_path)
    generation_model_id = resolve_generation_model_id(config)
    generation_revision = resolve_generation_revision(config)

    warn_unsupported_generation_tokenizer(config)

    print(f"Using generation backend: {generation_backend}")
    if generation_model_id != str(config["model_id"]):
        print(
            "Using generation_model_id_for_template override: "
            f"{generation_model_id} (logical output model_id remains {config['model_id']})"
        )

    prepare_questions_file(config, sorry_bench_dir)

    model_answer = sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{config['model_id']}.jsonl"
    generated_model_answer = (
        sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{generation_model_id}.jsonl"
    )
    model_judgment = sorry_bench_dir / "data" / "sorry_bench" / "model_judgment" / "ft-mistral-7b-instruct-v0.2.jsonl"
    # Sorry-bench generation/judgment scripts append by default; clear per-run artifacts to force fresh eval.
    overwrite_loud(model_answer, "sorry-bench model answer cache")
    if generated_model_answer != model_answer:
        overwrite_loud(generated_model_answer, "sorry-bench model answer cache (template model id override)")
    overwrite_loud(model_judgment, "sorry-bench model judgment cache")

    if generation_backend == "vllm":
        cmd = [
            "python",
            "gen_model_answer_vllm.py",
            "--bench-name",
            "sorry_bench",
            "--model-path",
            model_path,
            "--model-id",
            generation_model_id,
            "--dtype",
            generation_dtype,
            "--num-gpus-per-model",
            "2",
            "--num-gpus-total",
            "2",
        ]
        if generation_revision:
            cmd.extend(["--revision", generation_revision])
        run_cmd(shlex.join(cmd), cwd=sorry_bench_dir)
    else:
        cmd = [
            "python",
            "gen_model_answer.py",
            "--bench-name",
            "sorry_bench",
            "--model-path",
            model_path,
            "--model-id",
            generation_model_id,
            "--dtype",
            generation_dtype,
            "--num-gpus-per-model",
            "1",
            "--num-gpus-total",
            "1",
        ]
        if generation_revision:
            cmd.extend(["--revision", generation_revision])
        run_cmd(shlex.join(cmd), cwd=sorry_bench_dir)

    ensure_exists(generated_model_answer, "generated model answer file")
    if generated_model_answer != model_answer:
        copy_overwrite_loud(generated_model_answer, model_answer, "normalized model answer file")
    ensure_exists(model_answer, "model answer file")

    # Remove thinking tokens before judgment generation
    backup_model_answer = remove_thinking_tokens(model_answer)

    run_cmd(
        f"python gen_judgment_safety_vllm.py --model-list {config['model_id']}",
        cwd=sorry_bench_dir,
    )
    ensure_exists(model_judgment, "model judgment file")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_dir = output_dir / "model_answer_stripped"
    backup_dir = output_dir / "model_answer_backup"
    judgment_dir = output_dir / "model_judgment"
    answer_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    judgment_dir.mkdir(parents=True, exist_ok=True)

    copied_answer = answer_dir / f"{config['model_id']}_model_answer.jsonl"
    copied_backup = backup_dir / f"{config['model_id']}_model_answer_backup.jsonl"
    copied_judgment = judgment_dir / f"{config['model_id']}_model_judgment.jsonl"
    merged_csv = output_dir / f"{config['model_id']}_english_results.csv"
    detailed_csv = output_dir / f"{config['model_id']}_english_results_detailed.csv"

    copy_overwrite_loud(model_answer, copied_answer, "output stripped model answer")
    copy_overwrite_loud(backup_model_answer, copied_backup, "output raw model answer backup")
    copy_overwrite_loud(model_judgment, copied_judgment, "output model judgment")

    questions_df = read_jsonl(Path(config["english_questions_jsonl"]))
    answers_df = read_jsonl(copied_answer)
    raw_answers_df = read_jsonl(copied_backup)
    judgments_df = read_jsonl(copied_judgment)

    answers_df["Output"] = answers_df["choices"].apply(extract_output)
    raw_answers_df["Output"] = raw_answers_df["choices"].apply(extract_output)
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
    write_csv_overwrite_loud(merged_df, merged_csv, "merged English eval CSV")

    detailed_questions_df = questions_df[["question_id", "Prompt"]].copy()
    detailed_questions_df["Prompt"] = detailed_questions_df["Prompt"].apply(
        lambda x: x[0] if isinstance(x, list) and x else x
    )
    raw_outputs_df = raw_answers_df[["question_id", "Output"]].rename(columns={"Output": "model_output_raw"})
    stripped_outputs_df = answers_df[["question_id", "Output"]].rename(columns={"Output": "model_output_stripped"})
    detailed_judgments_df = judgments_df[["question_id", "Rating"]].rename(columns={"Rating": "judgement"})

    detailed_df = detailed_questions_df.merge(raw_outputs_df, on="question_id", how="inner")
    detailed_df = detailed_df.merge(stripped_outputs_df, on="question_id", how="inner")
    detailed_df = detailed_df.merge(detailed_judgments_df, on="question_id", how="inner")
    detailed_df = detailed_df.rename(columns={"Prompt": "prompt"})
    detailed_df = detailed_df[["prompt", "model_output_raw", "model_output_stripped", "judgement"]]
    write_csv_overwrite_loud(detailed_df, detailed_csv, "detailed English eval CSV")

    return copied_answer, copied_backup, copied_judgment, merged_csv, detailed_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run English sorry-bench eval")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    answer_file, backup_file, judgment_file, result_file, detailed_result_file = run_eval(config)

    print(f"Saved model answers to: {answer_file}")
    print(f"Saved model answers backup to: {backup_file}")
    print(f"Saved model judgments to: {judgment_file}")
    print(f"Saved merged results to: {result_file}")
    print(f"Saved detailed merged results to: {detailed_result_file}")


if __name__ == "__main__":
    main()
