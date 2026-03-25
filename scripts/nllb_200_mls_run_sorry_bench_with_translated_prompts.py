"""
Run SORRY-Bench using translated prompts and NLLB-translated outputs.

Purpose:
    Evaluate local-language prompts by generating model outputs, translating outputs to English,
    and running sorry-bench autorating on translated outputs.

Inputs:
    - Config JSON (via --config)
    - Local-language prompts CSV
    - English prompts JSONL
    - Local sorry-bench repo with autorater weights already prepared
    - Local NLLB CTranslate2 model directory already prepared

Outputs:
    - Raw model answers JSONL
    - Translated model answers JSONL
    - Model judgments JSONL
    - Final merged CSV

Usage:
    python scripts/nllb_200_mls_run_sorry_bench_with_translated_prompts.py --config configs/nllb_200_mls_run_sorry_bench_with_translated_prompts.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import ctranslate2
import pandas as pd
import torch
from transformers import AutoTokenizer


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


def resolve_ct2_device(accelerator: str) -> str:
    # CTranslate2 supports CUDA and CPU backends.
    return "cuda" if accelerator == "cuda" else "cpu"


def resolve_requested_ct2_device(requested: Any, accelerator: str) -> str:
    fallback = resolve_ct2_device(accelerator)
    if requested is None:
        return fallback

    requested_norm = str(requested).strip().lower()
    if requested_norm in {"", "auto"}:
        return fallback
    if requested_norm in {"cuda", "nvidia"}:
        if torch.cuda.is_available():
            return "cuda"
        print("Requested translate_device=cuda but CUDA is unavailable; falling back to cpu.")
        return "cpu"
    if requested_norm in {"mps", "cpu"}:
        # CTranslate2 currently has no MPS backend, so use CPU.
        return "cpu"

    print(f"Unknown translate_device={requested!r}; using auto-selected device '{fallback}'.")
    return fallback


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


def to_jsonl_from_prompt_csv(input_csv: Path, output_jsonl: Path) -> None:
    df = pd.read_csv(input_csv)
    if "turns" not in df.columns:
        raise ValueError(f"Expected turns column in {input_csv}")
    df["turns"] = df["turns"].apply(lambda x: [x] if isinstance(x, str) else x)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def batch_translate(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    translator: Any,
    tokenizer: Any,
    batch_size: int,
) -> list[str]:
    tokenizer.src_lang = src_lang
    results: list[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        source = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in batch]
        target_prefix = [[tgt_lang]] * len(batch)

        translated = translator.translate_batch(
            source,
            target_prefix=target_prefix,
            beam_size=1,
            max_batch_size=batch_size,
        )

        for row in translated:
            text = tokenizer.decode(tokenizer.convert_tokens_to_ids(row.hypotheses[0]))
            results.append(text.replace(tgt_lang, "").strip())

    return results


def run_pipeline(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    sorry_bench_dir = Path(config["sorry_bench_dir"])
    nllb_ct2_dir = Path(config["nllb_ct2_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = detect_accelerator()
    generation_dtype = resolve_dtype(accelerator)
    model_path = resolve_model_path(config)

    ensure_exists(sorry_bench_dir, "sorry_bench_dir")
    ensure_exists(nllb_ct2_dir, "nllb_ct2_dir")

    local_question_jsonl = sorry_bench_dir / "data" / "sorry_bench" / "question.jsonl"
    english_question_jsonl = sorry_bench_dir / "data" / "sorry_bench" / "question_en.jsonl"

    to_jsonl_from_prompt_csv(Path(config["local_prompt_csv"]), local_question_jsonl)
    shutil.copy(Path(config["english_prompt_jsonl"]), english_question_jsonl)

    run_cmd(
        "python gen_model_answer_vllm.py "
        f"--bench-name sorry_bench --model-path {model_path} --model-id {config['model_id']} --dtype {generation_dtype}",
        cwd=sorry_bench_dir,
    )

    answer_path = sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{config['model_id']}.jsonl"
    ensure_exists(answer_path, "model answer jsonl")

    raw_answers = load_jsonl(answer_path)
    cleaned_answers: list[dict[str, Any]] = []
    source_texts: list[str] = []

    for item in raw_answers:
        original_text = item["choices"][0]["turns"][0]
        cleaned_text = re.sub(r"<think>.*?</think>", "", original_text, flags=re.DOTALL).strip()
        item["choices"][0]["turns"][0] = cleaned_text
        cleaned_answers.append(item)
        source_texts.append(cleaned_text)

    translator_device = resolve_requested_ct2_device(config.get("translate_device"), accelerator)
    translator = ctranslate2.Translator(str(nllb_ct2_dir), device=translator_device, compute_type="int8")
    tokenizer = AutoTokenizer.from_pretrained(config.get("nllb_model_name", "facebook/nllb-200-3.3B"))

    translated_texts = batch_translate(
        texts=source_texts,
        src_lang=config["source_lang_code"],
        tgt_lang="eng_Latn",
        translator=translator,
        tokenizer=tokenizer,
        batch_size=int(config.get("translation_batch_size", 32)),
    )

    translated_path = sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{config['model_id']}_translated.jsonl"
    translated_rows: list[dict[str, Any]] = []
    for i, item in enumerate(cleaned_answers):
        item["choices"][0]["turns"][0] = translated_texts[i]
        item["translation_meta"] = {"engine": "CTranslate2", "model": config.get("nllb_model_name", "facebook/nllb-200-3.3B")}
        translated_rows.append(item)
    save_jsonl(translated_path, translated_rows)

    del translator
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    autorater_generic = sorry_bench_dir / "data" / "sorry_bench" / "model_judgment" / "ft-mistral-7b-instruct-v0.2.jsonl"
    autorater_target = sorry_bench_dir / "data" / "sorry_bench" / "model_judgment" / f"{config['model_id']}_ft-mistral-7b-instruct-v0.2.jsonl"

    backup_local = local_question_jsonl.with_suffix(".backup.jsonl")
    shutil.move(local_question_jsonl, backup_local)
    shutil.copy(english_question_jsonl, local_question_jsonl)

    try:
        run_cmd(
            f"python gen_judgment_safety_vllm.py --model-list {config['model_id']}_translated",
            cwd=sorry_bench_dir,
        )
    finally:
        local_question_jsonl.unlink(missing_ok=True)
        shutil.move(backup_local, local_question_jsonl)

    ensure_exists(autorater_generic, "autorater generic output")
    if autorater_target.exists():
        autorater_target.unlink()
    shutil.move(autorater_generic, autorater_target)

    questions = load_jsonl(english_question_jsonl)
    answers = load_jsonl(translated_path)
    judgments = load_jsonl(autorater_target)

    safe_length = min(len(questions), len(answers), len(judgments))
    merged = []
    for i in range(safe_length):
        merged.append(
            {
                "question_id": questions[i].get("question_id", "N/A"),
                "category": questions[i].get("category", "N/A"),
                "prompt": questions[i].get("turns", [""])[0],
                "translated_output": answers[i]["choices"][0]["turns"][0],
                "rating": judgments[i].get("judgment", judgments[i].get("score", "Error")),
            }
        )

    raw_copy = output_dir / f"{config['model_id']}_raw_local_answers.jsonl"
    translated_copy = output_dir / f"{config['model_id']}_translated_answers.jsonl"
    judgment_copy = output_dir / f"{config['model_id']}_translated_judgment.jsonl"
    merged_csv = output_dir / f"{config['model_id']}_{config['language_code']}_translated_eval.csv"

    shutil.copy(answer_path, raw_copy)
    shutil.copy(translated_path, translated_copy)
    shutil.copy(autorater_target, judgment_copy)
    pd.DataFrame(merged).to_csv(merged_csv, index=False)

    return raw_copy, translated_copy, judgment_copy, merged_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local-language translated sorry-bench eval")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_file, translated_file, judgment_file, result_file = run_pipeline(config)

    print(f"Saved raw outputs to: {raw_file}")
    print(f"Saved translated outputs to: {translated_file}")
    print(f"Saved judgments to: {judgment_file}")
    print(f"Saved merged results to: {result_file}")


if __name__ == "__main__":
    main()
