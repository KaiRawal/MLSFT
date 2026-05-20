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
    python src/nllb_200_mls_run_sorry_bench_with_translated_prompts.py --config configs/nllb_200_mls_run_sorry_bench_with_translated_prompts.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any
import sys
import atexit

from gpu_cleanup import safe_run_cmd, cleanup_torch, register_signal_handlers

import ctranslate2
import pandas as pd
import torch
from transformers import AutoTokenizer

from gpu_selection import resolve_generation_gpu_count


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(command: str, cwd: Path) -> None:  # kept for backward compatibility (not used)
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


def append_epoch_tag(identifier: str, epoch_tag: str | None) -> str:
    if not epoch_tag:
        return identifier
    if identifier.endswith(epoch_tag) or f"-{epoch_tag}" in identifier:
        return identifier
    return f"{identifier}-{epoch_tag}"


def resolve_output_model_id(config: dict[str, Any]) -> str:
    epoch_tag = get_optional_config_str(config, "epoch_tag")
    return append_epoch_tag(str(config["model_id"]), epoch_tag)


def resolve_generation_model_id(config: dict[str, Any], output_model_id: str) -> str:
    return get_optional_config_str(config, "generation_model_id_for_template") or output_model_id


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


def to_jsonl_from_prompt_csv(input_csv: Path, output_jsonl: Path) -> None:
    df = pd.read_csv(input_csv)
    if "turns" not in df.columns:
        raise ValueError(f"Expected turns column in {input_csv}")
    df["turns"] = df["turns"].apply(lambda x: [x] if isinstance(x, str) else x)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        print(f"Overwriting existing local question JSONL: {output_jsonl}")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Overwriting existing JSONL: {path}")
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def run_pipeline(config: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path, Path]:
    sorry_bench_dir = Path(config["sorry_bench_dir"])
    nllb_ct2_dir = Path(config["nllb_ct2_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir / "model_answer_backup"
    stripped_dir = output_dir / "stripped_local_answers"
    translated_dir = output_dir / "translated_answers"
    judgment_dir = output_dir / "translated_judgment"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stripped_dir.mkdir(parents=True, exist_ok=True)
    translated_dir.mkdir(parents=True, exist_ok=True)
    judgment_dir.mkdir(parents=True, exist_ok=True)
    accelerator = detect_accelerator()
    generation_dtype = resolve_generation_dtype(config, accelerator)
    model_path = resolve_model_path(config)
    generation_backend = resolve_generation_backend(config, model_path)
    output_model_id = resolve_output_model_id(config)
    generation_model_id = resolve_generation_model_id(config, output_model_id)
    generation_gpu_count = resolve_generation_gpu_count(model_path, generation_model_id)
    generation_revision = resolve_generation_revision(config)

    warn_unsupported_generation_tokenizer(config)

    print(f"Using generation backend: {generation_backend}")
    if generation_model_id != output_model_id:
        print(
            "Using generation_model_id_for_template override: "
            f"{generation_model_id} (logical output model_id remains {output_model_id})"
        )

    ensure_exists(sorry_bench_dir, "sorry_bench_dir")
    ensure_exists(nllb_ct2_dir, "nllb_ct2_dir")

    local_question_jsonl = sorry_bench_dir / "data" / "sorry_bench" / "question.jsonl"
    english_question_jsonl = sorry_bench_dir / "data" / "sorry_bench" / "question_en.jsonl"

    # Prepare local and English question JSONL files.
    # If a local-language CSV is provided and exists, convert it to JSONL.
    # Otherwise (e.g., English run), reuse the provided English JSONL as the local prompts.
    local_prompt_csv = Path(config.get("local_prompt_csv", ""))
    english_prompt_cfg = Path(config.get("english_prompt_jsonl", ""))

    if local_prompt_csv.exists():
        to_jsonl_from_prompt_csv(local_prompt_csv, local_question_jsonl)
    else:
        # No local CSV available (likely the English run); ensure english prompt exists and copy it
        if not english_prompt_cfg.exists():
            raise FileNotFoundError(f"Missing required English prompt JSONL: {english_prompt_cfg}")
        copy_overwrite_loud(english_prompt_cfg, local_question_jsonl, "English question JSONL used as local prompts")

    # Always populate the canonical english_question_jsonl used later in the pipeline
    copy_overwrite_loud(english_prompt_cfg, english_question_jsonl, "English question JSONL")

    answer_path = sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{output_model_id}.jsonl"
    generated_answer_path = (
        sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{generation_model_id}.jsonl"
    )
    stripped_path = sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{output_model_id}_stripped.jsonl"
    translated_path = sorry_bench_dir / "data" / "sorry_bench" / "model_answer" / f"{output_model_id}_translated.jsonl"
    autorater_generic = sorry_bench_dir / "data" / "sorry_bench" / "model_judgment" / "ft-mistral-7b-instruct-v0.2.jsonl"
    autorater_target = (
        sorry_bench_dir / "data" / "sorry_bench" / "model_judgment" / f"{output_model_id}_ft-mistral-7b-instruct-v0.2.jsonl"
    )

    # Sorry-bench generation/judgment scripts append by default; clear per-run artifacts to force fresh eval.
    overwrite_loud(answer_path, "sorry-bench model answer cache")
    if generated_answer_path != answer_path:
        overwrite_loud(generated_answer_path, "sorry-bench model answer cache (template model id override)")
    overwrite_loud(stripped_path, "sorry-bench stripped model answer cache")
    overwrite_loud(translated_path, "sorry-bench translated model answer cache")
    overwrite_loud(autorater_generic, "sorry-bench model judgment cache")
    overwrite_loud(autorater_target, "sorry-bench translated model judgment cache")

    if generation_backend == "vllm":
        cmd = [
            sys.executable,
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
            str(generation_gpu_count),
            "--num-gpus-total",
            str(generation_gpu_count),
        ]
        if generation_revision:
            cmd.extend(["--revision", generation_revision])
        safe_run_cmd(cmd, cwd=sorry_bench_dir)
    else:
        cmd = [
            sys.executable,
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
            str(generation_gpu_count),
            "--num-gpus-total",
            str(generation_gpu_count),
        ]
        if generation_revision:
            cmd.extend(["--revision", generation_revision])
        safe_run_cmd(cmd, cwd=sorry_bench_dir)

    ensure_exists(generated_answer_path, "generated model answer jsonl")
    if generated_answer_path != answer_path:
        copy_overwrite_loud(generated_answer_path, answer_path, "normalized model answer jsonl")
    ensure_exists(answer_path, "model answer jsonl")

    backup_answer_path = answer_path.with_stem(f"{answer_path.stem}_backup")
    shutil.copy(answer_path, backup_answer_path)

    raw_answers = load_jsonl(answer_path)
    stripped_answers: list[dict[str, Any]] = []
    source_texts: list[str] = []

    for item in raw_answers:
        stripped_item = copy.deepcopy(item)
        original_text = stripped_item["choices"][0]["turns"][0]
        cleaned_text = strip_nested_think_tags(original_text)
        stripped_item["choices"][0]["turns"][0] = cleaned_text
        stripped_answers.append(stripped_item)
        source_texts.append(cleaned_text)

    save_jsonl(stripped_path, stripped_answers)

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

    translated_rows: list[dict[str, Any]] = []
    for i, item in enumerate(stripped_answers):
        translated_item = copy.deepcopy(item)
        translated_item["choices"][0]["turns"][0] = translated_texts[i]
        translated_item["translation_meta"] = {
            "engine": "CTranslate2",
            "model": config.get("nllb_model_name", "facebook/nllb-200-3.3B"),
        }
        translated_rows.append(translated_item)
    save_jsonl(translated_path, translated_rows)

    # Prefer explicit close if available, then best-effort cleanup.
    try:
        if hasattr(translator, "close"):
            try:
                translator.close()
            except Exception:
                pass
    except Exception:
        pass

    try:
        del translator
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    gc.collect()
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        elif torch.backends.mps.is_available() and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass

    backup_local = local_question_jsonl.with_suffix(".backup.jsonl")
    shutil.move(local_question_jsonl, backup_local)
    copy_overwrite_loud(english_question_jsonl, local_question_jsonl, "temporary English question JSONL swap")

    try:
        safe_run_cmd(
            [sys.executable, "gen_judgment_safety_vllm.py", "--model-list", f"{output_model_id}_translated"],
            cwd=sorry_bench_dir,
        )
    finally:
        local_question_jsonl.unlink(missing_ok=True)
        shutil.move(backup_local, local_question_jsonl)

    ensure_exists(autorater_generic, "autorater generic output")
    overwrite_loud(autorater_target, "translated autorater output")
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

    backup_copy = backup_dir / f"{output_model_id}_model_answer_backup.jsonl"
    stripped_copy = stripped_dir / f"{output_model_id}_stripped_local_answers.jsonl"
    translated_copy = translated_dir / f"{output_model_id}_translated_answers.jsonl"
    judgment_copy = judgment_dir / f"{output_model_id}_translated_judgment.jsonl"
    merged_csv = output_dir / f"{output_model_id}_{config['language_code']}_translated_eval.csv"
    detailed_csv = output_dir / f"{output_model_id}_{config['language_code']}_translated_eval_detailed.csv"

    copy_overwrite_loud(backup_answer_path, backup_copy, "output raw model answer backup")
    copy_overwrite_loud(stripped_path, stripped_copy, "output stripped local answers")
    copy_overwrite_loud(translated_path, translated_copy, "output translated answers")
    copy_overwrite_loud(autorater_target, judgment_copy, "output translated judgments")
    write_csv_overwrite_loud(pd.DataFrame(merged), merged_csv, "merged translated eval CSV")

    english_questions_df = pd.DataFrame(load_jsonl(english_question_jsonl))
    local_questions_df = pd.DataFrame(load_jsonl(local_question_jsonl))
    raw_answers_df = pd.DataFrame(load_jsonl(backup_answer_path))
    stripped_answers_df = pd.DataFrame(load_jsonl(stripped_path))
    translated_answers_df = pd.DataFrame(load_jsonl(translated_path))
    judgments_df = pd.DataFrame(load_jsonl(autorater_target))

    english_questions_df = english_questions_df[["question_id", "turns"]].rename(columns={"turns": "prompt_english"})
    local_questions_df = local_questions_df[["question_id", "turns"]].rename(columns={"turns": "prompt_translated"})
    english_questions_df["prompt_english"] = english_questions_df["prompt_english"].apply(
        lambda x: x[0] if isinstance(x, list) and x else x
    )
    local_questions_df["prompt_translated"] = local_questions_df["prompt_translated"].apply(
        lambda x: x[0] if isinstance(x, list) and x else x
    )

    raw_answers_df["model_output_raw"] = raw_answers_df["choices"].apply(extract_output)
    stripped_answers_df["model_output_stripped"] = stripped_answers_df["choices"].apply(extract_output)
    translated_answers_df["model_output_translated"] = translated_answers_df["choices"].apply(extract_output)

    if "judgment" in judgments_df.columns:
        judgments_df = judgments_df.rename(columns={"judgment": "judgement"})
    elif "score" in judgments_df.columns:
        judgments_df = judgments_df.rename(columns={"score": "judgement"})
    else:
        judgments_df["judgement"] = ""

    detailed_df = english_questions_df.merge(local_questions_df, on="question_id", how="inner")
    detailed_df = detailed_df.merge(raw_answers_df[["question_id", "model_output_raw"]], on="question_id", how="inner")
    detailed_df = detailed_df.merge(
        stripped_answers_df[["question_id", "model_output_stripped"]], on="question_id", how="inner"
    )
    detailed_df = detailed_df.merge(
        translated_answers_df[["question_id", "model_output_translated"]], on="question_id", how="inner"
    )
    detailed_df = detailed_df.merge(judgments_df[["question_id", "judgement"]], on="question_id", how="inner")
    detailed_df = detailed_df[
        [
            "prompt_english",
            "prompt_translated",
            "model_output_raw",
            "model_output_stripped",
            "model_output_translated",
            "judgement",
        ]
    ]
    write_csv_overwrite_loud(detailed_df, detailed_csv, "detailed translated eval CSV")

    # Best-effort cleanup before returning.
    try:
        cleanup_torch()
    except Exception:
        pass

    return backup_copy, stripped_copy, translated_copy, judgment_copy, merged_csv, detailed_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local-language translated sorry-bench eval")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    # Register cleanup handlers for abrupt termination.
    atexit.register(cleanup_torch)
    register_signal_handlers()

    backup_file, stripped_file, translated_file, judgment_file, result_file, detailed_result_file = run_pipeline(config)

    print(f"Saved model answers backup to: {backup_file}")
    print(f"Saved stripped local-language outputs to: {stripped_file}")
    print(f"Saved translated outputs to: {translated_file}")
    print(f"Saved judgments to: {judgment_file}")
    print(f"Saved merged results to: {result_file}")
    print(f"Saved detailed merged results to: {detailed_result_file}")


if __name__ == "__main__":
    main()
