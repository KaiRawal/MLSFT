"""
MLS fine-tuning with templates.

Purpose:
    Fine-tune a base model using local CSV training data and LoRA settings.

Inputs:
    - Config JSON (via --config)
    - CSV training data exported from Google Sheets

Outputs:
    - Training artifacts in output_dir
    - Local merged model directory in local_model_dir
    - Training summary JSON in summary_json
    - Optional push to Hugging Face Hub (if enabled)

Usage:
    python scripts/mls_fine_tuning_with_templates.py --config configs/mls_fine_tuning_with_templates.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"instruction", "input", "response"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def detect_accelerator() -> str:
    # Prefer MPS first as requested, then CUDA, then CPU.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def precision_flags(config: dict[str, Any], accelerator: str) -> tuple[bool, bool]:
    if "fp16" in config:
        fp16 = bool(config["fp16"])
    else:
        fp16 = accelerator in {"mps", "cuda"}

    if "bf16" in config:
        bf16 = bool(config["bf16"])
    else:
        bf16 = accelerator == "cuda" and bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    if accelerator == "cpu":
        return False, False

    if accelerator == "cuda" and bf16:
        # Prefer bf16 on supported CUDA devices.
        fp16 = False

    if fp16 and bf16:
        # Avoid an invalid mixed-precision config.
        bf16 = False

    return fp16, bf16


def build_dataset(input_csv: Path, template_name: str, tokenizer: Any) -> tuple[Dataset, Any]:
    df = pd.read_csv(input_csv)
    validate_columns(df)
    df["instruction"] = df["instruction"].astype(str)
    df["input"] = df["input"].astype(str)
    df["response"] = df["response"].astype(str)

    dataset = Dataset.from_pandas(df)

    def format_to_messages(example: dict[str, Any]) -> dict[str, Any]:
        user_content = example["instruction"]
        if example["input"] and str(example["input"]).lower() != "nan":
            user_content += "\n\n" + example["input"]

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["response"]},
        ]

        # Apply chat template and return only the formatted text string.
        # Returning only "text" (not "messages") is critical: if a "messages"
        # column (list of dicts) is present in the dataset, the HF data collator
        # will try to tensorise it and crash with "Could not infer dtype of dict".
        # Unsloth patches this away for text-only models but not for multimodal
        # Processors (e.g. Gemma-3's Gemma3Processor), so we avoid the column
        # entirely to stay compatible across all model families.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_to_messages)
    return dataset, template_name





def resolve_hf_push_target(config: dict[str, Any]) -> tuple[str, str, str]:
    hf_repo = config.get("hf_repo")
    hf_user = (os.environ.get("HF_USER") or "").strip()
    hf_token = (config.get("hf_token") or os.environ.get("HF_TOKEN") or "").strip()
    if not hf_repo or not hf_user or not hf_token:
        raise ValueError(
            "push_to_hub=true requires hf_repo in config, HF_USER and HF_TOKEN in env "
            "(or hf_token in config)"
        )

    full_repo_id = hf_repo if "/" in hf_repo else f"{hf_user}/{hf_repo}"
    return hf_repo, full_repo_id, hf_token


def enforce_existing_repo_policy(full_repo_id: str, hf_token: str, allow_existing_hf_repo: bool) -> None:
    api = HfApi()
    try:
        api.model_info(repo_id=full_repo_id, token=hf_token)
        repo_exists = True
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            repo_exists = False
        else:
            raise RuntimeError(
                f"Unable to validate Hugging Face repo status for '{full_repo_id}' (HTTP {status_code})."
            ) from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to validate Hugging Face repo status for '{full_repo_id}'.") from exc

    if repo_exists and not allow_existing_hf_repo:
        raise ValueError(
            "Target Hugging Face repo already exists: "
            f"{full_repo_id}. Set allow_existing_hf_repo=true to permit re-training and overwriting."
        )


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    local_model_dir = Path(config.get("local_model_dir", output_dir / "model_merged"))
    local_model_dir.mkdir(parents=True, exist_ok=True)
    accelerator = detect_accelerator()
    fp16, bf16 = precision_flags(config, accelerator)

    push_to_hub = bool(config.get("push_to_hub", False))
    full_repo_id = ""
    hf_token = ""
    if push_to_hub:
        _, full_repo_id, hf_token = resolve_hf_push_target(config)
        enforce_existing_repo_policy(
            full_repo_id=full_repo_id,
            hf_token=hf_token,
            allow_existing_hf_repo=bool(config.get("allow_existing_hf_repo", False)),
        )

    model, tokenizer = FastModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=bool(config.get("load_in_4bit", False)),
        load_in_8bit=bool(config.get("load_in_8bit", False)),
        full_finetuning=bool(config.get("full_finetuning", False)),
    )

    model = FastModel.get_peft_model(
        model,
        r=int(config.get("lora_r", 16)),
        target_modules=config.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        bias=config.get("lora_bias", "none"),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", "unsloth"),
        random_state=int(config["random_seed"]),
        use_rslora=bool(config.get("use_rslora", False)),
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=config["template_name"])
    dataset, template_name = build_dataset(Path(config["input_csv"]), config["template_name"], tokenizer)

    sft_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        warmup_steps=int(config.get("warmup_steps", 5)),
        num_train_epochs=int(config.get("num_train_epochs", 1)),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        logging_steps=int(config.get("logging_steps", 1)),
        optim=config.get("optim", "adamw_8bit"),
        weight_decay=float(config.get("weight_decay", 0.01)),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        seed=int(config["random_seed"]),
        output_dir=str(output_dir),
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(config.get("dataloader_num_workers", 2)),
        # Must be False for Gemma-3 (and any model whose processing_class is a
        # multimodal Processor rather than a bare tokenizer). LengthGroupedSampler
        # requires input_ids to be present in the dataset before training begins;
        # unsloth pre-populates these for text-only models but not for Processor-
        # based ones, causing a crash. False is the safe default across all families.
        group_by_length=bool(config.get("group_by_length", False)),
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "eval_dataset": None,
        "args": sft_args,
        "processing_class": tokenizer,
    }

    trainer = SFTTrainer(**trainer_kwargs)

    runtime_stats = trainer.train()

    # Always export a local merged model so downstream eval can run without HF push.
    try:
        model.save_pretrained_merged(
            str(local_model_dir),
            tokenizer,
            save_method=config.get("save_method", "merged_16bit"),
        )
    except AttributeError:
        # Fallback path if merged-save API is unavailable in the installed version.
        model.save_pretrained(str(local_model_dir))
        tokenizer.save_pretrained(str(local_model_dir))

    peak_reserved_gb = None
    if accelerator == "cuda" and torch.cuda.is_available():
        peak_reserved_gb = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    elif accelerator == "mps":
        mps_stats = getattr(torch, "mps", None)
        if mps_stats and hasattr(mps_stats, "current_allocated_memory"):
            peak_reserved_gb = round(mps_stats.current_allocated_memory() / 1024 / 1024 / 1024, 3)

    summary = {
        "model_name": config["model_name"],
        "model_id": config["model_id"],
        "language": config["language"],
        "random_seed": int(config["random_seed"]),
        "num_train_epochs": int(config.get("num_train_epochs", 1)),
        "rows": len(dataset),
        "train_runtime_seconds": runtime_stats.metrics.get("train_runtime"),
        "accelerator": accelerator,
        "peak_reserved_gb": peak_reserved_gb,
        "fp16": fp16,
        "bf16": bf16,
        "output_dir": str(output_dir),
        "local_model_dir": str(local_model_dir),
    }

    if push_to_hub:
        model.push_to_hub_merged(
            full_repo_id,
            tokenizer,
            save_method=config.get("save_method", "merged_16bit"),
            token=hf_token,
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MLS fine-tuning runner")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_training(config)

    summary_json = Path(config["summary_json"])
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved training summary to {summary_json}")


if __name__ == "__main__":
    main()