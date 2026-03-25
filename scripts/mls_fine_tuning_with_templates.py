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
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


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
        fp16 = accelerator == "mps"

    if "bf16" in config:
        bf16 = bool(config["bf16"])
    else:
        bf16 = accelerator == "cuda" and bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    if accelerator == "cpu":
        return False, False

    if fp16 and bf16:
        # Avoid an invalid mixed-precision config.
        bf16 = False

    return fp16, bf16


def build_dataset(input_csv: Path, template_name: str) -> tuple[Dataset, Any]:
    df = pd.read_csv(input_csv)
    validate_columns(df)
    df["instruction"] = df["instruction"].astype(str)
    df["input"] = df["input"].astype(str)
    df["response"] = df["response"].astype(str)

    dataset = Dataset.from_pandas(df)

    return dataset, template_name


def apply_template(dataset: Dataset, tokenizer: Any) -> Dataset:
    def format_row(example: dict[str, Any]) -> dict[str, str]:
        user_content = example["instruction"]
        if example.get("input") and str(example["input"]).lower() != "nan":
            user_content += "\n\n" + str(example["input"])

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(example["response"])},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(format_row)


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    local_model_dir = Path(config.get("local_model_dir", output_dir / "model_merged"))
    local_model_dir.mkdir(parents=True, exist_ok=True)
    accelerator = detect_accelerator()
    fp16, bf16 = precision_flags(config, accelerator)

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

    dataset, template_name = build_dataset(Path(config["input_csv"]), config["template_name"])
    tokenizer = get_chat_template(tokenizer, chat_template=template_name)
    dataset = apply_template(dataset, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
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
            group_by_length=bool(config.get("group_by_length", True)),
        ),
    )

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

    if bool(config.get("push_to_hub", False)):
        hf_repo = config.get("hf_repo")
        hf_token = config.get("hf_token")
        if not hf_repo or not hf_token:
            raise ValueError("push_to_hub=true requires hf_repo and hf_token in config")
        model.push_to_hub_merged(
            hf_repo,
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
