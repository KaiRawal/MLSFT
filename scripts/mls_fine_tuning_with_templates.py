"""
MLS fine-tuning with templates.

Purpose:
    Fine-tune a base model using local CSV training data and LoRA settings.

Inputs:
    - Config JSON (via --config)
    - CSV training data exported from Google Sheets

Outputs:
    - Training artifacts in output_dir
    - Optional local merged model directory in local_model_dir (disabled by default)
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
from transformers import AutoProcessor
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


def ensure_text_tokenizer(tokenizer_or_processor: Any) -> Any:
    """Normalize multimodal processor objects to a plain text tokenizer.

    Some checkpoints (for example Gemma-3 4B) can expose a Processor-like
    object with a nested `.tokenizer`. For text-only SFT we must use the
    underlying tokenizer so TRL stays on the text training path.
    """
    nested_tokenizer = getattr(tokenizer_or_processor, "tokenizer", None)
    if nested_tokenizer is not None and hasattr(nested_tokenizer, "apply_chat_template"):
        print("Detected multimodal processor; using its underlying text tokenizer.")
        return nested_tokenizer
    return tokenizer_or_processor


def build_dataset(
    input_csv: Path,
    template_name: str,
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[Dataset, Any]:
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

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # if 'llama' in template_name or 'gemma' in template_name:
        #     return {"messages": messages}
        return {"text": text}

    dataset = dataset.map(format_to_messages)

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        # Tokenize rendered chat text explicitly for model families that expect tokenized inputs.
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )

    # if 'llama' in template_name or 'gemma' in template_name:
    #     dataset = dataset.map(tokenize_function, batched=True)
    print(f"Using chat template: {template_name}")
    print("Example formatted prompt:\n", dataset[0])
    return dataset, template_name


def resolve_hf_push_target(config: dict[str, Any], epoch_tag: str | None) -> tuple[str, str, str]:
    hf_repo = get_optional_config_str(config, "hf_repo")
    hf_user = (os.environ.get("HF_USER") or "").strip()
    hf_token = (config.get("hf_token") or os.environ.get("HF_TOKEN") or "").strip()
    if not hf_repo or not hf_user or not hf_token:
        raise ValueError(
            "push_to_hub=true requires hf_repo in config, HF_USER and HF_TOKEN in env "
            "(or hf_token in config)"
        )

    hf_repo = append_epoch_tag(hf_repo, epoch_tag)
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
    epoch_tag = get_optional_config_str(config, "epoch_tag")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_local_model = bool(config.get("save_local_model", False))
    local_model_dir = Path(config.get("local_model_dir", output_dir / "model_merged"))
    if save_local_model:
        local_model_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("\n" + "!" * 88)
        print("WARNING: Local merged model export is DISABLED (save_local_model=false).")
        print(f"No model will be written to: {local_model_dir}")
        print("Hugging Face push behavior is unchanged and will still run if push_to_hub=true.")
        print("!" * 88 + "\n")
    accelerator = detect_accelerator()
    fp16, bf16 = precision_flags(config, accelerator)

    push_to_hub = bool(config.get("push_to_hub", False))
    full_repo_id = ""
    hf_token = ""
    if push_to_hub:
        _, full_repo_id, hf_token = resolve_hf_push_target(config, epoch_tag)
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

    tokenizer = ensure_text_tokenizer(tokenizer)

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
    dataset, template_name = build_dataset(
        Path(config["input_csv"]),
        config["template_name"],
        tokenizer,
        int(config.get("max_seq_length", 2048)),
    )

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
        group_by_length=bool(config.get("group_by_length", False)),
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "eval_dataset": None,
        "args": sft_args,
        # Keep training on the text-only SFT path across model families.
        # `tokenizer` here is normalized by ensure_text_tokenizer().
        "tokenizer": tokenizer,
    }

    trainer = SFTTrainer(**trainer_kwargs)

    runtime_stats = trainer.train()

    if save_local_model:
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
            
        if "gemma-3-4b" in config.get("model_name", "").lower():
            processor = AutoProcessor.from_pretrained(config["model_name"], trust_remote_code=True)
            processor.save_pretrained(str(local_model_dir))
    else:
        print("Skipped local merged-model save to preserve disk space.")

    peak_reserved_gb = None
    if accelerator == "cuda" and torch.cuda.is_available():
        peak_reserved_gb = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    elif accelerator == "mps":
        mps_stats = getattr(torch, "mps", None)
        if mps_stats and hasattr(mps_stats, "current_allocated_memory"):
            peak_reserved_gb = round(mps_stats.current_allocated_memory() / 1024 / 1024 / 1024, 3)

    summary = {
        "model_name": config["model_name"],
        "model_id": append_epoch_tag(str(config["model_id"]), epoch_tag),
        "epoch_tag": epoch_tag,
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
        "save_local_model": save_local_model,
        "local_model_dir": str(local_model_dir) if save_local_model else None,
    }

    if push_to_hub:
        model.push_to_hub_merged(
            full_repo_id,
            tokenizer,
            save_method=config.get("save_method", "merged_16bit"),
            token=hf_token,
        )
        if "gemma-3-4b" in config.get("model_name", "").lower():
            processor = AutoProcessor.from_pretrained(config["model_name"], trust_remote_code=True)
            processor.push_to_hub(full_repo_id, token=hf_token)

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