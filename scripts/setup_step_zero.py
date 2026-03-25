#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

from ctranslate2.converters.transformers import TransformersConverter
from huggingface_hub import snapshot_download


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    token = os.environ["HF_TOKEN"]

    autorater_dir = (
        root
        / "external"
        / "sorry-bench"
        / "ckpts"
        / "finetuned_models"
        / "ft-mistral-7b-instruct-v0.2-sorry-bench-202406"
    )
    autorater_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
        local_dir=str(autorater_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

    src_question = (
        root / "external" / "sorry-bench" / "data" / "sorry_bench" / "question.jsonl"
    )
    dst_question = (
        root / "data" / "inputs" / "eval_prompts" / "sorry-bench-questions.jsonl"
    )
    dst_question.parent.mkdir(parents=True, exist_ok=True)
    if src_question.exists() and not dst_question.exists():
        shutil.copy2(src_question, dst_question)

    ct2_dir = root / "external" / "nllb-3.3b-ct2-int8"
    if not (ct2_dir / "model.bin").exists():
        # transformers>=4.52 may pass dtype=None in a way M2M100 rejects; patch converter loader.
        original_load_model = TransformersConverter.load_model

        def patched_load_model(self, model_class, model_name_or_path, **kwargs):
            kwargs.pop("dtype", None)
            return original_load_model(self, model_class, model_name_or_path, **kwargs)

        TransformersConverter.load_model = patched_load_model
        try:
            TransformersConverter("facebook/nllb-200-3.3B").convert(
                str(ct2_dir), quantization="int8", force=True
            )
        finally:
            TransformersConverter.load_model = original_load_model

    print("Step Zero downloads and model prep complete.")


if __name__ == "__main__":
    main()