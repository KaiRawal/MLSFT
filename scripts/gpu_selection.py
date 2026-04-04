from __future__ import annotations


def resolve_generation_gpu_count(model_path: str, model_id: str) -> int:
    """Return generation GPU count.

    Defaults to 1 GPU. Returns 2 only for Qwen 32B models.
    """
    combined = f"{model_path} {model_id}".lower()
    if "qwen" in combined and "32b" in combined:
        return 2
    return 1
