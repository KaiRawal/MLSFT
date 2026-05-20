#!/usr/bin/env python3
"""One-time migration for Hugging Face model repos to include ratio tag.

This script enumerates all models for HF_USER and renames any repo ID that matches
MLSFT SynthDolly naming without a ratio tag by inserting '-r16alpha32' before
'-E<epoch>-S<seed>'.

Default mode is dry-run. Use --apply to perform renames.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from huggingface_hub import HfApi

RATIO_TAG = "r16alpha32"
INSERT_RE = re.compile(r"(SynthDolly)(?!-r\d+alpha\d+)(-E\d+-S\d+)")


def inject_ratio_tag(text: str) -> str:
    return INSERT_RE.sub(rf"\1-{RATIO_TAG}\2", text)


def get_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        print(f"ERROR: {name} environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time HF repo rename to include ratio tags")
    parser.add_argument("--apply", action="store_true", help="Apply repo renames (default is dry-run)")
    args = parser.parse_args()

    hf_user = get_env("HF_USER")
    hf_token = get_env("HF_TOKEN")

    apply = bool(args.apply)
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] Enumerating model repos for user: {hf_user}")

    api = HfApi(token=hf_token)
    models = list(api.list_models(author=hf_user))
    print(f"Found {len(models)} total model repos under {hf_user}.")

    planned = 0
    renamed = 0
    failed = 0

    for model in models:
        source_id = getattr(model, "id", "")
        if not source_id:
            continue
        if "SynthDolly" not in source_id:
            continue

        target_id = inject_ratio_tag(source_id)
        if target_id == source_id:
            continue

        planned += 1
        print(f"[PLAN] {source_id} -> {target_id}")

        if not apply:
            continue

        try:
            api.move_repo(
                from_id=source_id,
                to_id=target_id,
                repo_type="model",
                token=hf_token,
            )
            renamed += 1
            print(f"[OK] Renamed: {source_id} -> {target_id}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] Could not rename {source_id} -> {target_id}: {exc}")

    print("\nMigration summary")
    print("-----------------")
    print(f"Mode: {mode}")
    print(f"Planned renames: {planned}")
    print(f"Successful renames: {renamed}")
    print(f"Failed renames: {failed}")


if __name__ == "__main__":
    main()
