#!/usr/bin/env python3
"""One-time migration for local MLSFT artifacts to include ratio tag.

This script inserts '-r16alpha32' before '-E<epoch>-S<seed>' in artifact names
and in relevant JSON/JSONL/CSV string fields.

Default mode is dry-run. Use --apply to perform writes/renames.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

RATIO_TAG = "r16alpha32"
INSERT_RE = re.compile(r"(SynthDolly)(?!-r\d+alpha\d+)(-E\d+-S\d+)")
TARGET_EXTENSIONS = {".csv", ".json", ".jsonl"}
ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [
    ROOT / "data" / "outputs",
    ROOT / "data" / "results",
    ROOT / "data" / "run_summaries",
]


def inject_ratio_tag(text: str) -> str:
    return INSERT_RE.sub(rf"\1-{RATIO_TAG}\2", text)


def transform_json_obj(value: Any) -> tuple[Any, bool]:
    changed = False

    if isinstance(value, str):
        updated = inject_ratio_tag(value)
        return updated, updated != value

    if isinstance(value, list):
        out = []
        for item in value:
            new_item, item_changed = transform_json_obj(item)
            out.append(new_item)
            changed = changed or item_changed
        return out, changed

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            new_item, item_changed = transform_json_obj(item)
            out[key] = new_item
            changed = changed or item_changed
        return out, changed

    return value, False


def rewrite_json(path: Path, apply: bool) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[SKIP] Could not parse JSON: {path} ({exc})")
        return False

    updated, changed = transform_json_obj(data)
    if not changed:
        return False

    print(f"[EDIT] JSON content update: {path}")
    if apply:
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def rewrite_jsonl(path: Path, apply: bool) -> bool:
    changed = False
    lines_out: list[str] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if not line.strip():
                    lines_out.append(raw_line)
                    continue
                item = json.loads(line)
                updated, item_changed = transform_json_obj(item)
                changed = changed or item_changed
                lines_out.append(json.dumps(updated, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[SKIP] Could not parse JSONL: {path} ({exc})")
        return False

    if not changed:
        return False

    print(f"[EDIT] JSONL content update: {path}")
    if apply:
        path.write_text("".join(lines_out), encoding="utf-8")
    return True


def rewrite_csv(path: Path, apply: bool) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                print(f"[SKIP] CSV missing header: {path}")
                return False
            rows = list(reader)
    except Exception as exc:
        print(f"[SKIP] Could not parse CSV: {path} ({exc})")
        return False

    changed = False
    out_rows: list[dict[str, str]] = []
    for row in rows:
        out_row: dict[str, str] = {}
        for key, value in row.items():
            raw = "" if value is None else str(value)
            updated = inject_ratio_tag(raw)
            out_row[key] = updated
            if updated != raw:
                changed = True
        out_rows.append(out_row)

    if not changed:
        return False

    print(f"[EDIT] CSV content update: {path}")
    if apply:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reader.fieldnames))
            writer.writeheader()
            writer.writerows(out_rows)
    return True


def rename_path(path: Path, apply: bool) -> bool:
    new_name = inject_ratio_tag(path.name)
    if new_name == path.name:
        return False

    target = path.with_name(new_name)
    print(f"[RENAME] {path} -> {target}")
    if not apply:
        return True

    if target.exists():
        print(f"[SKIP] Target already exists, not renaming: {target}")
        return False

    path.rename(target)
    return True


def iter_existing_roots() -> list[Path]:
    existing = [root for root in SEARCH_ROOTS if root.exists()]
    missing = [root for root in SEARCH_ROOTS if not root.exists()]
    for root in missing:
        print(f"[INFO] Root not present, skipping: {root}")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time migration for local artifact ratio tags")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    apply = bool(args.apply)
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] Starting artifact migration with ratio tag '{RATIO_TAG}'.")

    roots = iter_existing_roots()
    if not roots:
        print("[INFO] No target roots found. Nothing to do.")
        return

    file_edits = 0
    file_renames = 0
    dir_renames = 0

    all_paths: list[Path] = []
    for root in roots:
        all_paths.extend(p for p in root.rglob("*") if p.is_file())

    for path in all_paths:
        if path.suffix.lower() in TARGET_EXTENSIONS:
            if path.suffix.lower() == ".json":
                if rewrite_json(path, apply):
                    file_edits += 1
            elif path.suffix.lower() == ".jsonl":
                if rewrite_jsonl(path, apply):
                    file_edits += 1
            elif path.suffix.lower() == ".csv":
                if rewrite_csv(path, apply):
                    file_edits += 1

        if rename_path(path, apply):
            file_renames += 1

    all_dirs: list[Path] = []
    for root in roots:
        all_dirs.extend(p for p in root.rglob("*") if p.is_dir())

    # Rename deepest directories first.
    for path in sorted(all_dirs, key=lambda p: len(p.parts), reverse=True):
        if rename_path(path, apply):
            dir_renames += 1

    print("\nMigration summary")
    print("-----------------")
    print(f"Mode: {mode}")
    print(f"Content edits: {file_edits}")
    print(f"File renames: {file_renames}")
    print(f"Directory renames: {dir_renames}")


if __name__ == "__main__":
    main()
