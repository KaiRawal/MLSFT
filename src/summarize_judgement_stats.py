#!/usr/bin/env python3
"""Summarize judgement stats across evaluation CSV files.

Scans CSV files in:
- data/outputs/eval_translated/*_detailed*.csv
- data/outputs/eval_english/*_detailed*.csv

For each file, computes:
- count_zeros: number of judgement values equal to 0
- count_ones: number of judgement values equal to 1
- total_length: total number of rows in the judgement column

Pre/post rates are computed as proportion of ones:
- rate = count_ones / total_length

Writes one consolidated CSV to data/results/judgement_stats_summary.csv,
overwriting it on each run.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATTERNS = [
    ROOT / "data/outputs/eval_translated/*_detailed*.csv",
    ROOT / "data/outputs/eval_english/*_detailed*.csv",
]
OUTPUT_DIR = ROOT / "data" / "results"
OUTPUT_CSV = OUTPUT_DIR / "judgement_stats_summary.csv"
COMPARISON_OUTPUT_CSV = OUTPUT_DIR / "compliance_rate_stats.csv"


def normalize_judgement(value: str) -> str:
    """Normalize judgement values so 0/1 can be counted robustly."""
    cleaned = value.strip()
    if cleaned in {"0", "0.0"}:
        return "0"
    if cleaned in {"1", "1.0"}:
        return "1"
    return cleaned


def analyze_csv(file_path: Path) -> Dict[str, int | str]:
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {file_path}")

        lower_to_original = {name.lower(): name for name in reader.fieldnames if name}
        judgement_column = lower_to_original.get("judgement")
        if not judgement_column:
            raise ValueError(f"Missing 'judgement' column in {file_path}")

        count_zeros = 0
        count_ones = 0
        total_length = 0

        for row in reader:
            raw = row.get(judgement_column, "")
            value = normalize_judgement(str(raw) if raw is not None else "")

            total_length += 1
            if value == "0":
                count_zeros += 1
            elif value == "1":
                count_ones += 1

    return {
        "csv_name": file_path.name,
        "count_zeros": count_zeros,
        "count_ones": count_ones,
        "total_length": total_length,
    }


def collect_input_files() -> List[Path]:
    files: List[Path] = []
    for pattern in INPUT_PATTERNS:
        files.extend(sorted(ROOT.glob(str(pattern.relative_to(ROOT)))))
    return files


def write_summary(rows: List[Dict[str, int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["csv_name", "count_zeros", "count_ones", "total_length"],
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_model_finetune_and_eval_type(csv_name: str) -> Tuple[str, str, str]:
    """Parse summary row filename into model key, status, and eval type."""
    if csv_name.endswith("_english_results_detailed.csv"):
        stem = csv_name[: -len("_english_results_detailed.csv")]
        eval_type = "english"
    elif csv_name.endswith("_translated_eval_detailed.csv"):
        translated_stem = csv_name[: -len("_translated_eval_detailed.csv")]
        model_part, sep, _eval_lang = translated_stem.rpartition("_")
        stem = model_part if sep else translated_stem
        eval_type = "translated"
    else:
        raise ValueError(f"Unrecognized evaluation filename format: {csv_name}")

    if "-PREFT" in stem:
        model_finetune = stem.replace("-PREFT", "", 1)
        status = "preft"
    else:
        model_finetune = stem
        status = "postft"

    return model_finetune, status, eval_type


def format_rate(rate: Optional[float]) -> str:
    if rate is None:
        return ""
    return f"{rate:.6f}"


def build_pre_post_rate_rows(rows: List[Dict[str, int | str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}

    for row in rows:
        csv_name = str(row["csv_name"])
        model_finetune, status, eval_type = extract_model_finetune_and_eval_type(csv_name)

        total = int(row["total_length"])
        ones = int(row["count_ones"])
        rate = (ones / total) if total else 0.0

        if model_finetune not in grouped:
            grouped[model_finetune] = {
                "preft": {},
                "postft": {},
            }

        grouped[model_finetune][status][eval_type] = rate

    output_rows: List[Dict[str, str]] = []
    for model_finetune in sorted(grouped):
        pre = grouped[model_finetune]["preft"]
        post = grouped[model_finetune]["postft"]

        output_rows.append(
            {
                "model_finetune": model_finetune,
                "preft_rate_english": format_rate(pre.get("english")),
                "preft_rate_translated": format_rate(pre.get("translated")),
                "postft_rate_english": format_rate(post.get("english")),
                "postft_rate_translated": format_rate(post.get("translated")),
            }
        )

    return output_rows


def write_pre_post_rates(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_finetune",
                "preft_rate_english",
                "preft_rate_translated",
                "postft_rate_english",
                "postft_rate_translated",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    files = collect_input_files()
    results = [analyze_csv(path) for path in files]
    write_summary(results, OUTPUT_CSV)
    comparison_rows = build_pre_post_rate_rows(results)
    write_pre_post_rates(comparison_rows, COMPARISON_OUTPUT_CSV)

    print(f"Wrote {len(results)} rows to {OUTPUT_CSV}")
    print(f"Wrote {len(comparison_rows)} rows to {COMPARISON_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
