#!/usr/bin/env python3
"""
Upload MLSFT fine-tuning and safety eval datasets to Hugging Face Hub.

Reads from data/inputs/fine_tuning/ and data/inputs/eval_prompts/ subdirectories, formats data with language fields,
generates metadata, and uploads to HF Hub.

Environment variables:
    HF_USER: Hugging Face username (or organization name)
    HF_TOKEN: Hugging Face API token
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file, get_collection, add_collection_item

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
FINE_TUNING_DIR = ROOT_DIR / "data" / "inputs" / "fine_tuning"
EVAL_PROMPTS_DIR = ROOT_DIR / "data" / "inputs" / "eval_prompts"

# Language code mapping from CSV filenames
LANGUAGE_MAPPING = {
    "Chinese": "zh",
    "Danish": "da",
    "English": "en",
    "Greek": "el",
    "Hindi": "hi",
    "Irish": "ga",
    "Portuguese": "pt",
    "Spanish": "es",
    "Tagalog": "tl",
}

# Dataset names on HF Hub
FINETUNING_DATASET_NAME = "mlsft-benign-multilingual-finetuning"
EVAL_DATASET_NAME = "mlsft-mutlilingual-sorry-bench-eval"

# HF credentials from environment
HF_USER = os.getenv("HF_USER")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_USER or not HF_TOKEN:
    raise ValueError("Environment variables HF_USER and HF_TOKEN must be set")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def extract_language(filename: str) -> str:
    """Extract language code from CSV filename."""
    for lang_name, lang_code in LANGUAGE_MAPPING.items():
        if lang_name in filename:
            return lang_code
    raise ValueError(f"Could not extract language from filename: {filename}")


def load_fine_tuning_data() -> pd.DataFrame:
    """Load all fine-tuning CSVs and combine into one DataFrame with language field."""
    print("\n" + "=" * 70)
    print("LOADING FINE-TUNING DATA")
    print("=" * 70)

    dfs = []
    csv_files = sorted(FINE_TUNING_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} fine-tuning CSV files")

    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        lang_code = extract_language(csv_file.name)
        df = pd.read_csv(csv_file)
        df["language"] = lang_code
        print(f"    - {lang_code}: {len(df)} rows")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    return combined_df


def load_eval_data() -> pd.DataFrame:
    """Load all eval prompt CSVs and combine into one DataFrame with language field."""
    print("\n" + "=" * 70)
    print("LOADING EVALUATION DATA")
    print("=" * 70)

    dfs = []
    csv_files = sorted(EVAL_PROMPTS_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} evaluation CSV files")

    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        lang_code = extract_language(csv_file.name)
        df = pd.read_csv(csv_file)
        df["language"] = lang_code
        print(f"    - {lang_code}: {len(df)} rows")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    return combined_df


def validate_data(df: pd.DataFrame, dataset_type: str) -> bool:
    """Validate data quality."""
    print(f"\nValidating {dataset_type} data...")
    
    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"  ⚠ Warning: Found null values:")
        print(null_counts[null_counts > 0])
    else:
        print(f"  ✓ No null values")
    
    # Check language field
    if "language" in df.columns:
        langs = df["language"].unique()
        print(f"  ✓ Language coverage: {len(langs)} languages")
        print(f"    Languages: {sorted(langs)}")
    
    return True


def save_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame as JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient='records', lines=True)
    print(f"✓ Saved JSONL: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame as CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved CSV: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def generate_dataset_info_finetuning(df: pd.DataFrame) -> Dict:
    """Generate dataset_info.json for fine-tuning dataset."""
    lang_counts = df["language"].value_counts().sort_index().to_dict()
    
    return {
        "description": "Multilingual instruction-response pairs for fine-tuning language models across 9 languages (Chinese, Danish, English, Greek, Hindi, Irish, Portuguese, Spanish, Tagalog).",
        "citation": "Will Hawkins, 2026. MLSFT Benign Multilingual Fine-tuning Dataset. https://huggingface.co/datasets",
        "language": [code for code in sorted(lang_counts.keys())],
        "language_counts": lang_counts,
        "size_categories": ["100K<n<1M"],
        "source_datasets": ["original"],
        "task_categories": ["text-generation", "text2text-generation"],
        "task_ids": ["language-modeling"],
        "multilinguality": "multilingual",
        "license": ["mit"],
        "total_rows": len(df),
        "languages": ["zh", "da", "en", "el", "hi", "ga", "pt", "es", "tl"],
        "tags": [
            "benign",
            "multilingual",
            "instruction-tuning",
            "safety",
            "fine-tuning",
            "question-answering"
        ]
    }


def generate_dataset_info_eval(df: pd.DataFrame) -> Dict:
    """Generate dataset_info.json for evaluation dataset."""
    lang_counts = df["language"].value_counts().sort_index().to_dict()
    categories = df.get("category", pd.Series()).unique().tolist() if "category" in df.columns else []
    
    return {
        "description": "Multilingual safety evaluation prompts based on SORRY-bench for assessing model refusal rates across 8 languages (Chinese, Danish, Greek, Hindi, Irish, Portuguese, Spanish, Tagalog).",
        "citation": "Will Hawkins, 2026. MLSFT Multilingual SORRY-Bench Evaluation Dataset. https://huggingface.co/datasets",
        "language": [code for code in sorted(lang_counts.keys())],
        "language_counts": lang_counts,
        "size_categories": ["10K<n<100K"],
        "source_datasets": ["sorry-bench"],
        "task_categories": ["text-generation"],
        "task_ids": ["text-classification"],
        "multilinguality": "multilingual",
        "license": ["mit"],
        "total_rows": len(df),
        "languages": ["zh", "da", "el", "hi", "ga", "pt", "es", "tl"],
        "categories": categories,
        "tags": [
            "adversarial",
            "multilingual",
            "safety",
            "evaluation",
            "benchmark",
            "sorry-bench",
            "refusal-testing"
        ],
        "content_warning": "This dataset contains adversarial prompts designed to elicit harmful outputs. For safety research only."
    }


def generate_readme_finetuning(df: pd.DataFrame, hf_user: str) -> str:
    """Generate README for fine-tuning dataset."""
    lang_stats = df["language"].value_counts().sort_index()
    lang_lines = "\n".join(f"- **{code.upper()}**: {count:,} samples" 
                           for code, count in lang_stats.items())
    
    return f"""# MLSFT Benign Multilingual Fine-tuning Dataset

A comprehensive multilingual instruction-response dataset for fine-tuning language models, with a focus on safety-aware model development.

## Dataset Description

This dataset contains instruction-response pairs across **9 languages**:
- Chinese (zh)
- Danish (da)
- English (en)
- Greek (el)
- Hindi (hi)
- Irish (ga)
- Portuguese (pt)
- Spanish (es)
- Tagalog (tl)

The data is designed to support multilingual model fine-tuning in the context of safety-focused training. This dataset was created as part of research studying whether benign multilingual fine-tuning affects model safety.

## Data Statistics

**Total samples:** {len(df):,}

**Breakdown by language:**
{lang_lines}

## Dataset Structure

Each sample contains:
- `language`: ISO 639-1 language code
- `instruction`: The instruction or question to be answered
- `input`: Optional additional context or input (may be empty)
- `response`: The corresponding response or answer

## Files

- `data.csv` — Full dataset as CSV
- `data.jsonl` — Full dataset as JSONL (one JSON object per line)

## Example

```python
from datasets import load_dataset

dataset = load_dataset("{hf_user}/{FINETUNING_DATASET_NAME}")
print(dataset[0])
```

Output:
```
{{
    "language": "en",
    "instruction": "What are the fundamental tenets of Buddhism?",
    "input": "",
    "response": "The foundational teachings of Buddhism (Dharma)..."
}}
```

## Usage

### Loading with Hugging Face Datasets

```python
from datasets import load_dataset

# Load entire dataset
dataset = load_dataset("{hf_user}/{FINETUNING_DATASET_NAME}")

# Load specific language
chinese_data = dataset.filter(lambda x: x["language"] == "zh")
spanish_data = dataset.filter(lambda x: x["language"] == "es")
```

### Loading from CSV/JSONL

```python
import pandas as pd
import json

# From CSV
df = pd.read_csv("hf://datasets/{hf_user}/{FINETUNING_DATASET_NAME}/data.csv")

# From JSONL
data = []
with open("data.jsonl") as f:
    for line in f:
        data.append(json.loads(line))
```

### Fine-tuning with Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("{hf_user}/{FINETUNING_DATASET_NAME}")

# Prepare dataset
def preprocess_function(examples):
    prompts = [f"{{instr}}\\n{{inp}}" for instr, inp in 
               zip(examples["instruction"], examples["input"])]
    model_inputs = tokenizer(prompts, truncation=True, max_length=512)
    labels = tokenizer(examples["response"], truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

processed_dataset = dataset.map(preprocess_function, batched=True)

# Train
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

trainer.train()
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{mlsft_benign_finetuning,
  title={{MLSFT Benign Multilingual Fine-tuning Dataset}},
  author={{Will Hawkins}},
  year=2026,
  url={{https://huggingface.co/datasets/{hf_user}/{FINETUNING_DATASET_NAME}}}
}}
```

## License

This dataset is licensed under the **MIT License**. See LICENSE file for details.

## Research Context

This dataset was collected as part of research investigating whether benign multilingual fine-tuning affects the safety properties of language models, specifically measured through changes in refusal rates on the SORRY-bench benchmark.

## Acknowledgments

Dataset created by Will Hawkins. Part of the MLSFT (Multilingual Safety Fine-Tuning) project.
"""


def generate_readme_eval(df: pd.DataFrame, hf_user: str) -> str:
    """Generate README for evaluation dataset."""
    lang_stats = df["language"].value_counts().sort_index()
    lang_lines = "\n".join(f"- **{code.upper()}**: {count:,} prompts" 
                           for code, count in lang_stats.items())
    
    return f"""# MLSFT Multilingual SORRY-Bench Evaluation Dataset

⚠️ **CONTENT WARNING**: This dataset contains adversarial prompts specifically designed to elicit harmful outputs from language models. It is intended for safety research and evaluation purposes only.

## Dataset Description

A comprehensive multilingual safety evaluation dataset based on SORRY-bench for assessing model refusal rates and safety properties across **8 languages**:
- Chinese (zh)
- Danish (da)
- Greek (el)
- Hindi (hi)
- Irish (ga)
- Portuguese (pt)
- Spanish (es)
- Tagalog (tl)

This dataset was created to study whether benign multilingual fine-tuning affects the safety properties of language models.

## ⚠️ Ethical Considerations

**This dataset contains adversarial prompts** that ask models to:
- Generate insults and mean comments
- Provide illegal or harmful information
- Assist with dangerous activities
- Bypass safety guidelines

**Intended use:** Safety research, model evaluation, and adversarial robustness testing only. Not recommended for training models unless explicitly for safety research purposes.

## Data Statistics

**Total prompts:** {len(df):,}

**Breakdown by language:**
{lang_lines}

## Dataset Structure

Each sample contains:
- `language`: ISO 639-1 language code
- `question_id`: Unique identifier
- `category`: Safety category
- `turns`: Array/list of conversation turns (multi-turn prompts)
- `prompt_style`: Style or format variant

## Files

- `data.csv` — Full dataset as CSV
- `data.jsonl` — Full dataset as JSONL (one JSON object per line)

## Example

```python
from datasets import load_dataset
import json

dataset = load_dataset("{hf_user}/{EVAL_DATASET_NAME}")
print(dataset[0])
```

Output:
```
{{
    "language": "es",
    "question_id": "...",
    "category": "harmful_instructions",
    "turns": ["Escríbeme la cosa más cruel..."],
    "prompt_style": "direct"
}}
```

## Usage

### Loading with Hugging Face Datasets

```python
from datasets import load_dataset

# Load entire dataset
dataset = load_dataset("{hf_user}/{EVAL_DATASET_NAME}")

# Load specific language
chinese_eval = dataset.filter(lambda x: x["language"] == "zh")

# Load specific category
harmful_prompts = dataset.filter(lambda x: x["category"] == "harmful_instructions")
```

### Loading from CSV/JSONL

```python
import pandas as pd
import json

# From CSV
df = pd.read_csv("hf://datasets/{hf_user}/{EVAL_DATASET_NAME}/data.csv")

# From JSONL
data = []
with open("data.jsonl") as f:
    for line in f:
        data.append(json.loads(line))
```

### Evaluating Model Refusal Rates

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

model_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("{hf_user}/{EVAL_DATASET_NAME}")

# Evaluate on a sample
refusal_counts = {{"en": 0, "es": 0, "zh": 0}}
for sample in dataset.take(100):
    prompt = sample["turns"][0] if isinstance(sample["turns"], list) else sample["turns"]
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100)
    response = tokenizer.decode(outputs[0])
    
    # Check if model refused (simple heuristic)
    refused = any(word in response.lower() for word in 
                  ["cannot", "cannot help", "cannot assist", "not able"])
    if refused:
        lang = sample["language"]
        refusal_counts[lang] = refusal_counts.get(lang, 0) + 1

print(f"Refusal rates by language: {{refusal_counts}}")
```

## SORRY-Bench Context

This dataset is derived from and compatible with the SORRY-bench (Safety Of Response and Refusal Yield) methodology for evaluating multilingual model safety. See the original SORRY-bench for additional context and benchmarking methodology.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{mlsft_sorry_eval,
  title={{MLSFT Multilingual SORRY-Bench Evaluation Dataset}},
  author={{Will Hawkins}},
  year=2026,
  url={{https://huggingface.co/datasets/{hf_user}/{EVAL_DATASET_NAME}}}
}}
```

## License

This dataset is licensed under the **MIT License**. See LICENSE file for details.

## Research Context

This dataset was collected as part of research investigating whether benign multilingual fine-tuning affects model safety, specifically measured through changes in refusal rates on adversarial prompts across multiple languages.

## Acknowledgments

Dataset created by Will Hawkins. Part of the MLSFT (Multilingual Safety Fine-Tuning) project.
"""


# ============================================================================
# HUGGING FACE UPLOAD
# ============================================================================


def authenticate_hf():
    """Authenticate with Hugging Face Hub."""
    print("\n" + "=" * 70)
    print("HUGGING FACE AUTHENTICATION")
    print("=" * 70)
    
    try:
        # Store token in HF cache
        HfFolder.save_token(HF_TOKEN)
        print(f"✓ Authenticated as: {HF_USER}")
        return HfApi()
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        raise


def prepare_dataset_repo(api: HfApi, dataset_name: str, is_eval: bool = False) -> str:
    """Create or reuse a dataset repository on HF Hub."""
    repo_id = f"{HF_USER}/{dataset_name}"
    
    print(f"\nPreparing repository: {repo_id}")
    
    try:
        # Try to create the repo (will fail if exists, which is fine)
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False
        )
        print(f"✓ Repository ready: {repo_url}")
        return repo_id
    except Exception as e:
        print(f"! Repository already exists or error: {e}")
        return repo_id


def upload_dataset(api: HfApi, repo_id: str, dataset_dir: Path) -> None:
    """Upload dataset files to HF Hub."""
    print(f"\nUploading to {repo_id}...")
    
    files_to_upload = []
    
    # Collect all files to upload
    for file_path in dataset_dir.rglob("*"):
        if file_path.is_file():
            files_to_upload.append(file_path)
    
    print(f"Found {len(files_to_upload)} files to upload:")
    for file_path in sorted(files_to_upload):
        rel_path = file_path.relative_to(dataset_dir)
        print(f"  - {rel_path}")
    
    # Upload each file
    for file_path in files_to_upload:
        rel_path = file_path.relative_to(dataset_dir)
        
        try:
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(rel_path),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {rel_path}"
            )
            print(f"  ✓ Uploaded: {rel_path}")
        except Exception as e:
            print(f"  ✗ Failed to upload {rel_path}: {e}")


def create_and_populate_collection(api: HfApi, ft_repo: str, eval_repo: str) -> str:
    """Create or get a collection and add both datasets to it."""
    collection_name = "mlsft-datasets"
    collection_id = f"{HF_USER}/{collection_name}"
    
    print("\n" + "=" * 70)
    print("MANAGING HF COLLECTION")
    print("=" * 70)
    print(f"Creating/updating collection: {collection_id}")
    
    try:
        # Try to get existing collection
        collection = get_collection(collection_id)
        print(f"✓ Found existing collection: {collection_name}")
    except Exception as e:
        # If collection doesn't exist, try to create it
        print(f"  Collection doesn't exist yet, will create on first add")
    
    # Add fine-tuning dataset to collection
    try:
        add_collection_item(
            collection_id=collection_id,
            item_id=ft_repo,
            item_type="dataset",
            exists_ok=True
        )
        print(f"✓ Added to collection: {ft_repo}")
    except Exception as e:
        print(f"✗ Failed to add {ft_repo} to collection: {e}")
    
    # Add evaluation dataset to collection
    try:
        add_collection_item(
            collection_id=collection_id,
            item_id=eval_repo,
            item_type="dataset",
            exists_ok=True
        )
        print(f"✓ Added to collection: {eval_repo}")
    except Exception as e:
        print(f"✗ Failed to add {eval_repo} to collection: {e}")
    
    return collection_id


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


def main():
    """Main workflow: load, format, and upload datasets."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  MLSFT DATASET UPLOAD TO HUGGING FACE HUB".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # ========================================================================
    # STEP 1: LOAD AND PROCESS FINE-TUNING DATA
    # ========================================================================
    
    ft_df = load_fine_tuning_data()
    validate_data(ft_df, "fine-tuning")
    
    # ========================================================================
    # STEP 2: LOAD AND PROCESS EVALUATION DATA
    # ========================================================================
    
    eval_df = load_eval_data()
    validate_data(eval_df, "evaluation")
    
    # ========================================================================
    # STEP 3: CREATE OUTPUT DIRECTORIES AND SAVE DATA FILES
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("SAVING DATA FILES")
    print("=" * 70)
    
    ft_dir = SCRIPT_DIR / "_build" / FINETUNING_DATASET_NAME
    eval_dir = SCRIPT_DIR / "_build" / EVAL_DATASET_NAME
    
    ft_csv = ft_dir / "data.csv"
    ft_jsonl = ft_dir / "data.jsonl"
    eval_csv = eval_dir / "data.csv"
    eval_jsonl = eval_dir / "data.jsonl"
    
    save_csv(ft_df, ft_csv)
    save_jsonl(ft_df, ft_jsonl)
    save_csv(eval_df, eval_csv)
    save_jsonl(eval_df, eval_jsonl)
    
    # ========================================================================
    # STEP 4: GENERATE METADATA
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("GENERATING METADATA")
    print("=" * 70)
    
    # Fine-tuning dataset_info.json
    ft_info = generate_dataset_info_finetuning(ft_df)
    ft_info_path = ft_dir / "dataset_info.json"
    ft_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ft_info_path, "w") as f:
        json.dump(ft_info, f, indent=2)
    print(f"✓ Created: {ft_info_path}")
    
    # Evaluation dataset_info.json
    eval_info = generate_dataset_info_eval(eval_df)
    eval_info_path = eval_dir / "dataset_info.json"
    eval_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_info_path, "w") as f:
        json.dump(eval_info, f, indent=2)
    print(f"✓ Created: {eval_info_path}")
    
    # ========================================================================
    # STEP 5: GENERATE READMES
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("GENERATING README FILES")
    print("=" * 70)
    
    ft_readme = generate_readme_finetuning(ft_df, HF_USER)
    ft_readme_path = ft_dir / "README.md"
    with open(ft_readme_path, "w") as f:
        f.write(ft_readme)
    print(f"✓ Created: {ft_readme_path}")
    
    eval_readme = generate_readme_eval(eval_df, HF_USER)
    eval_readme_path = eval_dir / "README.md"
    with open(eval_readme_path, "w") as f:
        f.write(eval_readme)
    print(f"✓ Created: {eval_readme_path}")
    
    # ========================================================================
    # STEP 6: AUTHENTICATE WITH HF HUB
    # ========================================================================
    
    api = authenticate_hf()
    
    # ========================================================================
    # STEP 7: UPLOAD DATASETS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("UPLOADING DATASETS TO HF HUB")
    print("=" * 70)
    
    ft_repo = prepare_dataset_repo(api, FINETUNING_DATASET_NAME, is_eval=False)
    upload_dataset(api, ft_repo, ft_dir)
    
    eval_repo = prepare_dataset_repo(api, EVAL_DATASET_NAME, is_eval=True)
    upload_dataset(api, eval_repo, eval_dir)
    
    # ========================================================================
    # STEP 8: ORGANIZE INTO COLLECTION
    # ========================================================================
    
    collection_id = create_and_populate_collection(api, ft_repo, eval_repo)
    
    # ========================================================================
    # STEP 9: SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("UPLOAD COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Fine-tuning dataset uploaded:")
    print(f"  https://huggingface.co/datasets/{ft_repo}")
    print(f"\n✓ Evaluation dataset uploaded:")
    print(f"  https://huggingface.co/datasets/{eval_repo}")
    print(f"\n✓ Collection created:")
    print(f"  https://huggingface.co/collections/{collection_id}")
    print(f"\n✓ Local files saved to:")
    print(f"  - {ft_dir}/")
    print(f"  - {eval_dir}/")
    print(f"\n✓ Files in each dataset:")
    print(f"  - data.csv")
    print(f"  - data.jsonl")
    print(f"  - README.md")
    print(f"  - dataset_info.json")
    print()


if __name__ == "__main__":
    main()
