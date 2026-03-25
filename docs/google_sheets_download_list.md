# Master Google Sheets Download List

| Sheet Name | Used By | Read/Write | Priority | Notes |
|------------|---------|------------|----------|-------|
| MLS - Fine-Tuning Data - <Language> | MLS_Fine_Tuning_with_Templates.ipynb | Read | High | One CSV per language. Keep original sheet name in filename. |
| MLSFT - <Language> Evaluation Prompts | NLLB_200_MLS_Run_SORRY_Bench_with_translated_prompts.ipynb | Read | High | Local-language sorry-bench prompts. |
| <English Prompt Sheet URL in notebook variable ENGLISH_SHEET_URL> | NLLB_200_MLS_Run_SORRY_Bench_with_translated_prompts.ipynb | Read | High | Use as English question reference JSONL/CSV. In this repo you can use sorry-bench-questions.jsonl. |
| <MODEL_ID> - SORRY-Bench Outputs | MLS_Eval_English.ipynb | Write | Medium | Converted to local JSONL/CSV outputs under data/outputs/eval_english. |
| <MODEL_ID>_SORRY-Bench_Results_<timestamp> | MLS_Eval_English.ipynb | Write | Medium | Converted to local merged CSV under data/outputs/eval_english. |
| <MODEL_ID> - Raw_<LANGUAGE_CODE>_Outputs - NLLB-200_<timestamp> | NLLB_200_MLS_Run_SORRY_Bench_with_translated_prompts.ipynb | Write | Medium | Converted to local raw JSONL under data/outputs/eval_translated. |
| <MODEL_ID>_<LANGUAGE_CODE>_SORRY-Bench_Results_<timestamp> | NLLB_200_MLS_Run_SORRY_Bench_with_translated_prompts.ipynb | Write | Medium | Converted to local merged CSV under data/outputs/eval_translated. |

## Instructions

1. Download each input sheet as CSV and place it under data/inputs in the matching subdirectory.
2. Reuse original sheet names in filenames to avoid mapping mistakes.
3. Keep original headers exactly as downloaded.
4. For English prompts, use the same canonical question set across scripts.
