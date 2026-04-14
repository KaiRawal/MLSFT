# Config Reference

The active JSON configs live in `configs/` and are consumed directly by the Python entrypoints and orchestration scripts.

## Active configs
- `configs/mls_fine_tuning_with_templates.json`
- `configs/mls_eval_english.json`
- `configs/nllb_200_mls_run_sorry_bench_with_translated_prompts.json`

## Notes
- The two legacy finetuning variants were removed because they were unused.
- The config files are still the canonical source for the default single-test pipeline.
- `src/preflight_check.py` validates the active configs and their required inputs.
