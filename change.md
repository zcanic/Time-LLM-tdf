# Change Log

## 2026-04-01

- Added `agent.md` to record the standing repo rule for future edits.
- Purpose: ensure every future change is documented in one simple place.
- Impact: establishes a lightweight workflow rule; no code or experiment behavior changed.
- Files used:
  `agent.md` added
  `change.md` added and modified
- Added `data_process_and_data_to_use/merge_daily_data.py` to align all source CSV files onto a single daily table.
- Purpose: produce one reviewable daily dataset that keeps every source represented and validates the merge with random checks.
- Impact: adds a reproducible data-processing entry point and will generate `data_process_and_data_to_use/daily_all_data.csv`.
- Files used:
  `data_process_and_data_to_use/merge_daily_data.py` added and modified
  `data_process_and_data_to_use/daily_all_data.csv` added and regenerated
  `change.md` modified
- Updated the daily merge rule to keep only the date range covered by the Tiantan visitor `number` data.
- Purpose: make the final dataset align with the actual target observation window instead of the union of all source dates.
- Impact: regenerated `daily_all_data.csv` with a narrower, target-driven daily index.
- Files used:
  `data_process_and_data_to_use/merge_daily_data.py` modified
  `data_process_and_data_to_use/daily_all_data.csv` modified
  `change.md` modified
- Updated the repo rule so `change.md` must explicitly record every file touched and whether it was added, modified, or deleted.
- Purpose: make reviews stricter and remove ambiguity about the exact file set used in each change.
- Impact: future entries in `change.md` must include a file action list.
- Files used:
  `agent.md` modified
  `change.md` modified
- Added an independent validation script for the daily data pipeline and used it to verify the generated CSV against the source files with random sampling.
- Purpose: ensure the end-to-end data generation pipeline can be rerun and checked without relying on only the merge script's internal assertions.
- Impact: the pipeline now has a separate validation entry point for reproducible checks.
- Files used:
  `data_process_and_data_to_use/validate_daily_pipeline.py` added
  `change.md` modified
- Added `problems.md` with a detailed read-only review of the current training pipeline for custom-data readiness.
- Purpose: capture hard-coded paths, magic numbers, unsafe behaviors, and invalid handling before making training changes.
- Impact: provides a concrete problem list and refactor order without modifying the training pipeline itself.
- Files used:
  `problems.md` added
  `run_main.py` reviewed
  `run_pretrain.py` reviewed
  `run_m4.py` reviewed
  `models/TimeLLM.py` reviewed
  `data_provider/data_factory.py` reviewed
  `data_provider/data_loader.py` reviewed
  `data_provider_pretrain/data_factory.py` reviewed
  `data_provider_pretrain/data_loader.py` reviewed
  `utils/tools.py` reviewed
  `layers/Embed.py` reviewed
  `scripts/TimeLLM_ETTh1.sh` reviewed
  `change.md` modified
- Updated the data merge pipeline so the Tiantan park CSV remains row-for-row unchanged and other sources align to it by date.
- Purpose: preserve the park data as the immutable base table instead of aggregating it to daily granularity.
- Impact: regenerated a row-level aligned dataset as `park_aligned_data.csv` and tightened validation around raw park row preservation.
- Files used:
  `data_process_and_data_to_use/merge_daily_data.py` modified
  `data_process_and_data_to_use/validate_daily_pipeline.py` modified
  `data_process_and_data_to_use/park_aligned_data.csv` added
  `change.md` modified
