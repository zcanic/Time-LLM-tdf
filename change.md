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
