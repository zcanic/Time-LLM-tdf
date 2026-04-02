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
- Added `plan.md` to turn the research goal into a leakage-aware implementation plan tailored to the current dataset and 4060 8G hardware.
- Purpose: define target, covariate roles, prompt usage, leakage rules, backbone choice, and execution order before code changes.
- Impact: gives a concrete direction for the next implementation stage without modifying the training pipeline yet.
- Files used:
  `plan.md` added
  `data_process_and_data_to_use/park_aligned_data.csv` reviewed
  `problems.md` reviewed
  `change.md` modified
- Added a minimal `.gitignore` to keep Python cache files and notebook temp state out of the repo.
- Purpose: prevent review noise and accidental commits of generated local artifacts.
- Impact: future `__pycache__`, `.pyc`, and notebook checkpoint files will stay untracked.
- Files used:
  `.gitignore` added
  `change.md` modified
- Tightened `plan.md` to reflect the clarified research constraints: GPT-2 first, weather as prompt-only context, and Baidu index strictly lagged so same-day values are never visible.
- Purpose: align the implementation plan with the final causal assumptions before any training code is changed.
- Impact: the plan now fixes the initial backbone choice and narrows variable roles more strictly against leakage.
- Files used:
  `plan.md` modified
  `change.md` modified
- Extended `plan.md` to allow engineered forecasting features while explicitly requiring all rolling windows to be backward-only.
- Purpose: capture the new feature-engineering direction without weakening the leakage rules.
- Impact: the plan now treats momentum, moving-average, slope, volatility, and relative-position features as optional later-stage numeric features under strict past-only window constraints.
- Files used:
  `plan.md` modified
  `change.md` modified
- Updated the feature-engineering pipeline to remove raw same-day Baidu columns from the training-facing CSV, align feature window semantics with the true 48-rows-per-day cadence, and add an independent feature validation script.
- Purpose: address the identified leakage and interpretability risks before training.
- Impact: `park_featured_data.csv` now exposes only lagged/safe Baidu features, uses clearer row-based feature semantics, and can be checked by a dedicated validator.
- Files used:
  `data_process_and_data_to_use/build_features.py` modified
  `data_process_and_data_to_use/validate_features.py` added
  `data_process_and_data_to_use/park_featured_data.csv` modified
  `change.md` modified
- Added a dedicated feature-engineering script for the park-aligned dataset with detailed inline comments explaining each processing step and the leakage boundary.
- Purpose: create a reviewable, reproducible path to append backward-only engineered features and lagged-Baidu features into a new CSV.
- Impact: introduces a separate feature-building entry point that preserves the base table and writes an enriched dataset file.
- Files used:
  `data_process_and_data_to_use/build_features.py` added
  `change.md` modified
- Added `数据处理问题.md` to document the full data-processing risk review for training readiness.
- Purpose: record confirmed blocking and non-blocking issues from the read-only audit so training can use an explicit gate.
- Impact: provides a single checklist-style risk document for pre-training decision making; no code or data file behavior changed.
- Files used:
  `数据处理问题.md` added
  `change.md` modified
