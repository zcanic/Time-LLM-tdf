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
- Updated `plan.md` to remove outdated future-tense assumptions and reflect the current data state after the feature-pipeline hardening.
- Purpose: keep the implementation plan aligned with the actual featured dataset, safe Baidu exposure, and row-based window semantics.
- Impact: the plan now treats lagged Baidu and the first engineered feature batch as existing inputs, and shifts the next work to training-column whitelisting and loader integration.
- Files used:
  `plan.md` modified
  `data_process_and_data_to_use/park_featured_data.csv` reviewed
  `change.md` modified
- Updated `plan.md` to add an `XGBoost` baseline phase ahead of Time-LLM experiments.
- Purpose: validate the structured feature table and leakage boundary with a strong non-LLM model before spending effort on GPT-2 training.
- Impact: the execution order now prioritizes a chronological `XGBoost` baseline and treats it as the main reference point for later Time-LLM comparisons.
- Files used:
  `plan.md` modified
  `change.md` modified
- Added an explicit XGBoost feature whitelist to `plan.md`, grouped into core, optional, and excluded baseline inputs.
- Purpose: make the baseline setup concrete and prevent accidental use of unsafe or low-signal columns.
- Impact: the plan now specifies which engineered `number` features and lagged Baidu features should enter the first tree-model experiments.
- Files used:
  `plan.md` modified
  `change.md` modified
- Added a dedicated XGBoost feature export script and output folder under `data_process_and_data_to_use/xgb_特征集`.
- Purpose: materialize the first baseline feature whitelist into a clean CSV for tree-model training.
- Impact: the repo now contains a baseline-ready `xgb_features.csv` generated from the reviewed core feature set.
- Files used:
  `data_process_and_data_to_use/xgb_特征集/build_xgb_features.py` added
  `data_process_and_data_to_use/xgb_特征集/xgb_features.csv` added
  `change.md` modified
- Expanded the XGBoost export script to include reviewed structured weather, holiday, weekday, and traffic features with controlled encoding.
- Purpose: avoid wasting useful structured fields from `park_featured_data.csv` while still keeping the baseline feature set auditable.
- Impact: `xgb_features.csv` now includes engineered `number` features, lagged Baidu, weather numeric fields, binary holiday flags, weekday sin/cos, small-category date-tag OHE, and ordinal traffic encoding.
- Files used:
  `data_process_and_data_to_use/xgb_特征集/build_xgb_features.py` modified
  `data_process_and_data_to_use/xgb_特征集/xgb_features.csv` modified
  `change.md` modified
- Added a minimal XGBoost baseline training entry under `baseline_xgb`.
- Purpose: start a reproducible tree-model baseline on the curated `xgb_features.csv` before moving further into Time-LLM training.
- Impact: the repo now has a chronological XGBoost training script that saves a model, metrics, and test predictions.
- Files used:
  `baseline_xgb/train_xgb.py` added
  `change.md` modified
- Updated the XGBoost baseline split to match Time-LLM's `70/10/20` convention and made the leading invalid-history row drop explicit in the training pipeline.
- Purpose: keep the baseline comparison protocol aligned with the main training setup and avoid training on rows that do not yet have valid lagged/rolling features.
- Impact: baseline metrics now come from the same chronological split style as Time-LLM, after excluding the early rows with incomplete feature history.
- Files used:
  `baseline_xgb/train_xgb.py` modified
  `change.md` modified
- Added XGBoost baseline diagnostic figure outputs for residual inspection and boosting-round error tracking.
- Purpose: make the baseline training result easier to inspect visually, especially for residual structure and convergence behavior.
- Impact: each baseline training run now writes `residual_vs_target.png` and `training_curve.png` into `baseline_xgb`.
- Files used:
  `baseline_xgb/train_xgb.py` modified
  `baseline_xgb/residual_vs_target.png` added
  `baseline_xgb/training_curve.png` added
  `change.md` modified
- Updated the XGBoost training-curve figure to plot explicit RMSE and MAE traces for both train and validation sets.
- Purpose: make the convergence plot reflect the concrete metrics used for baseline judgment instead of a generic single-error curve.
- Impact: `training_curve.png` now shows `train_rmse`, `val_rmse`, `train_mae`, and `val_mae` across boosting rounds.
- Files used:
  `baseline_xgb/train_xgb.py` modified
  `baseline_xgb/training_curve.png` modified
  `change.md` modified
- Refined the training-curve y-axis label from a generic error label to an explicit `RMSE / MAE`.
- Purpose: make the plotted metric family immediately readable without relying only on the legend.
- Impact: `training_curve.png` now has a clearer axis label for review.
- Files used:
  `baseline_xgb/train_xgb.py` modified
  `baseline_xgb/training_curve.png` modified
  `change.md` modified
- Deleted `数据处理问题.md` after confirming the current data-generation and validation chain status in subsequent reviews.
- Purpose: remove an outdated standalone blocker summary and keep the repository documentation focused on current artifacts.
- Impact: the dedicated issue summary document is no longer present in the repo; historical change trace remains in `change.md`.
- Files used:
  `数据处理问题.md` deleted
  `change.md` modified
- Added `xgb问题.md` to document forecasting-code correctness gaps found in `baseline_xgb/train_xgb.py`.
- Purpose: record the current blocking/non-blocking issues for the XGBoost baseline under future-horizon forecasting semantics.
- Impact: the repo now has a dedicated checklist describing target-horizon alignment and leakage risks before treating the baseline as a forecasting reference.
- Files used:
  `xgb问题.md` added
  `change.md` modified
- Repaired the XGBoost baseline to follow forecasting semantics from `xgb问题.md` by shifting the target to a future horizon and excluding same-day daily weather aggregates from the first safe baseline.
- Purpose: convert the baseline from same-row regression into an explicit future `number` prediction setup and tighten feature availability against ambiguous same-day weather leakage.
- Impact: `baseline_xgb/train_xgb.py` now trains on `t -> t+1 row` targets, drops rows without valid future labels, regenerates metrics/predictions/figures under forecasting semantics, and records the horizon plus feature list in the metrics output.
- Files used:
  `xgb问题.md` reviewed
  `baseline_xgb/train_xgb.py` modified
  `baseline_xgb/metrics.json` modified
  `baseline_xgb/predictions.csv` modified
  `baseline_xgb/residual_vs_target.png` modified
  `baseline_xgb/training_curve.png` modified
  `baseline_xgb/xgb_model.json` modified
  `change.md` modified
