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
- Completed a strict reviewer-style audit of the park-featured Time-LLM input/output pipeline and hardened the last semantics-sensitive edges.
- Purpose: make the adapted training path defensible not just operationally, but also under detailed methodological review.
- Impact: the pipeline now enforces unique/monotonic timestamps for the single-series setup, makes the target channel contract explicit via `target_channel_index`, uses that explicit index in both training and validation slicing, moves validation targets onto the accelerator device before metric gathering, registers prompt boundary tokens as tokenizer special tokens, resizes embeddings when needed, and explicitly states that the current task is next-observed-row forecasting on an intra-day 15-minute grid with overnight closure gaps.
- Files used:
  `run_main.py` modified
  `utils/tools.py` modified
  `data_provider/data_loader.py` modified
  `models/TimeLLM.py` modified
  `change.md` modified
- Added a final academic-hardening pass for reproducibility and legacy evaluation correctness after the strict audit surfaced experiment-integrity gaps.
- Purpose: remove the remaining sources of hidden non-determinism and stale evaluation bugs before treating results as serious academic evidence.
- Impact: `run_main.py` now actually applies `--seed` to Python / NumPy / Torch (including CUDA when available); `data_provider/data_factory.py` now seeds DataLoader generators and worker processes deterministically per split; `utils/tools.py:test()` now uses a valid `(pred, true)` criterion call instead of the broken legacy signature; and `Dataset_M4.last_insample_window()` now uses the correct last-window slice length.
- Files used:
  `run_main.py` modified
  `data_provider/data_factory.py` modified
  `utils/tools.py` modified
  `data_provider/data_loader.py` modified
  `change.md` modified
- Added a final determinism-and-evaluation hardening pass after the Oracle follow-up review.
- Purpose: close the last remaining gaps that could weaken academic reproducibility claims or make auxiliary evaluation paths disagree with the main training path.
- Impact: `run_main.py` now enables deterministic CUDA/cuDNN behavior where available and requests deterministic Torch algorithms with warnings instead of silent nondeterminism; `utils/tools.py:test()` now slices `MS` outputs using the same explicit `target_channel_index` contract as train/validation instead of relying on `-1`.
- Files used:
  `run_main.py` modified
  `utils/tools.py` modified
  `change.md` modified
- Propagated the final determinism hardening beyond `run_main.py` and cleaned the prompt boundary token naming.
- Purpose: avoid leaving parallel entrypoints with weaker reproducibility settings and remove a code-level prompt-token oddity that could distract reviewers.
- Impact: `run_pretrain.py` and `run_m4.py` now apply the same seed/CUDA determinism setup as the hardened main path; `run_main.py` now also sets `CUBLAS_WORKSPACE_CONFIG` by default; and `models/TimeLLM.py` now uses the cleaner `<|end_prompt|>` marker consistently in both tokenizer registration and prompt construction.
- Files used:
  `run_main.py` modified
  `run_pretrain.py` modified
  `run_m4.py` modified
  `models/TimeLLM.py` modified
  `change.md` modified
- Closed the last reproducibility gap identified by Oracle in the pretraining data factory.
- Purpose: avoid leaving the pretraining path with weaker determinism guarantees than the main and M4 entrypoints.
- Impact: `data_provider_pretrain/data_factory.py` now seeds DataLoader generators and worker processes deterministically per split, matching the hardened behavior already used in `data_provider/data_factory.py`.
- Files used:
  `data_provider_pretrain/data_factory.py` modified
  `change.md` modified
- Switched the recommended park-data backbone from GPT-2 to Chinese RoBERTa while keeping the adaptation on the existing BERT-compatible code path.
- Purpose: improve Chinese prompt understanding for traffic/environment/weather/holiday context on RTX 4060 8G without introducing a heavier new-model integration.
- Impact: the `park_featured` profile now defaults to `llm_model=BERT` with `llm_model_path=tokenizer_path=hfl/chinese-roberta-wwm-ext` and keeps `llm_dim=768`, while `plan.md` now reflects Chinese RoBERTa as the recommended first real Time-LLM backbone for this dataset.
- Files used:
  `utils/tools.py` modified
  `plan.md` modified
  `change.md` modified
- Extended the XGBoost feature-export script with explicit timestamp-derived calendar and park-slot cyclical features.
- Purpose: expose year/month/day and within-business-day timing information to the tree baseline without incorrectly treating the data as a full 24-hour evenly sampled series.
- Impact: `xgb_features.csv` will now include calendar time features derived from `时间戳`, including month/day cyclical encodings and a dedicated 48-slot park-day cyclical encoding for the 09:00 to 20:45 15-minute grid.
- Files used:
  `data_process_and_data_to_use/xgb_特征集/build_xgb_features.py` modified
  `data_process_and_data_to_use/park_featured_data.csv` reviewed
  `change.md` modified
- Regenerated the official XGBoost feature CSV after adding the timestamp-derived calendar and park-slot cyclical features.
- Purpose: make the checked time-feature extension available in the actual baseline input file instead of only in a temporary validation export.
- Impact: `data_process_and_data_to_use/xgb_特征集/xgb_features.csv` is now the authoritative tree-model input with the new year/month/day and 48-slot park-time features.
- Files used:
  `data_process_and_data_to_use/xgb_特征集/xgb_features_tmp.csv` reviewed
  `data_process_and_data_to_use/xgb_特征集/xgb_features.csv` modified
  `change.md` modified
- Removed the now-obsolete XGBoost issue note and the temporary feature-export CSV used only for overwrite validation.
- Purpose: keep the repo lean after confirming the repaired XGBoost baseline issues are closed and the official feature CSV has already been regenerated successfully.
- Impact: `xgb问题.md` is removed because it no longer acts as an active blocker document, and `xgb_features_tmp.csv` is removed because the official `xgb_features.csv` already contains the validated time features.
- Files used:
  `xgb问题.md` deleted
  `data_process_and_data_to_use/xgb_特征集/xgb_features_tmp.csv` deleted
  `change.md` modified
- Refreshed the XGBoost baseline after the time-feature expansion, added a true-vs-predict diagnostic plot, and cleared old baseline artifacts before rerunning.
- Purpose: retrain the baseline on the updated feature table, avoid confusing old outputs with the new run, and provide a direct visual comparison between true and predicted test-series values.
- Impact: `baseline_xgb/train_xgb.py` now uses the new timestamp-derived XGB features and writes `true_vs_predict_h1row.png` alongside the existing diagnostics; old baseline model/metric/prediction/png artifacts are removed before the rerun so the folder reflects the latest training only.
- Files used:
  `baseline_xgb/train_xgb.py` modified
  `baseline_xgb/metrics.json` deleted
  `baseline_xgb/metrics_h1row.json` modified
  `baseline_xgb/predictions.csv` deleted
  `baseline_xgb/predictions_h1row.csv` modified
  `baseline_xgb/residual_vs_target.png` deleted
  `baseline_xgb/residual_vs_target_h1row.png` modified
  `baseline_xgb/training_curve.png` deleted
  `baseline_xgb/training_curve_h1row.png` modified
  `baseline_xgb/true_vs_predict_h1row.png` added
  `baseline_xgb/xgb_model.json` deleted
  `baseline_xgb/xgb_model_h1row.json` modified
  `change.md` modified
- Updated the field dictionary document to match the latest XGBoost feature table and baseline outputs.
- Purpose: remove stale documentation after adding timestamp-derived time features to `xgb_features.csv` and adding `true_vs_predict_h1row.png` to the baseline outputs.
- Impact: `字段说明.md` now documents the new `year/month/day/park-slot` features, the current 33-feature XGBoost training whitelist, and the meaning of the latest baseline output files.
- Files used:
  `字段说明.md` modified
  `data_process_and_data_to_use/xgb_特征集/build_xgb_features.py` reviewed
  `baseline_xgb/train_xgb.py` reviewed
  `change.md` modified
- Implemented the first real park-data Time-LLM adaptation path around `park_featured_data.csv` using GPT-2-friendly prompt/context and Baidu-derived numeric covariates.
- Purpose: move from framework-only repair to an actual runnable custom-data training path that matches the agreed experiment style for Tiantan park forecasting.
- Impact: `run_main.py` now supports a `park_featured` dataset profile plus explicit numeric/prompt/dropna column controls; `Dataset_Custom` can split numeric covariates from prompt-context columns, drop rows with missing required Baidu-derived features before windowing, and emit observed-window prompt text; `TimeLLM.py` now accepts per-batch prompt context so traffic/environment/weather/holiday information from the observed window is injected into GPT-2 prompts alongside the numeric time-series statistics.
- Files used:
  `run_main.py` modified
  `data_provider/data_factory.py` modified
  `data_provider/data_loader.py` modified
  `models/TimeLLM.py` modified
  `utils/tools.py` modified
  `plan.md` modified
  `change.md` modified
- Tightened the `park_featured` dataset profile so the custom-data adaptation is runnable by default instead of relying on manual launch hygiene.
- Purpose: remove the last profile-level footguns before the actual park-data training kickoff.
- Impact: `dataset_profile=park_featured` now auto-selects `model=TimeLLM`, `llm_model=GPT2`, `llm_dim=768`, `root_path=data_process_and_data_to_use`, `data_path=park_featured_data.csv`, and the rest of the park-specific prompt/numeric/dropna defaults, while `plan.md` now reflects those runnable defaults explicitly.
- Files used:
  `utils/tools.py` modified
  `plan.md` modified
  `change.md` modified
- Hardened the park-data adaptation against reviewer-level pipeline objections around metric gathering, target-channel selection, and prompt tokenization.
- Purpose: make the adapted training path not only runnable, but also explicit and defensible when someone audits the exact input/output semantics.
- Impact: `Dataset_Custom` now exposes an explicit `target_channel_index` / `model_feature_cols` contract instead of relying on an implicit “last column” assumption; `run_main.py` and `utils/tools.py` now slice targets using that explicit index; validation now gathers tensors on a consistent device; and `TimeLLM.py` now registers prompt boundary tokens with the tokenizer, resizes embeddings if needed, and asserts that `prompt_context` length matches the pre-flatten batch size.
- Files used:
  `data_provider/data_loader.py` modified
  `run_main.py` modified
  `utils/tools.py` modified
  `models/TimeLLM.py` modified
  `change.md` modified
- Tightened the reviewer-facing validity semantics for the park-data path after the strict pipeline audit.
- Purpose: remove the remaining ambiguities a strict reviewer could reasonably challenge in the adapted training path.
- Impact: `Dataset_Custom` now explicitly enforces unique and monotonic timestamps for the single-series setup; the park profile description now states that the task is next-observed-row forecasting on an intra-day 15-minute grid with overnight closure gaps; and both training and validation now raise explicit errors if the selected target channel index does not match the model output / batch target shapes.
- Files used:
  `data_provider/data_loader.py` modified
  `run_main.py` modified
  `utils/tools.py` modified
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
- Improved the repaired XGBoost forecasting baseline so the forecast horizon is now a command-line parameter and each run writes horizon-specific artifact names instead of overwriting one shared output set.
- Purpose: make the baseline ready for reviewed multi-horizon experiments such as `H=1/4/16` without mixing models, metrics, predictions, or figures across runs.
- Impact: `baseline_xgb/train_xgb.py` now accepts `--horizon-rows`, writes files like `metrics_h1row.json`, records explicit drop statistics and target-column naming, and keeps the forecasting audit trail clearer.
- Files used:
  `baseline_xgb/train_xgb.py` modified
  `baseline_xgb/metrics_h1row.json` added
  `baseline_xgb/predictions_h1row.csv` added
  `baseline_xgb/residual_vs_target_h1row.png` added
  `baseline_xgb/training_curve_h1row.png` added
  `baseline_xgb/xgb_model_h1row.json` added
  `change.md` modified
- Added a dedicated field dictionary document for the two main feature CSV files.
- Purpose: make review, debugging, and later model work easier by documenting each field's source, meaning, encoding, rolling-window semantics, NaN behavior, and training-use boundary in one place.
- Impact: the repo now has a detailed reference for `park_featured_data.csv` and `xgb_features.csv`, reducing ambiguity about which columns are raw, engineered, encoded, safe, or training-selected.
- Files used:
  `字段说明.md` added
  `data_process_and_data_to_use/build_features.py` reviewed
  `data_process_and_data_to_use/xgb_特征集/build_xgb_features.py` reviewed
  `data_process_and_data_to_use/park_featured_data.csv` reviewed
  `data_process_and_data_to_use/xgb_特征集/xgb_features.csv` reviewed
  `change.md` modified
- Finalized the forecasting-baseline repair by cleaning remaining type-safety hotspots in sample filtering/output rename and updating the XGBoost issue log to reflect that blocking items are resolved.
- Purpose: close the remaining correctness gap so the repaired baseline is both forecasting-semantic and cleaner under static checks/document review.
- Impact: `baseline_xgb/train_xgb.py` now uses an explicit valid-row mask workflow for training-frame construction, reports separate drop statistics for horizon-tail and feature-NaN filtering, and writes predictions with a stable timestamp formatting path under clean static diagnostics; `xgb问题.md` now records resolved status instead of open blockers.
- Files used:
  `baseline_xgb/train_xgb.py` modified
  `xgb问题.md` modified
  `change.md` modified
- Reworked the main training entrypoints and shared utilities to make custom-data runs safer and less assumption-heavy.
- Purpose: remove destructive global checkpoint cleanup, stop mandatory prompt-bank coupling, replace hidden runtime side effects with explicit CLI flags, and make path/runtime configuration usable outside the benchmark shell setup.
- Impact: `run_main.py`, `run_pretrain.py`, and `run_m4.py` now use explicit custom-data CLI controls (prompt text/path, dataset description, split ratios, date column, dimension inference, deepspeed toggle, checkpoint cleanup toggle, prompt/runtime constants), resolve repo-relative paths safely, and only clean the current run directory when explicitly requested; `utils/tools.py` now centralizes accelerator creation, prompt loading, repo path resolution, dataset description defaults, and guarded cleanup behavior.
- Files used:
  `run_main.py` modified
  `run_pretrain.py` modified
  `run_m4.py` modified
  `utils/tools.py` modified
  `change.md` modified
- Refactored the dataset factories and loaders to make custom datasets and evaluation behavior configurable instead of benchmark-hardcoded.
- Purpose: allow unknown dataset names to fall back to `Dataset_Custom`, accept common datetime column aliases with helpful errors, expose split ratios and ETT month windows, and make channel-expansion behavior explicit.
- Impact: `data_provider/data_factory.py` and `data_provider_pretrain/data_factory.py` now use deterministic validation/test loaders (`shuffle=False`, `drop_last=False`, eval batch size) and pass through custom-data controls; `data_provider/data_loader.py` and `data_provider_pretrain/data_loader.py` now support `channel_independence`, explicit custom split ratios, custom date columns, safer timestamp parsing, configurable ETT split windows, and clearer sample-shape semantics.
- Files used:
  `data_provider/data_factory.py` modified
  `data_provider/data_loader.py` modified
  `data_provider_pretrain/data_factory.py` modified
  `data_provider_pretrain/data_loader.py` modified
  `change.md` modified
- Exposed previously hard-coded Time-LLM model/runtime assumptions and replaced the benchmark ETTh1 launch script with a safer parameterized example.
- Purpose: make backbone source selection, prompt context, token budget, lag count, patch dtype, and script launch defaults configurable enough for custom-data bring-up.
- Impact: `models/TimeLLM.py` now accepts overrideable model/tokenizer paths, local-files-only behavior, generic dataset descriptions, configurable `top_k` / `num_tokens` / prompt max length / patch embedding dtype, and no longer injects a fixed ETT description by default; `scripts/TimeLLM_ETTh1.sh` now uses environment-overridable single-process-friendly defaults instead of hard-wiring 8 processes, bf16, and fixed dimensions.
- Files used:
  `models/TimeLLM.py` modified
  `scripts/TimeLLM_ETTh1.sh` modified
  `change.md` modified
- Closed the final custom-data safety gaps from the follow-up review by adding guarded checkpoint cleanup and richer custom split handling.
- Purpose: turn the first repaired pass into a safer end-to-end custom-data training baseline instead of stopping at a conditional pass.
- Impact: checkpoint cleanup now refuses to delete anything outside the configured checkpoints root (and will not delete the root itself), `Dataset_Custom` now supports univariate `date+target` datasets when `--features S`, custom datasets can use explicit `--train_end_date/--val_end_date` split boundaries, and both custom/ETT loaders now guard against negative split borders from too-short datasets or oversized sequence windows.
- Files used:
  `utils/tools.py` modified
  `run_main.py` modified
  `run_pretrain.py` modified
  `run_m4.py` modified
  `data_provider/data_factory.py` modified
  `data_provider/data_loader.py` modified
  `data_provider_pretrain/data_factory.py` modified
  `data_provider_pretrain/data_loader.py` modified
  `change.md` modified
- Finished the remaining follow-up fixes needed to fully retire the training-pipeline problem checklist and then removed `problems.md`.
- Purpose: eliminate the last benchmark-era assumptions that were still only partially addressed, so the repo’s custom-data training path no longer depends on hidden loader defaults, fixed month math, model identifiers inside the model class, or a fixed prompt-bank directory.
- Impact: prompt bank lookup is now directory-configurable via `--prompt_bank_dir`; LLM default source selection is prepared before model construction instead of being hard-coded inside `TimeLLM.py`; ETT and pretrain ETT loaders now split by dataset-length ratios instead of `30*24` month math; loader construction now requires explicit sequence windows; `channel_independence` defaults to `0`; and Oracle re-check confirmed the `problems.md` issue list is resolved enough to remove the document.
- Files used:
  `utils/tools.py` modified
  `run_main.py` modified
  `run_pretrain.py` modified
  `run_m4.py` modified
  `data_provider/data_factory.py` modified
  `data_provider/data_loader.py` modified
  `data_provider_pretrain/data_factory.py` modified
  `data_provider_pretrain/data_loader.py` modified
  `models/TimeLLM.py` modified
  `problems.md` deleted
  `change.md` modified
- Updated `plan.md` to reflect that the framework-repair phase is largely complete and to define the remaining experiment-wiring steps before your own data can be run end-to-end.
- Purpose: shift the project plan from “fix training infrastructure” to “freeze experiment table, split dates, feature whitelist, and launch order” so the next work starts the real park-data run instead of repeating framework cleanup.
- Impact: `plan.md` now explicitly distinguishes finished pipeline repairs from the remaining launch-preparation tasks (authoritative table selection, horizon choice, chronological split locking, XGBoost/DLinear/Time-LLM execution order, and conservative first-run settings for RTX 4060 8G).
- Files used:
  `plan.md` modified
  `change.md` modified
