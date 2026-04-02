# Time-LLM Adaptation Plan

## Goal

Use Time-LLM to predict the `number` column from the Tiantan park base table, while:
- keeping the Tiantan park rows unchanged
- using Baidu index as a history-only explanatory variable
- using the other aligned fields mainly as textual prompt context
- avoiding future information leakage
- fitting a practical setup for an RTX 4060 8G machine

## Feasibility Judgment

Yes, this goal is possible.

The most important constraint is not the model idea itself. The main constraint is how the data is fed into the model.

The safe interpretation is:
- target: `number`
- numeric history input: past `number`, and lagged Baidu index features only
- prompt input: weather, holiday, traffic status, environment description, and other date-aligned context converted into text
- forbidden input: any future `number`, same-day or future Baidu index, or any prompt content built from future timestamps

This is feasible if every training sample is constructed only from information available up to the forecast origin.

## Key Research Decision

### Recommended setup

Use a single-target forecasting setup where:
- the model predicts only `number`
- Baidu index enters as a numeric covariate only after an explicit lag
- weather and holiday metadata are treated as prompt context only
- traffic status and environment description are treated carefully, because they come from the same park-side source and may behave more like concurrent observations than truly exogenous signals

Why this is the recommended setup:
- it matches your stated causal intent
- it minimizes leakage risk
- it is easier to debug
- it is much more realistic on 8 GB VRAM

### What to avoid first

Avoid starting with:
- multivariate output forecasting
- large LLaMA-class backbones
- complex prompt construction that changes per row using future-day summaries
- mixing too many numeric features before the leakage boundaries are explicit

## Leakage Rules

These are the rules the implementation should obey.

### 1. Target leakage

For a forecast origin at time `t`, the sample may use:
- `number` history up to `t`

The sample must not use:
- `number` after `t`

### 2. Baidu index leakage

Your stated idea is valid:
- Baidu index can be used as an explanatory variable without predicting it
- but only if the sample only sees the part of the Baidu series that would be known by time `t`

This means:
- use explicitly lagged historical Baidu values only
- do not use the same-day Baidu value
- do not inject Baidu values from forecast horizon timestamps into model input
- default to a one-day lag first, then only relax that rule if you intentionally redesign the causal assumption

### 3. Prompt leakage

Prompt text must also obey the same time boundary.

Safe prompt content:
- static dataset description
- historical summary statistics computed from the input window only
- holiday labels for timestamps that are already within the observed input side
- weather summaries from the observed side only
- text summaries of engineered behavior only if those summaries are built from past-only windows

Unsafe prompt content:
- summaries over the full day when the prediction origin is inside that day
- future holiday labels for forecast timestamps unless you intentionally define them as known-in-advance calendar covariates
- future weather facts unless they are coming from a separate forecast product and you intentionally treat them as known future covariates
- any rolling summary built with centered or forward-looking windows

## Variable Role Plan

### Target

- `number`

### Numeric covariates

Phase 1 numeric covariates:
- `number`
- `feat_baidu_lag1d`

Recommendation:
- start with only one Baidu series, represented by the already-safe lagged feature `feat_baidu_lag1d`
- do not reintroduce raw same-day Baidu columns into the training-facing table

Reason:
- they are highly related and may be redundant
- fewer covariates make leakage audits easier
- your current research intent does not require multiple Baidu channels at the start

### Engineered time-series features

These are now part of the current data-preparation direction, borrowing ideas from
financial forecasting, for example:
- change features
- momentum
- short and long moving-average relationships
- rolling slope
- rolling volatility
- relative position inside a past window

Recommendation:
- do not start with a large feature bank
- keep the current feature set small and interpretable
- treat these as numeric branch features, not prompt features

Hard rule:
- every rolling or window-based feature must be computed using past-only information
- rolling windows may only look backward
- no centered windows
- no future rows
- no full-day summary that includes timestamps after the forecast origin

This rule applies to:
- moving averages
- momentum
- rolling slope
- rolling standard deviation
- rolling min/max based relative position
- any derived feature based on local window statistics

Current implementation direction:
- `park_featured_data.csv` already contains a first batch of backward-only features
- feature names use explicit row-based semantics such as `4row`, `16row`, `48row`, `96row`
- on the current park source, `48` rows means 1 park day and `96` rows means 2 park days

### Prompt-only context

Good candidates:
- holiday fields
- weather fields
- `交通状况`
- `环境描述`

Explicit rule:
- weather values stay in the prompt/context path only
- weather values do not enter the numeric time-series branch

But there is an important caution:
- `交通状况` and `环境描述` are likely synchronous observations from the same source stream as the park signal
- they may still be useful, but should be treated as observed-history context only
- do not summarize them using future rows

## Hardware Plan

Machine constraint:
- RTX 4060 8G

Implication:
- 7B/9B class backbones are not the right starting point here
- the default repo setup is too heavy

Recommended backbone order:
1. `GPT2`
2. `BERT` only if GPT-2 fails to provide usable behavior
3. only consider larger models later if the pipeline is already proven and memory strategy is redesigned

Recommended starting point:
- `llm_model=GPT2`
- small batch size
- single GPU
- no DeepSpeed dependency on the first smoke run if possible

Why GPT-2 first:
- much lighter
- easier to fit on 8 GB
- enough to validate the data construction and prompting logic

## Baseline Plan

### XGBoost first

Before investing in Time-LLM training, start with an `XGBoost` baseline.

Why this should come first:
- it validates the data pipeline before introducing LLM complexity
- it is a strong fit for the current structured feature table
- it is much cheaper to iterate on with local hardware
- it gives a realistic reference point for judging whether Time-LLM adds value

Recommended first XGBoost inputs:
- `number`-derived engineered features
- safe lagged Baidu features
- optionally a small set of structured calendar fields

### XGBoost feature list

Use an explicit whitelist. Do not rely on automatic numeric-column selection.

#### Core features

These should be the first batch for the baseline:
- `feat_number_diff_1row`
- `feat_number_pct_change_1row`
- `feat_number_momentum_4row`
- `feat_number_momentum_16row`
- `feat_number_momentum_48row`
- `feat_number_ma_4row`
- `feat_number_ma_16row`
- `feat_number_ma_48row`
- `feat_number_ma_spread_16_48row`
- `feat_number_slope_16row`
- `feat_number_slope_48row`
- `feat_number_vol_16row`
- `feat_number_vol_48row`
- `feat_number_pos_48row`
- `feat_baidu_lag1d`

Why these are core:
- they cover short-term change
- they cover intraday and day-level trend
- they cover local volatility
- they cover current position relative to recent history
- they include the main lagged external explanatory variable

#### Optional features

These are useful for ablation after the core baseline is stable:
- `feat_number_ma_96row`
- `feat_number_ma_ratio_16_48row`
- `feat_baidu_diff_1d`
- `feat_baidu_ma_3d`
- `feat_baidu_ma_7d`
- `feat_baidu_ma_spread_3d_7d`
- structured calendar indicators derived from:
  - `holiday_是否周末`
  - `holiday_是否节假日放假`
  - `holiday_是否调休上班`
  - `holiday_日期标签`

Why these are optional:
- they may help, but they are not required to test the baseline hypothesis
- some are more redundant with the core features
- some require an extra encoding step for tree-model use

#### Not recommended at baseline start

Do not include these in the first XGBoost run:
- raw weather text fields
- raw traffic text fields
- `环境描述`
- any prompt-only text field
- any raw same-day `baidu_*` column
- uncontrolled one-hot expansion of many sparse text columns

Why they are excluded:
- they complicate the baseline without answering the core question first
- they increase the chance of data leakage or brittle preprocessing
- they are better reserved for later prompt-oriented Time-LLM experiments

Avoid in the first XGBoost baseline:
- raw text fields
- any same-day raw Baidu columns
- uncontrolled automatic inclusion of all numeric columns

Primary XGBoost ablations:
- `number` features only
- `number` features + lagged Baidu features
- `number` features + lagged Baidu features + selected structured calendar features

Interpretation goal:
- determine whether the current feature set already explains the prediction target well
- identify whether Baidu adds measurable value before moving to Time-LLM
- establish a credible structured-model baseline for later comparison

## Model Adaptation Plan

### Phase 1: build a valid forecasting problem

Objective:
- make Time-LLM consume your aligned park-based dataset without leakage

Needed changes:
- use `park_aligned_data.csv` as the immutable source table
- define `number` as the only prediction target
- create a custom loader that builds samples from minute-level rows
- ensure each sample uses a past window and predicts a future horizon
- consume `park_featured_data.csv` as the training-facing feature table
- keep same-day raw Baidu columns excluded from the training-facing feature table
- keep engineered features backward-only before sample extraction

Success criterion:
- a reproducible train/val/test pipeline that produces valid windows

### Phase 2: separate numeric input from prompt input

Objective:
- control which variables go into the numerical time-series branch and which go into the text prompt

Needed changes:
- numeric branch receives `number`, `feat_baidu_lag1d`, and a small audited set of engineered numeric features
- prompt builder receives weather/holiday/traffic/environment information only from the observed window
- prompt builder must not inspect future rows
- prompt builder must keep weather in text form only, not as numeric model input
- prompt builder should not read engineered numeric features as free-form text unless that transformation is explicitly reviewed

Success criterion:
- for any sample, it is possible to explain exactly why each input field is available at forecast time

### Phase 2.5: add engineered features conservatively

Objective:
- improve the numeric branch with interpretable state features without breaking causality

Status:
- a first batch already exists in `park_featured_data.csv`

Needed changes:
- audit which existing engineered columns are actually allowed into training
- prefer the smallest high-signal subset first
- keep all such features backward-only
- validate feature availability row by row before training

Success criterion:
- every engineered feature can be traced to past-only source rows
- the feature set remains small enough to audit manually

### Phase 3: make the runtime fit 4060 8G

Objective:
- replace benchmark-heavy defaults with a local, minimal training path

Needed changes:
- use GPT-2 first
- reduce batch size
- avoid multi-process defaults
- avoid mandatory DeepSpeed startup for first experiments
- keep sequence length and prompt size conservative

Success criterion:
- one end-to-end training run fits and finishes on local hardware

### Phase 0: establish a structured baseline

Objective:
- validate the feature table and leakage boundary with a strong non-LLM model first

Needed changes:
- define a strict feature whitelist for tree-based training
- create chronological train/val/test splits on the featured dataset
- train an `XGBoost` baseline on the structured numeric features
- compare variants with and without lagged Baidu features

Success criterion:
- a reproducible non-LLM baseline exists before Time-LLM experiments begin

## Data Split Plan

Use strict chronological splits.

Recommended order:
- train: earliest block
- validation: middle block
- test: latest block

Do not use:
- shuffled row splits
- random ratio splits that obscure temporal boundaries

Recommendation:
- start with explicit date ranges in code or config
- later expose them as parameters

## Evaluation Plan

Primary metrics:
- MAE
- RMSE
- MAPE only if `number` never gets too close to zero

Baselines to compare:
- naive last-value baseline
- `XGBoost` baseline
- DLinear baseline from the same repo
- Time-LLM with no Baidu covariate
- Time-LLM with lagged Baidu covariate
- Time-LLM with and without prompt context

Why this matters:
- it tells you whether a strong tabular model already solves most of the task
- it tells you whether Baidu helps
- it tells you whether prompt context helps
- it tells you whether the LLM layer is actually buying anything over simpler baselines

## Risk Register

### Risk 1

Baidu index is daily, park data is minute-level.

Risk:
- naive repetition of a daily value across every minute may overstate how much information is available

Mitigation:
- document the assumption explicitly
- enforce a lag so the same-day Baidu value is never visible
- start with a one-day lag as the default causal assumption
- only compare against less strict variants later if you explicitly want that ablation

Current state:
- raw same-day Baidu columns have already been removed from the training-facing feature table
- only lagged/safe Baidu-derived features remain there

### Risk 2

Weather and holiday data may contain fields that are known ahead of time and fields that are not.

Risk:
- calendar fields are known in advance
- realized weather is usually not

Mitigation:
- separate “known future calendar” from “observed weather history”
- do not treat realized future weather as free information

### Risk 3

Minute-level forecasting with prompt construction can be expensive.

Risk:
- prompt generation per sample may be slow
- memory usage may spike

Mitigation:
- begin with a simpler prompt
- precompute lightweight textual summaries
- keep prompt length short

### Risk 4

Engineered features may silently leak if window logic is implemented carelessly.

Risk:
- rolling functions often look safe while accidentally including the current boundary incorrectly
- centered windows or post-merge daily summaries can leak future information into minute-level rows

Mitigation:
- treat backward-only rolling windows as a non-negotiable rule
- review feature code separately from model code
- test engineered features with timestamp-level audits before training

Current state:
- there is already a dedicated feature validation script for this layer
- the remaining risk is no longer feature generation itself, but incorrect feature selection at training time

## Recommended Execution Order

1. Freeze the causal rules for each field.
2. Define a strict training column whitelist from `park_featured_data.csv`.
3. Build an `XGBoost` baseline on the whitelisted structured features.
4. Run chronological ablations for lagged Baidu and engineered features.
5. Build a custom minute-level dataset loader using the featured table for numeric inputs and the aligned contextual fields for prompt construction.
6. Add an explicit prompt builder that uses observed-window-only context.
7. Create a single-GPU GPT-2 training entry for smoke testing.
8. Run a tiny overfit test on a small slice to validate shapes and leakage boundaries.
9. Run a real chronological train/val/test experiment.
10. Compare Time-LLM against the `XGBoost` and DLinear baselines.

## Bottom Line

Your research target is valid and technically feasible.

The strongest version of the project is:
- predict `number`
- treat Baidu index as a lagged history-only explanatory variable
- treat weather/holiday/park metadata as controlled prompt context
- enforce time-availability rules sample by sample
- use GPT-2 first because 4060 8G is the real engineering constraint

The next implementation should optimize for validity first, not model size or benchmark parity.
