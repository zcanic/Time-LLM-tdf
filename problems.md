# Training Pipeline Problems

## Scope

This note reviews the current Time-LLM training pipeline for one goal:
make the repo usable for training on custom data with fewer hidden assumptions.

Reviewed files:
- `run_main.py`
- `run_pretrain.py`
- `run_m4.py`
- `models/TimeLLM.py`
- `data_provider/data_factory.py`
- `data_provider/data_loader.py`
- `data_provider_pretrain/data_factory.py`
- `data_provider_pretrain/data_loader.py`
- `utils/tools.py`
- `layers/Embed.py`
- `scripts/TimeLLM_ETTh1.sh`

## Blocking Problems For Custom Data

### 1. Checkpoint cleanup is destructive

The main training entries delete the whole `./checkpoints` directory after training.

Files:
- `run_main.py:268-270`
- `run_pretrain.py:268-270`
- `run_m4.py:307-310`

Why this is a problem:
- This is unrelated to one experiment run.
- It can delete checkpoints from other runs.
- It makes debugging and resuming training harder.
- It is dangerous in a shared research repo.

Impact on custom training:
- High. You can lose artifacts even when your custom run itself is valid.

### 2. Custom dataset support is only partial

`data_provider/data_factory.py` only recognizes:
- `ETTh1`
- `ETTh2`
- `ETTm1`
- `ETTm2`
- `ECL`
- `Traffic`
- `Weather`
- `m4`

Files:
- `data_provider/data_factory.py:4-13`
- `data_provider/data_factory.py:17`

Why this is a problem:
- A new dataset name is not plug-and-play.
- You cannot simply pass `--data my_dataset`.
- Prompt loading also assumes the dataset name is one of a small fixed set.

Impact on custom training:
- High. A custom dataset name requires code edits before training can start.

### 3. Prompt loading is hard-wired to built-in dataset names

`utils.tools.load_content()` maps `ETT` to one file and otherwise uses `args.data`
to load `./dataset/prompt_bank/{name}.txt`.

Files:
- `utils/tools.py:226-231`
- `run_main.py:139`
- `run_pretrain.py:139`
- `run_m4.py:151`

Why this is a problem:
- A new dataset name without a matching prompt file fails.
- Relative path is fixed to `./dataset/prompt_bank/`.
- The code assumes prompt files are mandatory once `args.content` is set.

Impact on custom training:
- High. Custom data setup will break unless the prompt file convention is followed.

### 4. Default dataset description is wrong for non-ETT data

When `prompt_domain=0`, Time-LLM uses a fixed ETT description in the prompt.

Files:
- `models/TimeLLM.py:166-169`

Why this is a problem:
- This injects incorrect domain context into the LLM prompt.
- It is especially bad for custom domains like tourism, finance, weather fusion, or traffic.

Impact on custom training:
- High. Even if training runs, the prompt semantics are wrong by default.

### 5. Input and output dimensions are manually hard-coded

The training entry defaults set:
- `enc_in=7`
- `dec_in=7`
- `c_out=7`

The example scripts also hard-code these values.

Files:
- `run_main.py:62-64`
- `run_pretrain.py:63-65`
- `run_m4.py:62-64`
- `scripts/TimeLLM_ETTh1.sh:27-29`
- `scripts/TimeLLM_ETTh1.sh:53-55`
- `scripts/TimeLLM_ETTh1.sh:79-81`
- `scripts/TimeLLM_ETTh1.sh:106-108`

Why this is a problem:
- Custom data often has a different number of variables.
- The code does not infer these values from the CSV.
- A mismatch will break the model shape assumptions.

Impact on custom training:
- High. Easy to misconfigure and fail at runtime.

### 6. `Dataset_Custom` requires a very specific schema

The custom dataset loader assumes:
- there is a `date` column
- the target column exists exactly by name
- all remaining columns are feature columns

Files:
- `data_provider/data_loader.py:243-253`

Why this is a problem:
- Many real datasets use `datetime`, `timestamp`, `ds`, `Date`, or Chinese column names.
- There is no schema validation error with a helpful message.
- The loader mutates column order based on assumptions.

Impact on custom training:
- High. Your data must be pre-normalized to this schema before the loader can work.

### 7. Train/val/test split rules are fixed and non-configurable

`Dataset_Custom` uses:
- 70% train
- 10% validation
- 20% test

Files:
- `data_provider/data_loader.py:253-256`

Why this is a problem:
- Time series experiments often require date-based splits, not ratio-only splits.
- There is no CLI for split ratios or explicit split dates.
- This makes exact experiment control weak.

Impact on custom training:
- High. You may get a technically valid run but the wrong evaluation setup.

## High-Risk Design And Runtime Problems

### 8. LLM backbone identifiers are hard-coded inside the model

The model directly hard-codes:
- `huggyllama/llama-7b`
- `openai-community/gpt2`
- `google-bert/bert-base-uncased`

Files:
- `models/TimeLLM.py:45-151`

Why this is a problem:
- Model source cannot be swapped by path without code changes.
- Local cache status changes runtime behavior.
- Online fallback is implicit and not transparent from the training CLI.

Impact:
- High for reproducibility.
- High for offline or enterprise environments.

### 9. Precision policy is partially hard-coded

The scripts use `--mixed_precision bf16`, and the model casts patches with
`x_enc.to(torch.bfloat16)` before the reprogramming layer.

Files:
- `models/TimeLLM.py:240`
- `scripts/TimeLLM_ETTh1.sh:14`
- `scripts/TimeLLM_ETTh1.sh:40`
- `scripts/TimeLLM_ETTh1.sh:66`
- `scripts/TimeLLM_ETTh1.sh:93`

Why this is a problem:
- Some GPUs do not support bf16 well.
- Precision strategy is split across shell script and model code.
- It is harder to downgrade to fp16 or fp32 cleanly.

Impact:
- High on weaker hardware.

### 10. Training runtime assumes heavy distributed setup by default

The main entries always initialize:
- `DistributedDataParallelKwargs(find_unused_parameters=True)`
- `DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')`

Files:
- `run_main.py:102-103`
- `run_pretrain.py:102-103`
- `run_m4.py:102-103`

Why this is a problem:
- This is heavier than needed for a first custom-data smoke test.
- It couples all training modes to DeepSpeed config presence.
- Small local runs should not require the same stack as large benchmark runs.

Impact:
- Medium to high depending on hardware and environment.

### 11. Example launch scripts are benchmark-oriented, not onboarding-friendly

The ETTh1 script hard-codes:
- `num_process=8`
- `master_port=00097`
- `--mixed_precision bf16`
- `--root_path ./dataset/ETT-small/`

Files:
- `scripts/TimeLLM_ETTh1.sh:6-7`
- `scripts/TimeLLM_ETTh1.sh:14-36`

Why this is a problem:
- These are not safe defaults for a new machine.
- They are tuned for a benchmark setup, not for custom data bring-up.

Impact:
- Medium. New users may think these are the recommended defaults.

### 12. Environment variables are changed inside the training entry

The entry scripts set:
- `CURL_CA_BUNDLE=''`
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`

Files:
- `run_main.py:17-18`
- `run_pretrain.py:16-17`
- `run_m4.py:17-18`

Why this is a problem:
- These are side effects hidden inside the code path.
- They may mask SSL problems instead of solving them.
- They may influence GPU memory behavior unexpectedly.

Impact:
- Medium.

## Magic Numbers And Hidden Assumptions

### 13. Reprogramming prompt and lag logic use unexposed constants

TimeLLM uses:
- `top_k = 5`
- `num_tokens = 1000`
- `max_length = 2048`

Files:
- `models/TimeLLM.py:38`
- `models/TimeLLM.py:178`
- `models/TimeLLM.py:234`

Why this matters:
- These values may be fine for the paper setup, but they are not obviously valid for custom datasets.
- Long custom prompts may be truncated silently.
- Vocabulary projection width is fixed without user control.

Impact:
- Medium.

### 14. ETT loaders are entirely built on dataset-specific calendar math

The ETT hour and minute loaders use fixed split boundaries like:
- `12 * 30 * 24`
- `4 * 30 * 24`
- `12 * 30 * 24 * 4`

Files:
- `data_provider/data_loader.py:51-52`
- `data_provider/data_loader.py:148-149`
- `data_provider_pretrain/data_loader.py:47-53`
- `data_provider_pretrain/data_loader.py:151-163`

Why this matters:
- These are not generic loaders.
- They show the codebase is benchmark-first, not schema-first.

Impact:
- Medium. Mostly a sign of architecture limits rather than a direct blocker if you avoid ETT loaders.

### 15. Default sequence lengths are inherited from benchmark conventions

Several loaders default to:
- `seq_len = 24 * 4 * 4`
- `label_len = 24 * 4`
- `pred_len = 24 * 4`

Files:
- `data_provider/data_loader.py:19-21`
- `data_provider/data_loader.py:117-119`
- `data_provider/data_loader.py:215-217`
- `data_provider_pretrain/data_loader.py:15-17`
- `data_provider_pretrain/data_loader.py:113-115`

Why this matters:
- These defaults are arbitrary for custom data.
- If a user forgets to override them, the run may be valid but meaningless.

Impact:
- Medium.

### 16. Data loader behavior silently expands the sample count by channel

For ETT and custom datasets:
- each feature channel becomes a separate sample
- `__len__` returns `window_count * enc_in`

Files:
- `data_provider/data_loader.py:105`
- `data_provider/data_loader.py:203`
- `data_provider/data_loader.py:308`
- `data_provider_pretrain/data_loader.py:104`
- `data_provider_pretrain/data_loader.py:214`

Why this matters:
- This is not obvious from the CLI.
- It changes the effective sample count and batch semantics.
- It matters when you interpret training speed, dataset size, and feature handling.

Impact:
- Medium.

### 17. Validation/test DataLoader policy is not neutral

`data_factory` sets:
- `drop_last=True` even for test in most cases
- validation uses `shuffle=True`

Files:
- `data_provider/data_factory.py:22-29`
- `data_provider_pretrain/data_factory.py:16-23`

Why this matters:
- Dropping the last test batch is usually not desirable.
- Shuffling validation data is unnecessary for deterministic evaluation.

Impact:
- Medium. It may not break metrics, but it is poor evaluation hygiene.

## Invalid Or Questionable Handling

### 18. `llm_dim` defaults are written as strings even though the CLI type is `int`

Examples:
- `default='4096'`

Files:
- `run_main.py:81`
- `run_pretrain.py:82`
- `run_m4.py:81`

Why this matters:
- `argparse` will usually coerce the default because `type=int` is set, so it may not crash.
- Still, this is sloppy and easy to copy into worse patterns later.

Impact:
- Low to medium.

### 19. `load_content()` is called even when the prompt is not domain-specific

The training entries populate `args.content` regardless, then the model only uses it when `prompt_domain` is enabled.

Files:
- `run_main.py:139`
- `run_pretrain.py:139`
- `run_m4.py:151`
- `models/TimeLLM.py:166-169`

Why this matters:
- Prompt file loading can fail even if you intended to use the default description path.
- This creates unnecessary coupling between every run and prompt-bank files.

Impact:
- Medium.

### 20. `run_m4.py` contains likely broken forecast CSV indexing logic

This code sets the DataFrame index to the forecast ids, then immediately sets the index again
using the first forecast column:

- `forecasts_df.index = ...`
- `forecasts_df.index.name = 'id'`
- `forecasts_df.set_index(forecasts_df.columns[0], inplace=True)`

Files:
- `run_m4.py:277-280`

Why this matters:
- The final CSV shape/index semantics are likely wrong.
- This is not directly your custom-data path, but it is a real pipeline correctness issue.

Impact:
- Medium for M4 validity.

### 21. Relative paths are assumed everywhere

Examples:
- `./dataset`
- `./checkpoints`
- `./dataset/prompt_bank/...`
- `./m4_results/...`
- `./ds_config_zero2.json`

Files:
- `run_main.py:41`
- `run_main.py:53`
- `run_main.py:103`
- `run_pretrain.py:41`
- `run_pretrain.py:54`
- `run_pretrain.py:103`
- `run_m4.py:41`
- `run_m4.py:53`
- `run_m4.py:103`
- `utils/tools.py:231`
- `run_m4.py:270`

Why this matters:
- Behavior depends on the current working directory.
- It is easy to break runs when launching from another path or wrapper script.

Impact:
- Medium.

## Practical Readiness Assessment

Current state for custom data:
- The repo is good as a paper reproduction baseline.
- The repo is not yet good as a robust custom-data training framework.

Main reasons:
- too many hidden assumptions about dataset names, schemas, and prompt files
- destructive cleanup behavior
- hard-coded model/backbone/runtime choices
- weak separation between benchmark defaults and reusable training logic

## Recommended Refactor Order

### Phase 1: make custom data runs safe

- stop deleting the global checkpoints directory
- make prompt loading optional and explicit
- allow custom dataset names without editing `data_dict`
- add explicit schema validation for date/target/features
- make validation/test loaders deterministic and non-dropping by default

### Phase 2: make custom data runs configurable

- add split controls by ratio or explicit dates
- infer `enc_in/dec_in/c_out` or validate them against the CSV
- move LLM model names and local paths to CLI/config
- expose prompt description text directly from CLI or file path

### Phase 3: make model/runtime assumptions explicit

- expose `top_k`, `num_tokens`, prompt max length, and dtype policy
- separate benchmark scripts from minimal local training scripts
- isolate DeepSpeed and multi-GPU setup from the default single-run path

## Bottom Line

If the next goal is to train on your own data, the first things to fix are not fancy model changes.
The first things to fix are:
- unsafe artifact cleanup
- rigid data schema assumptions
- prompt-path coupling
- hard-coded dimension and runtime defaults

Without those changes, custom-data training will be fragile even if the model code itself is mostly usable.
