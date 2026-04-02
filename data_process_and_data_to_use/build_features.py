from pathlib import Path

import numpy as np
import pandas as pd


# This script is intentionally verbose in comments.
# The goal is not only to generate features, but also to make the exact
# processing logic easy to inspect during review.


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_process_and_data_to_use"
INPUT_FILE = DATA_DIR / "park_aligned_data.csv"
OUTPUT_FILE = DATA_DIR / "park_featured_data.csv"

# The park source has a stable intraday grid of 48 rows per day:
# 09:00, 09:15, ..., 20:45.
# This constant is used to make window semantics explicit and reviewable.
ROWS_PER_PARK_DAY = 48

# These columns are considered unsafe to keep in the training-facing featured
# table because they expose same-day Baidu information. The raw aligned file is
# kept separately, so dropping them here does not destroy source data.
RAW_BAIDU_COLUMNS = [
    "baidu_关键词",
    "baidu_城市代码",
    "baidu_城市",
    "baidu_数据类型",
    "baidu_数据间隔(天)",
    "baidu_所属年份",
    "baidu_PC+移动指数",
    "baidu_移动指数",
    "baidu_PC指数",
    "baidu_爬取时间",
]


def load_base_data() -> pd.DataFrame:
    """
    Read the park-aligned base table without changing any original park rows.

    Important design rule:
    - `park_aligned_data.csv` is the immutable aligned source.
    - This script does not aggregate, drop, reorder, or resample the original
      park-side rows.
    - The only structural changes are:
      1. append engineered feature columns
      2. remove raw same-day Baidu columns from the *featured* output only

    Review implication:
    - row i in the output must still correspond to row i in the base park table
    - the park-side columns themselves must remain unchanged
    """
    df = pd.read_csv(INPUT_FILE)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    df["日期"] = pd.to_datetime(df["日期"], errors="raise").dt.strftime("%Y-%m-%d")
    df = df.sort_values("时间戳").reset_index(drop=True)
    return df


def build_baidu_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Baidu features at the daily level before joining them back.

    Why daily-first processing is required:
    - Baidu is naturally daily data.
    - If we compute rolling statistics after repeating a daily value onto every
      minute row, the window length becomes hard to interpret.
    - Daily-first processing keeps the time meaning explicit.

    Leakage rule:
    - the model must not see the Baidu value from the same date
    - therefore we shift the daily series by exactly one day first
    - every feature in this function is derived from that lagged daily series

    Semantics:
    - `lag1d` means previous calendar day
    - `ma_3d` means a backward-looking 3-day average over lagged daily values
    - `ma_7d` means a backward-looking 7-day average over lagged daily values
    """
    daily = (
        df[["日期", "baidu_PC+移动指数"]]
        .drop_duplicates(subset=["日期"])
        .sort_values("日期")
        .reset_index(drop=True)
    )

    daily["feat_baidu_lag1d"] = daily["baidu_PC+移动指数"].shift(1)
    daily["feat_baidu_diff_1d"] = daily["feat_baidu_lag1d"].diff(1)
    daily["feat_baidu_ma_3d"] = daily["feat_baidu_lag1d"].rolling(window=3, min_periods=3).mean()
    daily["feat_baidu_ma_7d"] = daily["feat_baidu_lag1d"].rolling(window=7, min_periods=7).mean()
    daily["feat_baidu_ma_spread_3d_7d"] = daily["feat_baidu_ma_3d"] - daily["feat_baidu_ma_7d"]

    return daily[
        [
            "日期",
            "feat_baidu_lag1d",
            "feat_baidu_diff_1d",
            "feat_baidu_ma_3d",
            "feat_baidu_ma_7d",
            "feat_baidu_ma_spread_3d_7d",
        ]
    ]


def attach_baidu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the safe Baidu feature set back onto each minute-level park row.

    Every minute row for the same date receives the same lagged daily Baidu
    feature values. This is acceptable because the source signal itself is daily.
    The critical safeguard is that the attached values are already lagged, so
    the current date never sees its own Baidu value.
    """
    baidu_features = build_baidu_daily_features(df)
    return df.merge(baidu_features, on="日期", how="left", sort=False)


def rolling_slope(values: pd.Series, window: int) -> pd.Series:
    """
    Compute a backward-looking rolling slope.

    Implementation detail:
    - Each slope is computed only from the current row and the previous
      `window - 1` rows.
    - The function never uses centered windows and never uses future rows.
    - A simple least-squares slope over an equally spaced x-axis is used.

    Interpretation:
    - positive slope means the local trend is rising
    - negative slope means the local trend is falling
    - magnitude reflects local trend steepness
    """
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    x_var = np.sum(x_centered ** 2)

    def _calc(arr: np.ndarray) -> float:
        y = arr.astype(float)
        y_centered = y - y.mean()
        return float(np.sum(x_centered * y_centered) / x_var)

    return values.rolling(window=window, min_periods=window).apply(_calc, raw=True)


def add_number_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add backward-only engineered features derived from `number`.

    All windows are intentionally one-sided:
    - they use the current row plus the past rows before it
    - they do not look into the future

    Window semantics are written against the actual park data cadence:
    - 4 rows   = about 1 hour
    - 16 rows  = about 4 hours
    - 48 rows  = 1 full park day
    - 96 rows  = 2 full park days

    The names use explicit row counts to avoid the earlier ambiguity where
    `96` rows was casually described as "1 day".
    """
    number = df["number"].astype(float)

    df["feat_number_diff_1row"] = number.diff(1)

    prev_number = number.shift(1).replace(0, np.nan)
    df["feat_number_pct_change_1row"] = (number - number.shift(1)) / prev_number

    df["feat_number_momentum_4row"] = number - number.shift(4)
    df["feat_number_momentum_16row"] = number - number.shift(16)
    df["feat_number_momentum_48row"] = number - number.shift(ROWS_PER_PARK_DAY)

    df["feat_number_ma_4row"] = number.rolling(window=4, min_periods=4).mean()
    df["feat_number_ma_16row"] = number.rolling(window=16, min_periods=16).mean()
    df["feat_number_ma_48row"] = number.rolling(window=ROWS_PER_PARK_DAY, min_periods=ROWS_PER_PARK_DAY).mean()
    df["feat_number_ma_96row"] = number.rolling(window=2 * ROWS_PER_PARK_DAY, min_periods=2 * ROWS_PER_PARK_DAY).mean()

    df["feat_number_ma_spread_16_48row"] = df["feat_number_ma_16row"] - df["feat_number_ma_48row"]
    df["feat_number_ma_ratio_16_48row"] = df["feat_number_ma_16row"] / df["feat_number_ma_48row"].replace(0, np.nan)

    df["feat_number_slope_16row"] = rolling_slope(number, 16)
    df["feat_number_slope_48row"] = rolling_slope(number, ROWS_PER_PARK_DAY)

    df["feat_number_vol_16row"] = number.rolling(window=16, min_periods=16).std()
    df["feat_number_vol_48row"] = number.rolling(window=ROWS_PER_PARK_DAY, min_periods=ROWS_PER_PARK_DAY).std()

    rolling_min_48 = number.rolling(window=ROWS_PER_PARK_DAY, min_periods=ROWS_PER_PARK_DAY).min()
    rolling_max_48 = number.rolling(window=ROWS_PER_PARK_DAY, min_periods=ROWS_PER_PARK_DAY).max()
    range_48 = (rolling_max_48 - rolling_min_48).replace(0, np.nan)
    df["feat_number_pos_48row"] = (number - rolling_min_48) / range_48

    return df


def drop_raw_baidu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove raw same-day Baidu columns from the training-facing featured output.

    Why this is necessary:
    - keeping both raw same-day Baidu and lagged Baidu in the same featured CSV
      creates a very easy leakage path
    - a later training script might accidentally auto-select all numeric columns
    - dropping the raw Baidu columns here makes misuse harder by construction

    Important boundary:
    - the raw aligned dataset is still preserved in `park_aligned_data.csv`
    - only the derived training-facing dataset removes the unsafe columns
    """
    existing = [col for col in RAW_BAIDU_COLUMNS if col in df.columns]
    return df.drop(columns=existing)


def run_checks(df: pd.DataFrame) -> None:
    """
    Perform sanity checks on the engineered output.

    The checks focus on three things:
    1. original park base rows remain unchanged
    2. unsafe raw Baidu columns are absent
    3. lagged Baidu daily features obey the no-same-day rule
    """
    base = load_base_data()

    assert len(df) == len(base)
    assert df["时间戳"].tolist() == base["时间戳"].tolist()
    assert df["number"].tolist() == base["number"].tolist()
    assert df["number_flag"].tolist() == base["number_flag"].tolist()
    assert df["交通状况"].tolist() == base["交通状况"].tolist()
    assert df["环境描述"].tolist() == base["环境描述"].tolist()
    assert df["原始文件名"].tolist() == base["原始文件名"].tolist()

    for col in RAW_BAIDU_COLUMNS:
        assert col not in df.columns

    daily_base = (
        base[["日期", "baidu_PC+移动指数"]]
        .drop_duplicates(subset=["日期"])
        .sort_values("日期")
        .reset_index(drop=True)
    )
    daily_featured = (
        df[["日期", "feat_baidu_lag1d"]]
        .drop_duplicates(subset=["日期"])
        .sort_values("日期")
        .reset_index(drop=True)
    )

    assert len(daily_base) == len(daily_featured)
    assert pd.isna(daily_featured.loc[0, "feat_baidu_lag1d"])
    for idx in range(1, len(daily_featured)):
        assert daily_featured.loc[idx, "feat_baidu_lag1d"] == daily_base.loc[idx - 1, "baidu_PC+移动指数"]


def main() -> None:
    """
    Full processing pipeline.

    Processing order:
    1. load the immutable park-aligned base table
    2. attach safe daily-level lagged Baidu features
    3. create backward-only `number` features with explicit row-based semantics
    4. remove unsafe raw same-day Baidu columns from the featured output
    5. run sanity checks
    6. save a new CSV without overwriting the base aligned table
    """
    df = load_base_data()
    df = attach_baidu_features(df)
    df = add_number_features(df)
    df = drop_raw_baidu_columns(df)
    run_checks(df)

    df["时间戳"] = df["时间戳"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
