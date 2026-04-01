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


def load_base_data() -> pd.DataFrame:
    """
    Read the park-aligned base table without changing any original rows.

    Important design rule:
    - `park_aligned_data.csv` is the immutable base table.
    - This script does not aggregate, drop, reorder, or resample the original
      park rows.
    - Every feature added later is appended as a new column on top of the same
      row order.

    This guarantees that downstream review can always compare:
    - original park row i
    - featured park row i
    and confirm that only new columns were added.
    """
    df = pd.read_csv(INPUT_FILE)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    df["日期"] = pd.to_datetime(df["日期"], errors="raise").dt.strftime("%Y-%m-%d")
    df = df.sort_values("时间戳").reset_index(drop=True)
    return df


def add_lagged_baidu_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a strictly lagged Baidu feature.

    Why this exists:
    - The research rule is that the model must not see the same-day Baidu value.
    - Therefore we first collapse Baidu to one value per calendar day.
    - Then we shift the daily series by exactly one day.
    - Finally, we merge that lagged daily value back onto every minute-level row
      of the corresponding day.

    Example:
    - all rows on 2022-11-19 receive the Baidu value from 2022-11-18
    - all rows on 2022-11-18 receive NaN because there is no earlier day inside
      the aligned park range

    Leakage rule:
    - no row can read Baidu from its own date
    - no row can read Baidu from a future date
    """
    baidu_daily = (
        df[["日期", "baidu_PC+移动指数"]]
        .drop_duplicates(subset=["日期"])
        .sort_values("日期")
        .reset_index(drop=True)
    )
    baidu_daily["baidu_PC+移动指数_lag1d"] = baidu_daily["baidu_PC+移动指数"].shift(1)

    df = df.merge(
        baidu_daily[["日期", "baidu_PC+移动指数_lag1d"]],
        on="日期",
        how="left",
        sort=False,
    )
    return df


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

    Chosen windows:
    - 4 rows  = about 1 hour  at 15-minute spacing
    - 16 rows = about 4 hours at 15-minute spacing
    - 96 rows = about 1 day   at 15-minute spacing

    These windows are simple enough to inspect and map well to the current data.
    """
    number = df["number"].astype(float)

    # One-step absolute change. This describes immediate local movement.
    df["feat_number_diff_1"] = number.diff(1)

    # One-step relative change. This normalizes the move by the previous level.
    # Replace exact zero denominators with NaN to avoid invalid infinite values.
    prev_number = number.shift(1).replace(0, np.nan)
    df["feat_number_pct_change_1"] = (number - number.shift(1)) / prev_number

    # Momentum over a short and medium history window.
    # Definition here is "current value minus value k rows ago".
    df["feat_number_momentum_4"] = number - number.shift(4)
    df["feat_number_momentum_16"] = number - number.shift(16)

    # Backward-looking moving averages.
    # These windows include the current row and the history before it.
    df["feat_number_ma_4"] = number.rolling(window=4, min_periods=4).mean()
    df["feat_number_ma_16"] = number.rolling(window=16, min_periods=16).mean()
    df["feat_number_ma_96"] = number.rolling(window=96, min_periods=96).mean()

    # Short-vs-long moving-average relationship.
    # This is a compact way to express local trend versus broader background.
    df["feat_number_ma_spread_16_96"] = df["feat_number_ma_16"] - df["feat_number_ma_96"]
    df["feat_number_ma_ratio_16_96"] = df["feat_number_ma_16"] / df["feat_number_ma_96"].replace(0, np.nan)

    # Rolling slope over two scales.
    # This gives a directional trend estimate rather than only a level difference.
    df["feat_number_slope_16"] = rolling_slope(number, 16)
    df["feat_number_slope_96"] = rolling_slope(number, 96)

    # Rolling volatility using standard deviation.
    # Again: backward-only, never future-looking.
    df["feat_number_vol_16"] = number.rolling(window=16, min_periods=16).std()
    df["feat_number_vol_96"] = number.rolling(window=96, min_periods=96).std()

    # Relative position inside the past 1-day window.
    # Value near 0 means close to the past-window minimum.
    # Value near 1 means close to the past-window maximum.
    rolling_min_96 = number.rolling(window=96, min_periods=96).min()
    rolling_max_96 = number.rolling(window=96, min_periods=96).max()
    range_96 = (rolling_max_96 - rolling_min_96).replace(0, np.nan)
    df["feat_number_pos_96"] = (number - rolling_min_96) / range_96

    return df


def add_lagged_baidu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small first batch of lagged-Baidu-derived features.

    The same leakage rule still applies:
    - all of these features are built from the already-lagged daily Baidu series
    - therefore same-day Baidu is still unavailable
    - rolling windows remain backward-only
    """
    baidu = df["baidu_PC+移动指数_lag1d"].astype(float)

    # Keep the raw lagged explanatory variable as an explicit feature column.
    df["feat_baidu_lag1d"] = baidu

    # One-day change on the lagged Baidu series.
    df["feat_baidu_diff_1d"] = baidu.diff(1)

    # Short and medium moving averages on the lagged Baidu series.
    # Because the value is daily but repeated to minute rows, these features are
    # more about slow regime shifts than minute-level variation.
    df["feat_baidu_ma_96"] = baidu.rolling(window=96, min_periods=96).mean()
    df["feat_baidu_ma_288"] = baidu.rolling(window=288, min_periods=288).mean()
    df["feat_baidu_ma_spread_96_288"] = df["feat_baidu_ma_96"] - df["feat_baidu_ma_288"]

    return df


def run_checks(df: pd.DataFrame) -> None:
    """
    Perform sanity checks on the engineered output.

    The checks focus on two things:
    1. the original park base rows must remain unchanged
    2. the key leakage-sensitive Baidu lag must behave as intended
    """
    base = load_base_data()

    # Original row count and ordering must remain unchanged.
    assert len(df) == len(base)
    assert df["时间戳"].tolist() == base["时间戳"].tolist()
    assert df["number"].tolist() == base["number"].tolist()
    assert df["原始文件名"].tolist() == base["原始文件名"].tolist()

    # Same-day Baidu must not be visible through the lagged feature.
    daily = (
        df[["日期", "baidu_PC+移动指数", "baidu_PC+移动指数_lag1d"]]
        .drop_duplicates(subset=["日期"])
        .sort_values("日期")
        .reset_index(drop=True)
    )
    assert pd.isna(daily.loc[0, "baidu_PC+移动指数_lag1d"])
    for idx in range(1, len(daily)):
        assert daily.loc[idx, "baidu_PC+移动指数_lag1d"] == daily.loc[idx - 1, "baidu_PC+移动指数"]


def main() -> None:
    """
    Full processing pipeline.

    Processing order:
    1. load the immutable park-aligned base table
    2. create a strictly lagged Baidu daily feature
    3. create backward-only `number` features
    4. create a small set of backward-only lagged-Baidu features
    5. run sanity checks
    6. save a new CSV without overwriting the base table
    """
    df = load_base_data()
    df = add_lagged_baidu_feature(df)
    df = add_number_features(df)
    df = add_lagged_baidu_features(df)
    run_checks(df)

    # Convert timestamp back to a stable string form before saving so downstream
    # scripts see a consistent representation.
    df["时间戳"] = df["时间戳"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
