from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = ROOT / "data_process_and_data_to_use" / "park_featured_data.csv"
OUTPUT_DIR = ROOT / "data_process_and_data_to_use" / "xgb_特征集"
OUTPUT_FILE = OUTPUT_DIR / "xgb_features.csv"


# This whitelist follows the current baseline plan exactly.
# We keep:
# - identifier columns for debugging and split inspection
# - the target column `number`
# - the first-batch core XGBoost features
#
# We intentionally do not export prompt-oriented text columns here.
# We also do not export optional features at this stage, because the goal of
# this file is to provide a clean, baseline-ready dataset rather than an
# everything-bagel feature dump.
XGB_COLUMNS = [
    "时间戳",
    "日期",
    "number",
    "feat_number_diff_1row",
    "feat_number_pct_change_1row",
    "feat_number_momentum_4row",
    "feat_number_momentum_16row",
    "feat_number_momentum_48row",
    "feat_number_ma_4row",
    "feat_number_ma_16row",
    "feat_number_ma_48row",
    "feat_number_ma_spread_16_48row",
    "feat_number_slope_16row",
    "feat_number_slope_48row",
    "feat_number_vol_16row",
    "feat_number_vol_48row",
    "feat_number_pos_48row",
    "feat_baidu_lag1d",
]


def load_featured_data() -> pd.DataFrame:
    """
    Read the full featured dataset and enforce a stable row order.

    Why the sort matters:
    - chronological order is part of the training contract for time-series work
    - later train/validation/test splits should be able to rely on this order
    """
    df = pd.read_csv(SOURCE_FILE)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    df = df.sort_values("时间戳").reset_index(drop=True)
    return df


def select_xgb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns approved for the first XGBoost baseline.

    This step acts like a lightweight feature gate:
    - if a required feature is missing, fail immediately
    - if the source file gains extra columns later, they will not silently slip
      into the XGBoost baseline
    """
    missing = [col for col in XGB_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing XGBoost columns: {missing}")
    return df[XGB_COLUMNS].copy()


def run_checks(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the exported XGBoost dataset.

    These checks are not model checks. They only confirm that the exported file:
    - is chronologically ordered
    - still contains the target
    - only contains the intended baseline columns
    """
    assert list(df.columns) == XGB_COLUMNS
    assert df["时间戳"].is_monotonic_increasing
    assert "number" in df.columns
    assert "feat_baidu_lag1d" in df.columns


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_featured_data()
    xgb_df = select_xgb_columns(df)
    run_checks(xgb_df)

    xgb_df["时间戳"] = xgb_df["时间戳"].dt.strftime("%Y-%m-%d %H:%M:%S")
    xgb_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(xgb_df)}")
    print(f"Columns: {len(xgb_df.columns)}")


if __name__ == "__main__":
    main()
