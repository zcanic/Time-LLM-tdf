from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = ROOT / "data_process_and_data_to_use" / "park_featured_data.csv"
OUTPUT_DIR = ROOT / "data_process_and_data_to_use" / "xgb_特征集"
OUTPUT_FILE = OUTPUT_DIR / "xgb_features.csv"


# Base columns copied directly from the reviewed featured table.
# These are already considered safe after the feature-pipeline hardening.
BASE_XGB_COLUMNS = [
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

# Additional structured columns approved for the expanded XGBoost baseline.
# These require lightweight encoding before export.
STRUCTURED_SOURCE_COLUMNS = [
    "weather_天气代码",
    "weather_最低气温_摄氏度",
    "weather_最高气温_摄氏度",
    "weather_平均气温_摄氏度",
    "weather_总降水量_毫米",
    "holiday_是否周末",
    "holiday_是否节假日放假",
    "holiday_是否调休上班",
    "holiday_日期标签",
    "holiday_星期",
    "交通状况",
]

# This mapping is intentionally manual instead of "unique value -> integer".
# The reason is that traffic state is naturally ordered and should be encoded
# according to that order, not according to arbitrary lexical order.
TRAFFIC_ORDER = {
    "通畅": 0,
    "较通畅": 1,
    "缓行": 2,
    "较为拥挤": 3,
    "拥挤": 4,
}

WEEKDAY_TO_INDEX = {
    "周一": 0,
    "周二": 1,
    "周三": 2,
    "周四": 3,
    "周五": 4,
    "周六": 5,
    "周日": 6,
}


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


def encode_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the additional structured columns approved for XGBoost.

    Encoding policy:
    - weather numeric values stay numeric
    - binary holiday flags become 0/1
    - weekday becomes sin/cos to preserve circular structure
    - date label uses one-hot encoding because the category count is very small
    - traffic status uses a manual ordinal mapping because the categories carry
      a natural congestion order

    Explicitly not done here:
    - no encoding of free-form text such as `环境描述`
    - no one-hot explosion of large-cardinality text columns
    """
    missing = [col for col in STRUCTURED_SOURCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing structured source columns: {missing}")

    out = df[BASE_XGB_COLUMNS].copy()

    out["weather_code"] = pd.to_numeric(df["weather_天气代码"], errors="raise")
    out["weather_temp_min"] = pd.to_numeric(df["weather_最低气温_摄氏度"], errors="raise")
    out["weather_temp_max"] = pd.to_numeric(df["weather_最高气温_摄氏度"], errors="raise")
    out["weather_temp_avg"] = pd.to_numeric(df["weather_平均气温_摄氏度"], errors="raise")
    out["weather_precip_total"] = pd.to_numeric(df["weather_总降水量_毫米"], errors="raise")

    out["is_weekend"] = (df["holiday_是否周末"] == "是").astype(int)
    out["is_holiday"] = (df["holiday_是否节假日放假"] == "是").astype(int)
    out["is_makeup_workday"] = (df["holiday_是否调休上班"] == "是").astype(int)

    weekday_index = df["holiday_星期"].map(WEEKDAY_TO_INDEX)
    if weekday_index.isna().any():
        unknown = sorted(df.loc[weekday_index.isna(), "holiday_星期"].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unknown weekday labels: {unknown}")
    angle = 2 * np.pi * weekday_index.astype(float) / 7.0
    out["weekday_sin"] = np.sin(angle)
    out["weekday_cos"] = np.cos(angle)

    date_tag_ohe = pd.get_dummies(df["holiday_日期标签"], prefix="date_tag", dtype=int)
    out = pd.concat([out, date_tag_ohe], axis=1)

    traffic_ord = df["交通状况"].map(TRAFFIC_ORDER)
    unknown_mask = traffic_ord.isna() & df["交通状况"].notna()
    if unknown_mask.any():
        unknown = sorted(df.loc[unknown_mask, "交通状况"].astype(str).unique().tolist())
        raise ValueError(f"Unknown traffic labels: {unknown}")
    out["traffic_level"] = traffic_ord.fillna(-1).astype(int)

    return out


def select_xgb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns approved for the expanded XGBoost baseline.

    This step acts like a lightweight feature gate:
    - if a required feature is missing, fail immediately
    - if the source file gains extra columns later, they will not silently slip
      into the XGBoost baseline
    """
    return encode_structured_features(df)


def run_checks(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the exported XGBoost dataset.

    These checks are not model checks. They only confirm that the exported file:
    - is chronologically ordered
    - still contains the target
    - contains the intended baseline feature families
    """
    assert df["时间戳"].is_monotonic_increasing
    assert "number" in df.columns
    assert "feat_baidu_lag1d" in df.columns
    assert "weather_code" in df.columns
    assert "is_weekend" in df.columns
    assert "weekday_sin" in df.columns
    assert "weekday_cos" in df.columns
    assert "traffic_level" in df.columns
    assert "环境描述" not in df.columns


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
