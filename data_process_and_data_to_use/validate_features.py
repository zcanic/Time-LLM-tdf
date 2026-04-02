from pathlib import Path

import pandas as pd

from build_features import INPUT_FILE, OUTPUT_FILE, RAW_BAIDU_COLUMNS, ROWS_PER_PARK_DAY, main as build_main


def load_featured() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_FILE)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    return df.sort_values("时间戳").reset_index(drop=True)


def load_aligned() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    return df.sort_values("时间戳").reset_index(drop=True)


def validate_base_row_preservation(featured: pd.DataFrame, aligned: pd.DataFrame) -> None:
    assert len(featured) == len(aligned)
    for col in ["时间戳", "number", "number_flag", "交通状况", "环境描述", "原始文件名", "日期"]:
        left = featured[col].fillna("__NA__").astype(str)
        right = aligned[col].fillna("__NA__").astype(str)
        assert left.equals(right), f"base column mismatch: {col}"


def validate_baidu_safety(featured: pd.DataFrame, aligned: pd.DataFrame) -> None:
    for col in RAW_BAIDU_COLUMNS:
        assert col not in featured.columns, f"unsafe raw Baidu column still present: {col}"

    base_daily = (
        aligned[["日期", "baidu_PC+移动指数"]]
        .drop_duplicates(subset=["日期"])
        .sort_values("日期")
        .reset_index(drop=True)
    )
    featured_daily = (
        featured[["日期", "feat_baidu_lag1d", "feat_baidu_diff_1d", "feat_baidu_ma_3d", "feat_baidu_ma_7d"]]
        .drop_duplicates(subset=["日期"])
        .sort_values("日期")
        .reset_index(drop=True)
    )

    assert pd.isna(featured_daily.loc[0, "feat_baidu_lag1d"])
    for idx in range(1, len(featured_daily)):
        assert featured_daily.loc[idx, "feat_baidu_lag1d"] == base_daily.loc[idx - 1, "baidu_PC+移动指数"]

    expected_diff = featured_daily["feat_baidu_lag1d"].diff(1)
    left = featured_daily["feat_baidu_diff_1d"].fillna(-999999)
    right = expected_diff.fillna(-999999)
    assert left.equals(right)


def validate_sampling_structure(featured: pd.DataFrame) -> None:
    counts = featured.groupby("日期").size()
    assert int(counts.min()) == ROWS_PER_PARK_DAY
    assert int(counts.max()) == ROWS_PER_PARK_DAY
    assert counts.nunique() == 1


def main() -> None:
    # Rebuild first so validation is always checking the latest code path.
    build_main()

    featured = load_featured()
    aligned = load_aligned()

    validate_base_row_preservation(featured, aligned)
    validate_baidu_safety(featured, aligned)
    validate_sampling_structure(featured)

    print("Feature validation passed.")
    print(f"Rows: {len(featured)}")
    print(f"Columns: {len(featured.columns)}")
    print(f"Date range: {featured['日期'].min()} to {featured['日期'].max()}")


if __name__ == "__main__":
    main()
