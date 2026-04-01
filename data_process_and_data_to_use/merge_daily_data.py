from pathlib import Path
import random

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "my_data_pure"
OUTPUT_DIR = ROOT / "data_process_and_data_to_use"
OUTPUT_FILE = OUTPUT_DIR / "daily_all_data.csv"
RANDOM_SEED = 20260401


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_prefix(df: pd.DataFrame, prefix: str, date_col: str = "日期") -> pd.DataFrame:
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col != date_col}
    return df.rename(columns=rename_map)


def load_weather() -> pd.DataFrame:
    path = SOURCE_DIR / "beijing_weather_2022_2026_open_meteo_zh_annotated.csv"
    df = read_csv(path)
    df["日期"] = pd.to_datetime(df["日期"], errors="raise").dt.strftime("%Y-%m-%d")
    return add_prefix(df, "weather_")


def load_holiday() -> pd.DataFrame:
    path = SOURCE_DIR / "china_holiday_calendar_2022_2026_zh.csv"
    df = read_csv(path)
    df["日期"] = pd.to_datetime(df["日期"], errors="raise").dt.strftime("%Y-%m-%d")
    return add_prefix(df, "holiday_")


def load_baidu() -> pd.DataFrame:
    path = SOURCE_DIR / "百度指数_daily_merge_tiantan_20220601_20260131.csv"
    df = read_csv(path)
    df["日期"] = pd.to_datetime(df["日期"], errors="raise").dt.strftime("%Y-%m-%d")
    return add_prefix(df, "baidu_")


def join_unique(values: pd.Series) -> str:
    items = []
    for value in values.dropna():
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
    return " | ".join(items)


def load_park_daily() -> pd.DataFrame:
    path = SOURCE_DIR / "data - 20221118 - 20251231 - 分钟数据-天坛公园.csv"
    df = read_csv(path)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    df["日期"] = df["时间戳"].dt.strftime("%Y-%m-%d")

    daily = (
        df.groupby("日期", as_index=False)
        .agg(
            park_时间戳_first=("时间戳", lambda s: s.min().strftime("%Y-%m-%d %H:%M:%S")),
            park_时间戳_last=("时间戳", lambda s: s.max().strftime("%Y-%m-%d %H:%M:%S")),
            park_number=("number", "mean"),
            park_number_min=("number", "min"),
            park_number_max=("number", "max"),
            park_number_count=("number", "size"),
            park_number_flag=("number_flag", join_unique),
            park_交通状况=("交通状况", join_unique),
            park_环境描述=("环境描述", join_unique),
            park_原始文件名=("原始文件名", join_unique),
        )
    )
    return daily


def build_daily_frame() -> pd.DataFrame:
    weather = load_weather()
    holiday = load_holiday()
    baidu = load_baidu()
    park = load_park_daily()

    daily_index = pd.DataFrame(
        {
            "日期": pd.date_range(
                pd.to_datetime(park["日期"]).min(),
                pd.to_datetime(park["日期"]).max(),
                freq="D",
            ).strftime("%Y-%m-%d")
        }
    )

    merged = daily_index.merge(weather, on="日期", how="left")
    merged = merged.merge(holiday, on="日期", how="left")
    merged = merged.merge(baidu, on="日期", how="left")
    merged = merged.merge(park, on="日期", how="left")
    return merged.sort_values("日期").reset_index(drop=True)


def run_random_checks(merged: pd.DataFrame) -> None:
    rng = random.Random(RANDOM_SEED)

    assert merged["日期"].is_monotonic_increasing
    assert merged["日期"].is_unique

    park = load_park_daily()
    expected_start = str(park["日期"].min())
    expected_end = str(park["日期"].max())
    assert merged["日期"].iloc[0] == expected_start
    assert merged["日期"].iloc[-1] == expected_end

    weather = load_weather()
    holiday = load_holiday()
    baidu = load_baidu()
    date_set = set(merged["日期"].tolist())

    def sample_and_check(source: pd.DataFrame, prefix: str | None = None) -> None:
        source = source.loc[source["日期"].isin(date_set)].reset_index(drop=True)
        sample_size = min(5, len(source))
        assert sample_size > 0
        indices = rng.sample(list(source.index), sample_size)
        for idx in indices:
            row = source.loc[idx]
            merged_row = merged.loc[merged["日期"] == row["日期"]]
            assert len(merged_row) == 1
            merged_row = merged_row.iloc[0]
            for col, value in row.items():
                if col == "日期":
                    continue
                target_col = col if prefix is None else f"{prefix}{col}"
                if pd.isna(value):
                    assert pd.isna(merged_row[target_col])
                else:
                    assert merged_row[target_col] == value

    sample_and_check(weather)
    sample_and_check(holiday)
    sample_and_check(baidu)
    sample_and_check(park, prefix=None)

    park_raw = read_csv(SOURCE_DIR / "data - 20221118 - 20251231 - 分钟数据-天坛公园.csv")
    park_raw["时间戳"] = pd.to_datetime(park_raw["时间戳"], errors="raise")
    park_raw["日期"] = park_raw["时间戳"].dt.strftime("%Y-%m-%d")
    sample_days = rng.sample(sorted(park_raw["日期"].unique().tolist()), 5)

    for day in sample_days:
        raw_day = park_raw.loc[park_raw["日期"] == day].copy()
        merged_row = merged.loc[merged["日期"] == day].iloc[0]
        assert merged_row["park_时间戳_first"] == raw_day["时间戳"].min().strftime("%Y-%m-%d %H:%M:%S")
        assert merged_row["park_时间戳_last"] == raw_day["时间戳"].max().strftime("%Y-%m-%d %H:%M:%S")
        assert abs(float(merged_row["park_number"]) - float(raw_day["number"].mean())) < 1e-12
        assert float(merged_row["park_number_min"]) == float(raw_day["number"].min())
        assert float(merged_row["park_number_max"]) == float(raw_day["number"].max())
        assert int(merged_row["park_number_count"]) == int(len(raw_day))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = build_daily_frame()
    run_random_checks(merged)
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()
