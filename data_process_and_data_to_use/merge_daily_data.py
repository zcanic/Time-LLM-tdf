from pathlib import Path
import random

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "my_data_pure"
OUTPUT_DIR = ROOT / "data_process_and_data_to_use"
OUTPUT_FILE = OUTPUT_DIR / "park_aligned_data.csv"
RANDOM_SEED = 20260401
PARK_FILE = SOURCE_DIR / "data - 20221118 - 20251231 - 分钟数据-天坛公园.csv"


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


def load_park_raw() -> pd.DataFrame:
    df = read_csv(PARK_FILE)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    df["日期"] = df["时间戳"].dt.strftime("%Y-%m-%d")
    df["时间戳"] = df["时间戳"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def build_daily_frame() -> pd.DataFrame:
    weather = load_weather()
    holiday = load_holiday()
    baidu = load_baidu()
    park = load_park_raw()

    merged = park.merge(weather, on="日期", how="left")
    merged = merged.merge(holiday, on="日期", how="left")
    merged = merged.merge(baidu, on="日期", how="left")
    return merged.reset_index(drop=True)


def run_random_checks(merged: pd.DataFrame) -> None:
    rng = random.Random(RANDOM_SEED)

    park = load_park_raw()

    assert len(merged) == len(park)
    assert merged["时间戳"].tolist() == park["时间戳"].tolist()
    assert merged["number"].tolist() == park["number"].tolist()
    assert merged["number_flag"].tolist() == park["number_flag"].tolist()
    assert merged["交通状况"].tolist() == park["交通状况"].tolist()
    assert merged["环境描述"].tolist() == park["环境描述"].tolist()
    assert merged["原始文件名"].tolist() == park["原始文件名"].tolist()

    expected_start = str(park["日期"].iloc[0])
    expected_end = str(park["日期"].iloc[-1])
    assert merged["日期"].iloc[0] == expected_start
    assert merged["日期"].iloc[-1] == expected_end

    weather = load_weather()
    holiday = load_holiday()
    baidu = load_baidu()
    date_set = set(merged["日期"].tolist())

    def sample_and_check(source: pd.DataFrame) -> None:
        source = source.loc[source["日期"].isin(date_set)].reset_index(drop=True)
        sample_size = min(5, len(source))
        assert sample_size > 0
        indices = rng.sample(list(source.index), sample_size)
        for idx in indices:
            row = source.loc[idx]
            merged_rows = merged.loc[merged["日期"] == row["日期"]]
            assert len(merged_rows) > 0
            for col, value in row.items():
                if col == "日期":
                    continue
                if pd.isna(value):
                    assert merged_rows[col].isna().all()
                else:
                    assert (merged_rows[col] == value).all()

    sample_and_check(weather)
    sample_and_check(holiday)
    sample_and_check(baidu)

    sample_size = min(20, len(park))
    indices = rng.sample(list(park.index), sample_size)
    for idx in indices:
        source_row = park.loc[idx]
        merged_row = merged.loc[idx]
        for col in park.columns:
            assert merged_row[col] == source_row[col]


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
