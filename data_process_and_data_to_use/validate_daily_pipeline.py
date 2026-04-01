import random

import pandas as pd

from merge_daily_data import (
    OUTPUT_FILE,
    PARK_FILE,
    RANDOM_SEED,
    SOURCE_DIR,
    build_daily_frame,
    run_random_checks,
)


def validate_saved_file(saved: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    assert list(saved.columns) == list(rebuilt.columns)
    assert len(saved) == len(rebuilt)
    assert saved["时间戳"].tolist() == rebuilt["时间戳"].tolist()
    assert saved["日期"].tolist() == rebuilt["日期"].tolist()


def validate_random_rows(saved: pd.DataFrame) -> None:
    rng = random.Random(RANDOM_SEED + 1)

    park_raw = pd.read_csv(PARK_FILE)
    park_raw["时间戳"] = pd.to_datetime(park_raw["时间戳"], errors="raise").dt.strftime("%Y-%m-%d %H:%M:%S")
    park_raw["日期"] = pd.to_datetime(park_raw["时间戳"], errors="raise").dt.strftime("%Y-%m-%d")

    sample_rows = rng.sample(list(range(len(saved))), min(20, len(saved)))
    for idx in sample_rows:
        saved_row = saved.iloc[idx]
        raw_row = park_raw.iloc[idx]

        for col in ["时间戳", "number", "number_flag", "交通状况", "环境描述", "原始文件名", "日期"]:
            if pd.isna(saved_row[col]) and pd.isna(raw_row[col]):
                continue
            assert saved_row[col] == raw_row[col]

    weather = pd.read_csv(SOURCE_DIR / "beijing_weather_2022_2026_open_meteo_zh_annotated.csv")
    weather["日期"] = pd.to_datetime(weather["日期"], errors="raise").dt.strftime("%Y-%m-%d")
    sample_days = rng.sample(sorted(saved["日期"].drop_duplicates().tolist()), 10)
    for day in sample_days:
        source = weather.loc[weather["日期"] == day]
        if source.empty:
            continue
        saved_rows = saved.loc[saved["日期"] == day]
        assert not saved_rows.empty
        source_row = source.iloc[0]
        for col in source.columns:
            if col == "日期":
                continue
            target_col = f"weather_{col}"
            if pd.isna(source_row[col]):
                assert saved_rows[target_col].isna().all()
            else:
                assert (saved_rows[target_col] == source_row[col]).all()


def main() -> None:
    rebuilt = build_daily_frame()
    run_random_checks(rebuilt)

    saved = pd.read_csv(OUTPUT_FILE)
    validate_saved_file(saved, rebuilt)
    validate_random_rows(saved)

    print("Validation passed.")
    print(f"Rows: {len(saved)}")
    print(f"Columns: {len(saved.columns)}")
    print(f"Date range: {saved['日期'].min()} to {saved['日期'].max()}")
    print(f"First timestamp: {saved['时间戳'].iloc[0]}")
    print(f"Last timestamp: {saved['时间戳'].iloc[-1]}")


if __name__ == "__main__":
    main()
