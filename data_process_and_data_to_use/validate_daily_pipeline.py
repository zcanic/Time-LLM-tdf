import random

import pandas as pd

from merge_daily_data import (
    OUTPUT_FILE,
    RANDOM_SEED,
    SOURCE_DIR,
    build_daily_frame,
    run_random_checks,
)


def validate_saved_file(saved: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    assert list(saved.columns) == list(rebuilt.columns)
    assert len(saved) == len(rebuilt)
    assert saved["日期"].tolist() == rebuilt["日期"].tolist()


def validate_random_days(saved: pd.DataFrame) -> None:
    rng = random.Random(RANDOM_SEED + 1)

    park_raw = pd.read_csv(SOURCE_DIR / "data - 20221118 - 20251231 - 分钟数据-天坛公园.csv")
    park_raw["时间戳"] = pd.to_datetime(park_raw["时间戳"], errors="raise")
    park_raw["日期"] = park_raw["时间戳"].dt.strftime("%Y-%m-%d")

    sample_days = rng.sample(sorted(saved["日期"].tolist()), min(10, len(saved)))
    for day in sample_days:
        saved_row = saved.loc[saved["日期"] == day].iloc[0]
        raw_day = park_raw.loc[park_raw["日期"] == day]

        assert not raw_day.empty
        assert saved_row["park_时间戳_first"] == raw_day["时间戳"].min().strftime("%Y-%m-%d %H:%M:%S")
        assert saved_row["park_时间戳_last"] == raw_day["时间戳"].max().strftime("%Y-%m-%d %H:%M:%S")
        assert abs(float(saved_row["park_number"]) - float(raw_day["number"].mean())) < 1e-12
        assert float(saved_row["park_number_min"]) == float(raw_day["number"].min())
        assert float(saved_row["park_number_max"]) == float(raw_day["number"].max())
        assert int(saved_row["park_number_count"]) == int(len(raw_day))


def main() -> None:
    rebuilt = build_daily_frame()
    run_random_checks(rebuilt)

    saved = pd.read_csv(OUTPUT_FILE)
    validate_saved_file(saved, rebuilt)
    validate_random_days(saved)

    print("Validation passed.")
    print(f"Rows: {len(saved)}")
    print(f"Columns: {len(saved.columns)}")
    print(f"Date range: {saved['日期'].min()} to {saved['日期'].max()}")


if __name__ == "__main__":
    main()
