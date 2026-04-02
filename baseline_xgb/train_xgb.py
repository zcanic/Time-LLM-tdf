from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data_process_and_data_to_use" / "xgb_特征集" / "xgb_features.csv"
OUTPUT_DIR = ROOT / "baseline_xgb"

BASE_TARGET_COL = "number"

# Feature columns are kept explicit so the training set does not accidentally
# expand when the source CSV gains extra columns later.
#
# Weather daily aggregates are intentionally excluded from this first repaired
# forecasting baseline. Their business-time availability is ambiguous for
# minute-level future prediction because a same-day daily min/max/avg/precip
# value can summarize information from later in the day.
FEATURE_COLS = [
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
    "is_weekend",
    "is_holiday",
    "is_makeup_workday",
    "weekday_sin",
    "weekday_cos",
    "date_tag_周末",
    "date_tag_工作日",
    "date_tag_节假日",
    "date_tag_调休上班",
    "traffic_level",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost forecasting baseline on curated time-series features."
    )
    parser.add_argument(
        "--horizon-rows",
        type=int,
        default=1,
        help="Forecast horizon in rows. A sample at row t predicts number at row t+horizon.",
    )
    return parser.parse_args()


def output_paths(horizon_rows: int) -> dict[str, Path]:
    suffix = f"_h{horizon_rows}row"
    return {
        "model": OUTPUT_DIR / f"xgb_model{suffix}.json",
        "metrics": OUTPUT_DIR / f"metrics{suffix}.json",
        "predictions": OUTPUT_DIR / f"predictions{suffix}.csv",
        "residual_fig": OUTPUT_DIR / f"residual_vs_target{suffix}.png",
        "train_curve_fig": OUTPUT_DIR / f"training_curve{suffix}.png",
    }


def target_col_name(horizon_rows: int) -> str:
    return f"target_number_t_plus_{horizon_rows}row"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    df = df.sort_values("时间戳").reset_index(drop=True)
    return df


def add_forecast_target(df: pd.DataFrame, horizon_rows: int) -> pd.DataFrame:
    # Forecast label construction:
    # - features are read from row t
    # - target is the future visitor count at row t + horizon_rows
    #
    # The trailing rows cannot form labels once the future point falls outside
    # the table, so they will be removed later in prepare_training_frame.
    out = df.copy()
    out[target_col_name(horizon_rows)] = out[BASE_TARGET_COL].shift(-horizon_rows)
    return out


def prepare_training_frame(df: pd.DataFrame, horizon_rows: int) -> tuple[pd.DataFrame, dict[str, int]]:
    # Drop rows that cannot form a valid feature vector because lagged Baidu and
    # backward-looking rolling features naturally produce NaN at the beginning
    # of the series. Also drop the trailing rows that cannot form a valid
    # future label once the target is shifted forward by horizon_rows.
    #
    # This is intentional. The leading rows do not have enough history yet, so
    # they are not valid training samples for the baseline. The trailing rows
    # do not have enough future horizon yet, so they are also invalid.
    target_col = target_col_name(horizon_rows)
    cols_needed = ["时间戳", "日期", BASE_TARGET_COL, target_col] + FEATURE_COLS
    missing = [col for col in cols_needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")

    candidate_df = df[cols_needed].copy()
    target_missing_mask = np.asarray(pd.isna(candidate_df[target_col]), dtype=bool)
    feature_missing_mask = np.asarray(pd.isna(candidate_df[FEATURE_COLS]), dtype=bool).any(axis=1)
    valid_mask = ~candidate_df.isna().to_numpy().any(axis=1)
    train_df = candidate_df.loc[valid_mask].reset_index(drop=True).copy()
    if train_df.empty:
        raise ValueError("Training frame is empty after dropping NaN feature rows.")

    first_valid_positions = np.flatnonzero(valid_mask)
    leading_history_drop = int(first_valid_positions[0]) if first_valid_positions.size > 0 else 0

    trailing_horizon_drop = 0
    for is_missing in target_missing_mask[::-1]:
        if bool(is_missing):
            trailing_horizon_drop += 1
        else:
            break

    feature_nan_rows_excluding_target = int((~target_missing_mask & feature_missing_mask).sum())

    drop_stats = {
        "raw_rows": int(len(df)),
        "leading_rows_dropped_for_history": leading_history_drop,
        "trailing_rows_dropped_for_horizon": trailing_horizon_drop,
        "rows_dropped_for_feature_nan": feature_nan_rows_excluding_target,
        "usable_rows": int(len(train_df)),
    }
    return train_df, drop_stats


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty split.")
    return train_df, val_df, test_df


def build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric=["rmse", "mae"],
        random_state=20260402,
        n_jobs=8,
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae)}


def save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, horizon_rows: int, output_file: Path) -> None:
    residuals = y_pred - y_true
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, residuals, s=10, alpha=0.35)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
    plt.xlabel(f"True future number (t+{horizon_rows})")
    plt.ylabel("Residual (prediction - true)")
    plt.title(f"XGBoost Residual vs Future Target (H={horizon_rows})")
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def save_training_curve(model: XGBRegressor, horizon_rows: int, output_file: Path) -> None:
    evals_result = model.evals_result()
    if "validation_0" not in evals_result:
        return

    train_rmse = evals_result["validation_0"].get("rmse")
    val_rmse = evals_result["validation_1"].get("rmse")
    train_mae = evals_result["validation_0"].get("mae")
    val_mae = evals_result["validation_1"].get("mae")

    if train_rmse is None or val_rmse is None or train_mae is None or val_mae is None:
        raise ValueError("Expected rmse and mae curves were not reported by XGBoost.")

    rounds = np.arange(1, len(train_rmse) + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(rounds, train_rmse, label="train_rmse", linewidth=1.6)
    plt.plot(rounds, val_rmse, label="val_rmse", linewidth=1.6)
    plt.plot(rounds, train_mae, label="train_mae", linewidth=1.4)
    plt.plot(rounds, val_mae, label="val_mae", linewidth=1.4)
    plt.xlabel("Boosting round")
    plt.ylabel("RMSE / MAE")
    plt.title(f"XGBoost Training / Validation RMSE and MAE (H={horizon_rows})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.horizon_rows < 1:
        raise ValueError("--horizon-rows must be at least 1.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.horizon_rows)
    target_col = target_col_name(args.horizon_rows)

    raw_df = load_data()
    labeled_df = add_forecast_target(raw_df, args.horizon_rows)
    df, drop_stats = prepare_training_frame(labeled_df, args.horizon_rows)
    train_df, val_df, test_df = chronological_split(df)

    model = build_model()
    model.fit(
        train_df[FEATURE_COLS],
        train_df[target_col],
        eval_set=[
            (train_df[FEATURE_COLS], train_df[target_col]),
            (val_df[FEATURE_COLS], val_df[target_col]),
        ],
        verbose=False,
    )

    val_pred = model.predict(val_df[FEATURE_COLS])
    test_pred = model.predict(test_df[FEATURE_COLS])

    metrics = {
        "horizon_rows": int(args.horizon_rows),
        "target_col": target_col,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(FEATURE_COLS)),
        "feature_columns": FEATURE_COLS,
        "split_scheme": "chronological_70_10_20",
        "drop_stats": drop_stats,
        "val": evaluate(val_df[target_col].to_numpy(), val_pred),
        "test": evaluate(test_df[target_col].to_numpy(), test_pred),
    }

    preds = test_df[["时间戳", "日期", BASE_TARGET_COL, target_col]].copy()
    preds["number_at_t"] = preds[BASE_TARGET_COL]
    preds = preds.drop(columns=[BASE_TARGET_COL]).copy()
    preds["prediction"] = test_pred

    save_residual_plot(test_df[target_col].to_numpy(), test_pred, args.horizon_rows, paths["residual_fig"])
    save_training_curve(model, args.horizon_rows, paths["train_curve_fig"])

    model.save_model(paths["model"])
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    timestamp_series = pd.Series(pd.to_datetime(preds["时间戳"], errors="raise"), index=preds.index)
    preds["时间戳"] = timestamp_series.dt.strftime("%Y-%m-%d %H:%M:%S")
    preds.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")

    print(f"Saved model: {paths['model']}")
    print(f"Saved metrics: {paths['metrics']}")
    print(f"Saved predictions: {paths['predictions']}")
    print(f"Saved residual plot: {paths['residual_fig']}")
    print(f"Saved training curve: {paths['train_curve_fig']}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
