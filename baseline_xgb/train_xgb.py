from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data_process_and_data_to_use" / "xgb_特征集" / "xgb_features.csv"
OUTPUT_DIR = ROOT / "baseline_xgb"
MODEL_FILE = OUTPUT_DIR / "xgb_model.json"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
PRED_FILE = OUTPUT_DIR / "predictions.csv"
RESIDUAL_FIG = OUTPUT_DIR / "residual_vs_target.png"
TRAIN_CURVE_FIG = OUTPUT_DIR / "training_curve.png"


# This baseline now uses an explicit future horizon instead of fitting the
# current-row `number` value. A sample at row t predicts the target at row t+H.
#
# Defaulting to 1 row keeps the change minimal while restoring valid
# forecasting semantics. The horizon is still a plain constant here so review
# stays simple and the generated artifacts remain deterministic.
HORIZON_ROWS = 1
BASE_TARGET_COL = "number"
TARGET_COL = f"target_number_t_plus_{HORIZON_ROWS}row"

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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["时间戳"] = pd.to_datetime(df["时间戳"], errors="raise")
    df = df.sort_values("时间戳").reset_index(drop=True)
    return df


def add_forecast_target(df: pd.DataFrame) -> pd.DataFrame:
    # Forecast label construction:
    # - features are read from row t
    # - target is the future visitor count at row t + HORIZON_ROWS
    #
    # The trailing rows cannot form labels once the future point falls outside
    # the table, so they will be removed later in prepare_training_frame.
    out = df.copy()
    out[TARGET_COL] = out[BASE_TARGET_COL].shift(-HORIZON_ROWS)
    return out


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows that cannot form a valid feature vector because lagged Baidu and
    # backward-looking rolling features naturally produce NaN at the beginning
    # of the series. Also drop the trailing rows that cannot form a valid
    # future label once the target is shifted forward by HORIZON_ROWS.
    #
    # This is intentional. The leading rows do not have enough history yet, so
    # they are not valid training samples for the baseline. The trailing rows
    # do not have enough future horizon yet, so they are also invalid.
    cols_needed = ["时间戳", "日期", BASE_TARGET_COL, TARGET_COL] + FEATURE_COLS
    missing = [col for col in cols_needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")
    train_df = df[cols_needed].dropna().reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training frame is empty after dropping NaN feature rows.")
    return train_df


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


def save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    residuals = y_pred - y_true
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, residuals, s=10, alpha=0.35)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
    plt.xlabel(f"True future number (t+{HORIZON_ROWS})")
    plt.ylabel("Residual (prediction - true)")
    plt.title(f"XGBoost Residual vs Future Target (H={HORIZON_ROWS})")
    plt.tight_layout()
    plt.savefig(RESIDUAL_FIG, dpi=180)
    plt.close()


def save_training_curve(model: XGBRegressor) -> None:
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
    plt.title("XGBoost Training / Validation RMSE and MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAIN_CURVE_FIG, dpi=180)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_data()
    labeled_df = add_forecast_target(raw_df)
    df = prepare_training_frame(labeled_df)
    train_df, val_df, test_df = chronological_split(df)

    model = build_model()
    model.fit(
        train_df[FEATURE_COLS],
        train_df[TARGET_COL],
        eval_set=[
            (train_df[FEATURE_COLS], train_df[TARGET_COL]),
            (val_df[FEATURE_COLS], val_df[TARGET_COL]),
        ],
        verbose=False,
    )

    val_pred = model.predict(val_df[FEATURE_COLS])
    test_pred = model.predict(test_df[FEATURE_COLS])

    metrics = {
        "horizon_rows": int(HORIZON_ROWS),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(FEATURE_COLS)),
        "feature_columns": FEATURE_COLS,
        "val": evaluate(val_df[TARGET_COL].to_numpy(), val_pred),
        "test": evaluate(test_df[TARGET_COL].to_numpy(), test_pred),
    }

    preds = test_df[["时间戳", "日期", BASE_TARGET_COL, TARGET_COL]].copy()
    preds = preds.rename(columns={BASE_TARGET_COL: "number_at_t"})
    preds["prediction"] = test_pred

    save_residual_plot(test_df[TARGET_COL].to_numpy(), test_pred)
    save_training_curve(model)

    model.save_model(MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    preds["时间戳"] = preds["时间戳"].dt.strftime("%Y-%m-%d %H:%M:%S")
    preds.to_csv(PRED_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved predictions: {PRED_FILE}")
    print(f"Saved residual plot: {RESIDUAL_FIG}")
    print(f"Saved training curve: {TRAIN_CURVE_FIG}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
