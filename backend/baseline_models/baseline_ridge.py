import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND_ROOT)
sys.path.insert(0, BACKEND_ROOT)

from edge_split import edge_split

DATASET_PATH = os.path.join(ROOT, "Datasets", "raw_dataset", "dataset.csv")
REPORT_PATH = os.path.join(ROOT, "evaluations", "baseline_model_reports", "ridge_report.md")

SEED = 42
STATE_CODES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]
CONTINUOUS_COLS = [
    "haversine_distance_km",
    "route_count",
    "seller_lat",
    "seller_lng",
    "customer_lat",
    "customer_lng",
]
ALPHAS = np.logspace(-3, 3, 13)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    seller_state_ohe = pd.get_dummies(df["seller_state"], prefix="sell_state").reindex(
        columns=[f"sell_state_{s}" for s in STATE_CODES], fill_value=0
    )
    customer_state_ohe = pd.get_dummies(df["customer_state"], prefix="cust_state").reindex(
        columns=[f"cust_state_{s}" for s in STATE_CODES], fill_value=0
    )

    features = pd.concat(
        [
            df[CONTINUOUS_COLS],
            seller_state_ohe,
            customer_state_ohe,
        ],
        axis=1,
    )
    return features


def main():
    df = pd.read_csv(DATASET_PATH)
    df = df.assign(
        route_count=df.groupby(["seller_zip_code_prefix", "customer_zip_code_prefix"])[
            "seller_zip_code_prefix"
        ].transform("count")
    )

    X = build_features(df).astype(np.float64)
    y = df["actual_transit_days"].values

    purchase = pd.to_datetime(df["order_purchase_timestamp"])
    estimated = pd.to_datetime(df["order_estimated_delivery_date"])
    window_days = (estimated - purchase).dt.total_seconds() / 86400.0
    true_label = (y <= window_days.values).astype(int)

    train_idx, val_idx, test_idx = edge_split(len(df))

    scaler = StandardScaler().fit(X.iloc[train_idx][CONTINUOUS_COLS])
    X_scaled = X.copy()
    X_scaled[CONTINUOUS_COLS] = scaler.transform(X_scaled[CONTINUOUS_COLS])

    X_train, X_val, X_test = (
        X_scaled.iloc[train_idx],
        X_scaled.iloc[val_idx],
        X_scaled.iloc[test_idx],
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    label_val, label_test = true_label[val_idx], true_label[test_idx]

    best_alpha, best_val_mae, model = None, float("inf"), None
    tuned = {}
    for alpha in ALPHAS:
        fit = Ridge(alpha=alpha, random_state=SEED)
        fit.fit(X_train, y_train)
        val_mae = mean_absolute_error(y_val, fit.predict(X_val))
        tuned[alpha] = val_mae
        if val_mae < best_val_mae:
            best_alpha, best_val_mae, model = alpha, val_mae, fit

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    pred_label = (y_pred <= window_days.values[test_idx]).astype(int)
    accuracy = (pred_label == label_test).mean()
    actual_on_time_rate = label_test.mean()

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = [
        "# Ridge Regression - Baseline Report",
        "",
        f"## Dataset",
        f"- Orders/edges: {len(df)}",
        f"- Features: {X.shape[1]} ({len(CONTINUOUS_COLS)} continuous + 54 one-hot states)",
        f"- Split: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} (70/15/15, seed {SEED})",
        "",
        "## Hyperparameter tuning (validation)",
        f"- Grid: {ALPHAS.tolist()}",
        f"- Best alpha: {best_alpha} (val MAE {best_val_mae:.4f})",
        "",
        "## Regression metrics (test)",
        f"- MAE : {mae:.4f} days",
        f"- RMSE: {rmse:.4f} days",
        "",
        "## On-time classification (test)",
        f"- Accuracy (predicted on-time == actual on-time): {accuracy:.4f}",
        f"- Actual on-time rate: {actual_on_time_rate:.4f}",
        "",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=== Ridge Baseline (test) ===")
    print(f"Best alpha: {best_alpha} (val MAE {best_val_mae:.4f})")
    print(f"MAE : {mae:.4f} days")
    print(f"RMSE: {rmse:.4f} days")
    print(f"On-time accuracy: {accuracy:.4f}")
    print(f"Actual on-time rate: {actual_on_time_rate:.4f}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()