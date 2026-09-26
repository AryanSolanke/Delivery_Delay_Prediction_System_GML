import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND_ROOT)
sys.path.insert(0, BACKEND_ROOT)

from edge_split import edge_split

DATASET_PATH = os.path.join(ROOT, "Datasets", "raw_dataset", "dataset.csv")
NODE_TABLE_PATH = os.path.join(BACKEND_ROOT, "artifacts", "node_table.csv")
REPORT_PATH = os.path.join(ROOT, "evaluations", "baseline_model_reports", "xgb_report.md")

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
PARAM_GRID = [
    {"learning_rate": 0.05, "max_depth": 4},
    {"learning_rate": 0.05, "max_depth": 6},
    {"learning_rate": 0.1, "max_depth": 4},
    {"learning_rate": 0.1, "max_depth": 6},
    {"learning_rate": 0.05, "max_depth": 8},
    {"learning_rate": 0.1, "max_depth": 8},
]
MAX_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30
TAUS = np.arange(-1.0, 2.01, 0.25)


def build_features(df: pd.DataFrame, node_table: pd.DataFrame) -> pd.DataFrame:
    seller_state_ohe = pd.get_dummies(df["seller_state"], prefix="sell_state").reindex(
        columns=[f"sell_state_{s}" for s in STATE_CODES], fill_value=0
    )
    customer_state_ohe = pd.get_dummies(df["customer_state"], prefix="cust_state").reindex(
        columns=[f"cust_state_{s}" for s in STATE_CODES], fill_value=0
    )

    zip_to_node = node_table.set_index("zip_code_prefix")
    seller_city_idx = df["seller_zip_code_prefix"].map(zip_to_node["city_idx"]).astype("int64")
    customer_city_idx = df["customer_zip_code_prefix"].map(zip_to_node["city_idx"]).astype("int64")

    features = pd.concat(
        [
            df[CONTINUOUS_COLS],
            seller_state_ohe,
            customer_state_ohe,
            seller_city_idx.rename("seller_city_idx"),
            customer_city_idx.rename("customer_city_idx"),
        ],
        axis=1,
    )
    return features


def main():
    df = pd.read_csv(DATASET_PATH)
    node_table = pd.read_csv(NODE_TABLE_PATH)
    df = df.assign(
        route_count=df.groupby(["seller_zip_code_prefix", "customer_zip_code_prefix"])[
            "seller_zip_code_prefix"
        ].transform("count")
    )

    X = build_features(df, node_table).astype(np.float64)
    y = df["actual_transit_days"].values

    purchase = pd.to_datetime(df["order_purchase_timestamp"])
    estimated = pd.to_datetime(df["order_estimated_delivery_date"])
    window_days = (estimated - purchase).dt.total_seconds() / 86400.0
    true_label = (y <= window_days.values).astype(int)

    train_idx, val_idx, test_idx = edge_split(len(df))

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    label_test = true_label[test_idx]

    best_params, best_val_mae, model = None, float("inf"), None
    tuned = []
    for params in PARAM_GRID:
        fit = xgb.XGBRegressor(
            n_estimators=MAX_ESTIMATORS,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            eval_metric="mae",
            **params,
        )
        fit.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_mae = mean_absolute_error(y_val, fit.predict(X_val))
        tuned.append((params, val_mae))
        print(f"Tuned {params} -> val MAE {val_mae:.4f}")
        if val_mae < best_val_mae:
            best_params, best_val_mae, model = params, val_mae, fit

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    val_pred = model.predict(X_val)
    label_val = true_label[val_idx]
    window_val = window_days.values[val_idx]

    tau_scores = []
    for tau in TAUS:
        val_acc = ((val_pred <= window_val + tau) == label_val).mean()
        tau_scores.append((tau, val_acc))
    best_tau, best_val_acc = max(tau_scores, key=lambda t: t[1])

    pred_label_tau0 = (y_pred <= window_days.values[test_idx]).astype(int)
    pred_label = (y_pred <= window_days.values[test_idx] + best_tau).astype(int)
    accuracy = (pred_label == label_test).mean()
    accuracy_tau0 = (pred_label_tau0 == label_test).mean()
    actual_on_time_rate = label_test.mean()

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = [
        "# XGBoost - Baseline Report",
        "",
        "## Dataset",
        f"- Orders/edges: {len(df)}",
        f"- Features: {X.shape[1]} ({len(CONTINUOUS_COLS)} continuous + 54 one-hot states + 2 city index codes)",
        f"- Split: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} (70/15/15, seed {SEED})",
        "",
        "## Hyperparameter tuning (validation, early stopping)",
        "",
        "| learning_rate | max_depth | val MAE |",
        "|---|---|---|",
    ]
    for params, val_mae in tuned:
        lines.append(f"| {params['learning_rate']} | {params['max_depth']} | {val_mae:.4f} |")
    lines += [
        "",
        f"- Best params: {best_params} (val MAE {best_val_mae:.4f})",
        f"- Best iteration: {model.best_iteration}",
        "",
        "## Threshold calibration (validation)",
        "",
        "| tau (days) | val accuracy |",
        "|---|---|",
    ]
    for tau, val_acc in sorted(tau_scores, key=lambda t: t[0]):
        lines.append(f"| {tau:+.2f} | {val_acc:.4f} |")
    lines += [
        "",
        f"- Best tau: {best_tau:+.2f} (val accuracy {best_val_acc:.4f})",
        "",
        "## Regression metrics (test)",
        f"- MAE : {mae:.4f} days",
        f"- RMSE: {rmse:.4f} days",
        "",
        "## On-time classification (test)",
        f"- Accuracy (tau 0, uncalibrated): {accuracy_tau0:.4f}",
        f"- Accuracy (tau {best_tau:+.2f}): {accuracy:.4f}",
        f"- Actual on-time rate: {actual_on_time_rate:.4f}",
        "",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n=== XGBoost Baseline (test) ===")
    print(f"Best params: {best_params} (val MAE {best_val_mae:.4f})")
    print(f"Best tau: {best_tau:+.2f} (val accuracy {best_val_acc:.4f})")
    print(f"MAE : {mae:.4f} days")
    print(f"RMSE: {rmse:.4f} days")
    print(f"On-time accuracy (tau 0): {accuracy_tau0:.4f}")
    print(f"On-time accuracy (tau {best_tau:+.2f}): {accuracy:.4f}")
    print(f"Actual on-time rate: {actual_on_time_rate:.4f}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()