# XGBoost - Baseline Report

## Dataset
- Orders/edges: 97323
- Features: 62 (6 continuous + 54 one-hot states + 2 city index codes)
- Split: train 68126 / val 14598 / test 14599 (70/15/15, seed 42)

## Hyperparameter tuning (validation, early stopping)

| learning_rate | max_depth | val MAE |
|---|---|---|
| 0.05 | 4 | 5.1202 |
| 0.05 | 6 | 5.1187 |
| 0.1 | 4 | 5.1310 |
| 0.1 | 6 | 5.1286 |
| 0.05 | 8 | 5.1118 |
| 0.1 | 8 | 5.1326 |

- Best params: {'learning_rate': 0.05, 'max_depth': 8} (val MAE 5.1118)
- Best iteration: 286

## Threshold calibration (validation)

| tau (days) | val accuracy |
|---|---|
| -1.00 | 0.8948 |
| -0.75 | 0.8971 |
| -0.50 | 0.8997 |
| -0.25 | 0.9014 |
| +0.00 | 0.9026 |
| +0.25 | 0.9040 |
| +0.50 | 0.9064 |
| +0.75 | 0.9077 |
| +1.00 | 0.9091 |
| +1.25 | 0.9101 |
| +1.50 | 0.9112 |
| +1.75 | 0.9123 |
| +2.00 | 0.9128 |

- Best tau: +2.00 (val accuracy 0.9128)

## Regression metrics (test)
- MAE : 5.0196 days
- RMSE: 7.8726 days

## On-time classification (test)
- Accuracy (tau 0, uncalibrated): 0.9081
- Accuracy (tau +2.00): 0.9175
- Actual on-time rate: 0.9217
