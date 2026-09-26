# Random Forest - Baseline Report

## Dataset
- Orders/edges: 97323
- Features: 62 (6 continuous + 54 one-hot states + 2 city index codes)
- Split: train 68126 / val 14598 / test 14599 (70/15/15, seed 42)

## Hyperparameter tuning (validation)

| n_estimators | max_depth | min_samples_leaf | val MAE |
|---|---|---|---|
| 150 | None | 1 | 5.4673 |
| 300 | None | 1 | 5.4566 |
| 500 | None | 1 | 5.4522 |
| 300 | None | 5 | 5.2307 |
| 150 | 24 | 5 | 5.2191 |
| 300 | 24 | 5 | 5.2150 |
| 300 | 24 | 10 | 5.1706 |
| 300 | 16 | 5 | 5.1749 |

- Best params: {'n_estimators': 300, 'max_depth': 24, 'min_samples_leaf': 10} (val MAE 5.1706)

## Threshold calibration (validation)

| tau (days) | val accuracy |
|---|---|
| -1.00 | 0.8936 |
| -0.75 | 0.8957 |
| -0.50 | 0.8977 |
| -0.25 | 0.9002 |
| +0.00 | 0.9018 |
| +0.25 | 0.9031 |
| +0.50 | 0.9055 |
| +0.75 | 0.9071 |
| +1.00 | 0.9075 |
| +1.25 | 0.9090 |
| +1.50 | 0.9105 |
| +1.75 | 0.9124 |
| +2.00 | 0.9127 |

- Best tau: +2.00 (val accuracy 0.9127)

## Regression metrics (test)
- MAE : 5.0823 days
- RMSE: 7.8882 days

## On-time classification (test)
- Accuracy (tau 0, uncalibrated): 0.9079
- Accuracy (tau +2.00): 0.9172
- Actual on-time rate: 0.9217
