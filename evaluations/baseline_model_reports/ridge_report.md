# Ridge Regression - Baseline Report

## Dataset
- Orders/edges: 97323
- Features: 60 (6 continuous + 54 one-hot states)
- Split: train 68126 / val 14598 / test 14599 (70/15/15, seed 42)

## Hyperparameter tuning (validation)
- Grid: [0.001, 0.0031622776601683794, 0.01, 0.03162277660168379, 0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0, 316.22776601683796, 1000.0]
- Best alpha: 3.1622776601683795 (val MAE 5.4306)

## Regression metrics (test)
- MAE : 5.3038 days
- RMSE: 8.0763 days

## On-time classification (test)
- Accuracy (predicted on-time == actual on-time): 0.9020
- Actual on-time rate: 0.9217
