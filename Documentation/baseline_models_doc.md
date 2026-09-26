# `baseline_models` Module Documentation

The `baseline_models` directory contains the classical machine learning baselines for this project: **Ridge Regression**, **Random Forest**, and **XGBoost**. Their purpose is to set a humble-but-honest performance bar that the Graph Neural Network (GNN) must beat. A baseline is a deliberately simple, well-understood model. If the fancy GNN cannot outperform even these simple models, then the added complexity of the graph is not paying for itself.

<br>

## **1. The Shared Training Pipeline**
All three baselines follow the exact same evaluation protocol so their results could be compared fairly with each other and with the GNN:

1. **Identical Data Split:** The `edge_split.py` module (kept in `backend/`, outside this folder) generates one fixed train / validation / test split of the orders (70% / 15% / 15%, seed 42). Every baseline and the GNN consume the *exact same* orders. Without this, a difference between two models' scores could be blamed on luck of the split.
2. **Same Features:** The models use the continuous features the GNN also sees — `haversine_distance_km`, `route_count`, `seller_lat`, `seller_lng`, `customer_lat`, `customer_lng` — plus the one-hot encoded states (27 for seller + 27 for customer). Random Forest and XGBoost additionally receive `seller_city_idx` and `customer_city_idx` because tree-based models can grind through high-cardinality integer codes, whereas linear models cannot.
3. **Tuning on Validation, Testing Last:** Each model's hyperparameters are tuned on the **validation** split only. The **test** split is held untouched until the very end, so the reported numbers are a fair estimate of real-world behaviour.
4. **Two Kinds of Scores:** Every baseline first predicts the *number of delivery days* (`MAE`, `RMSE`), and then converts that prediction into an on-time verdict to compute Accuracy.

<br>

## **2. On-Time Accuracy & Threshold Calibration**
The model predicts transit days. To turn that into a binary verdict we compare it against the **promised date** the customer actually received:

$$ \text{window\_days} = \text{estimated\_delivery} - \text{purchase\_date} $$

$$ \text{on-time} = \text{predicted\_days} \leq \text{window\_days} $$

The Accuracy is the fraction of orders where the model's verdict matches reality.

A small detail: asking "is the predicted time within the promise" can be made stricter or looser by adding a tolerance $\tau$. Calibrating $\tau$ on the validation split (`tau 0 = strict, +2.0 = forgiving`) squeezes real accuracy gains out of the same regression model.

**Heads-up (the ceiling):** because 92.17% of orders were actually on-time, a lazy model that answers "always on-time" already scores 92.17%. So any accuracy near 91–92% is close to this structural ceiling — the regression metrics are what actually separate the models.

<br>

## **3. Ridge Regression (`baseline_ridge.py`)**
Ridge is a linear model — it learns a straight-line (flat) relationship between the 60 features and the delivery days, and adds a penalty (`α`) to keep the coefficients from exploding.

**Explanation:** imagine judging delivery speed by a strict formula like "distance × weight + constant". Ridge finds the most reliable version of that formula.

- **Hyperparameter tuned:** the strength of the penalty `α` over a grid from $10^{-3}$ to $10^{3}$. Best: `α = 3.16` (val MAE 5.4306).
- **Features:** 60 (it does **not** use city codes — a linear model cannot make sense of 4,107 arbitrary integers).
- **Why it matters:** it is the "simple math" floor. If the GNN cannot beat a straight line, something is wrong.

**Results (test):** `MAE 5.3038 days` · `RMSE 8.0763 days` · `On-time Accuracy 90.20%`.

<br>

## **4. Random Forest (`baseline_rf.py`)**
Random Forest is an ensemble of many decision trees. Each tree votes on the delivery time, and the forest averages their answers. It can capture non-linear relationships that linear models miss.

**Explanation:** imagine asking 500 different delivery experts to each form their own rulebook, then averaging their forecasts. The crowd's average is usually wiser and more stable than any single expert.

- **Hyperparameters tuned (grid):**

| n_estimators | max_depth | min_samples_leaf | val MAE |
|---|---|---|---|
| 150 | None | 1 | 5.4673 |
| 300 | None | 1 | 5.4566 |
| 500 | None | 1 | 5.4522 |
| 300 | None | 5 | 5.2307 |
| 150 | 24 | 5 | 5.2191 |
| 300 | 24 | 5 | 5.2150 |
| **300** | **24** | **10** | **5.1706** |
| 300 | 16 | 5 | 5.1749 |

- Best config: **300 trees, depth 24, leaf size 10** (val MAE 5.1706). Calibrated decision threshold: `tau = +2.0`.

**Results (test):** `MAE 5.0823 days` · `RMSE 7.8882 days` · `On-time Accuracy 91.72%` (90.79% before calibration).

<br>

## **5. XGBoost (`baseline_xgb.py`)**
XGBoost is gradient-boosted trees: the trees are built *one at a time*, and each new tree focuses on correcting the mistakes of all the trees before it. This sequential "learning from your errors" usually makes it the strongest tabular model.

**Explanation:** instead of 500 independent experts voting, imagine a single expert who re-studies only his wrong answers every round until he stops improving.

- **Hyperparameters tuned with early stopping** (auto-stops when validation stops improving, so tree count is chosen automatically):

| learning_rate | max_depth | val MAE |
|---|---|---|
| 0.05 | 4 | 5.1202 |
| 0.05 | 6 | 5.1187 |
| 0.1 | 4 | 5.1310 |
| 0.1 | 6 | 5.1286 |
| **0.05** | **8** | **5.1118** |
| 0.1 | 8 | 5.1326 |

- Best config: **lr 0.05, depth 8**, early-stopped at 286 boosting rounds (val MAE 5.1118). Calibrated threshold: `tau = +2.0`.

**Results (test):** `MAE 5.0196 days` · `RMSE 7.8726 days` · `On-time Accuracy 91.75%` (90.81% before calibration).

<br>

## **6. Quick Comparison of All Baselines**
The final verdict table, fastest to slowest-model (all measured on the same test orders):

### Regression Metrics & Best Hyperparameters

| Model | MAE (days) | RMSE (days) | Best Hyperparameters | Features |
|---|---|---|---|---|
| Ridge Regression | 5.3038 | 8.0763 | α = 3.16 | 60 |
| Random Forest | 5.0823 | 7.8882 | trees 300, depth 24, leaf 10 | 62 |
| **XGBoost** | **5.0196** | **7.8726** | lr 0.05, depth 8 | 62 |

### On-Time Classification

| Model | Accuracy (strict) | Accuracy (calibrated τ=+2.0) |
|---|---|---|
| Ridge Regression | 90.20% | — |
| Random Forest | 90.79% | 91.72% |
| **XGBoost** | **90.81%** | **91.75%** |

**Takeaways:**
1. **XGBoost wins** on every metric, but the gap to Random Forest is small (0.06 MAE).
2. All three models sit within ~0.5% of the 92.17% on-time ceiling — accuracy here is nearly exhausted; the battle must be fought on `MAE`/`RMSE`.
3. This is the bar the GNN must clear: **MAE < 5.02 days** and **RMSE < 7.87 days** to be considered a genuinely better model, not just a more complex one.

<br><br>
 Documentation Ends Here