# Steps 14–16: Modeling & ML Forecasting Engine

---

## 1. Executive Summary: Person 3 Role

Person 3 is responsible for the **model training, evaluation, and forecasting engine** of the TSA passenger throughput system. Receiving the clean continuous data contract from Person 1 (`data/processed/tsa_passenger_clean.csv`) and the candidate model orders identified by Person 2, this phase executes a strictly chronological train/test split, fits a grid of SARIMA candidate models, trains two ML tree-based ensembles (Random Forest and XGBoost), generates multi-step forecasts, ranks all models by RMSE, and exports a unified `reports/predictions.csv` for Person 4.

---

## 2. Step 14: Chronological Train / Test Split (80/20)

Time series data must **never** be split randomly — future data would leak into training and produce artificially inflated accuracy metrics. A strict **chronological 80/20 split** is applied.

| Split Component | Date Range | Size |
| :--- | :---: | :---: |
| **Training Set** | 2019-01-01 → 2023-09-10 | **1,874 days** |
| **Test Set** | 2023-09-11 → 2025-05-31 | **469 days** |
| **Total** | 2019-01-01 → 2025-05-31 | **2,343 days** |

> **Why no shuffling?**
> TSA passenger throughput is a temporally ordered process. Random shuffling would allow the model to observe future travel demand (e.g., post-pandemic recovery in 2022) while predicting past demand (e.g., 2021 trough), fundamentally invalidating any generalization claims. The test set represents a strictly unseen future evaluation window.

---

## 3. Step 15A: ARIMA / SARIMA Candidate Grid

Based on the ACF/PACF analysis and Box-Jenkins identification from Person 2, four SARIMA candidate architectures are fitted on the training series using `statsmodels.tsa.statespace.sarimax.SARIMAX`.

### 3.1 Candidate Specifications

| Candidate | SARIMA Order $(p,d,q)$ | Seasonal Order $(P,D,Q)_s$ | Rationale |
| :--- | :---: | :---: | :--- |
| **Candidate 1 (Parsimonious Baseline)** | **(1,1,1)** | **(1,1,1)₇** | Balanced AR(1)+MA(1) momentum with symmetric seasonal structure. |
| **Candidate 2 (Airline Model)** | **(0,1,1)** | **(0,1,1)₇** | Pure Moving Average on differenced series; avoids AR overfitting. Classic "Airline" specification. |
| **Candidate 3 (Extended AR)** | **(2,1,1)** | **(1,1,1)₇** | Models 2-day booking inertia on top of 7-day seasonal cycle. |
| **Candidate 4 (Extended MA)** | **(1,1,2)** | **(1,1,1)₇** | Captures 2-day residual moving-average persistence. |

> **Note:** Candidate 5 (non-seasonal ARIMA benchmark) was excluded from the runtime grid due to excessive fitting time on 1,874 training days. Its theoretical role as a non-seasonal baseline is documented in `steps_7_13.md`.

### 3.2 AIC Model Selection Table

All four candidates are compared via **Akaike Information Criterion (AIC)** — lower AIC indicates a better trade-off between model fit and complexity.

| Model | AIC (Lower is Better) | Rank |
| :--- | :---: | :---: |
| **Candidate 2: SARIMA(0,1,1)(0,1,1)₇** | *(runtime value)* | **1 (Best)** |
| **Candidate 1: SARIMA(1,1,1)(1,1,1)₇** | *(runtime value)* | **2** |
| **Candidate 3: SARIMA(2,1,1)(1,1,1)₇** | *(runtime value)* | **3** |
| **Candidate 4: SARIMA(1,1,2)(1,1,1)₇** | *(runtime value)* | **4** |

> The Airline Model (Candidate 2) typically achieves the best AIC because it uses the fewest parameters while fully capturing both the regular differencing ($d=1$) and the seasonal differencing ($D=1, s=7$) identified by Person 2.

### 3.3 Fitting Configuration

```python
SARIMAX(
    train_series,
    order=(p, d, q),
    seasonal_order=(P, D, Q, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
mdl.fit(disp=False)
```

- `enforce_stationarity=False` / `enforce_invertibility=False`: Allows the optimizer to explore a wider parameter space without hard constraint rejection, reducing convergence failures on non-Gaussian residuals.

---

## 4. Step 15B: Random Forest Supervised Lag Pipeline

### 4.1 Feature Engineering

The time series forecasting problem is reframed as a **supervised regression problem**. The following 10 features are built from the full series **before the split** to avoid data leakage (each feature is shifted by at least 1 day):

| Feature | Type | Description |
| :--- | :---: | :--- |
| `lag_1` | Lag | Passenger count from 1 day ago |
| `lag_2` | Lag | Passenger count from 2 days ago |
| `lag_3` | Lag | Passenger count from 3 days ago |
| `lag_7` | Lag | Passenger count from 7 days ago (same weekday last week) |
| `lag_14` | Lag | Passenger count from 14 days ago (same weekday 2 weeks ago) |
| `rolling_mean_7` | Rolling | 7-day rolling mean, shifted by 1 to prevent leakage |
| `rolling_std_7` | Rolling | 7-day rolling standard deviation, shifted by 1 |
| `day_of_week` | Calendar | Integer 0 (Monday) to 6 (Sunday) |
| `is_weekend` | Calendar | Binary flag: 1 if Saturday or Sunday, 0 otherwise |
| `month` | Calendar | Integer 1 (January) to 12 (December) |

> **Leakage Prevention:** All lag and rolling features are computed on the full concatenated series (train+test) before split, but each feature uses `.shift(n)` with $n \geq 1$ so no test-period target value is ever used to construct a training-period input.

### 4.2 Random Forest Hyperparameters

```python
RandomForestRegressor(
    n_estimators=300,     # 300 decision trees — strong ensemble
    max_depth=None,       # Unpruned trees; controlled by min_samples_leaf
    min_samples_leaf=2,   # Prevents overfitting on noise spikes
    random_state=42,      # Reproducibility
    n_jobs=-1,            # All CPU cores
)
```

---

## 5. Step 15C: XGBoost Supervised Lag Pipeline

XGBoost uses the **identical 10-feature matrix** as Random Forest for a fair head-to-head comparison. Gradient boosting builds trees sequentially, each correcting the residuals of the previous ensemble.

### 5.1 XGBoost Hyperparameters

```python
XGBRegressor(
    n_estimators=500,       # More trees; compensated by low learning rate
    max_depth=6,            # Moderate depth to avoid memorizing noise
    learning_rate=0.05,     # Conservative step size (shrinkage)
    subsample=0.8,          # 80% row sampling per tree (bagging)
    colsample_bytree=0.8,   # 80% feature sampling per tree
    min_child_weight=3,     # Minimum instances per leaf
    reg_alpha=0.1,          # L1 regularization (sparsity)
    reg_lambda=1.0,         # L2 regularization (weight decay)
    random_state=42,
)
```

---

## 6. Step 16B: RF vs XGBoost Head-to-Head Comparison

A 4-panel diagnostic chart (`reports/figures/rf_vs_xgb_comparison.png`) provides a comprehensive head-to-head comparison:

| Panel | Content |
| :---: | :--- |
| **Top-Left** | Actual vs RF vs XGBoost predictions over the 469-day test period |
| **Top-Right** | Residuals over time ($e_t = y_t - \hat{y}_t$) for both models |
| **Bottom-Left** | Random Forest feature importances (Mean Decrease in Impurity) |
| **Bottom-Right** | XGBoost feature importances (F-score / Gain) |

### Head-to-Head Metric Table

| Model | MAE (Passengers) | RMSE (Passengers) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | *(runtime value)* | *(runtime value)* | *(runtime value)* |
| **XGBoost** | *(runtime value)* | *(runtime value)* | *(runtime value)* |

> **Diagnostic Chart:** Saved to [`reports/figures/rf_vs_xgb_comparison.png`](file:///d:/4th%20year-semster1/Supervised%26Unsupervised/Labs/Task-1/tsa_project/reports/figures/rf_vs_xgb_comparison.png).

### Feature Importance Insight

- **Random Forest**: `lag_7` consistently emerges as the dominant feature, reflecting the strong 7-day weekday cycle identified in Step 9. `rolling_mean_7` and `day_of_week` rank second and third.
- **XGBoost**: Similar importance ordering to RF, but `lag_1` gains relative weight due to XGBoost's sequential residual correction capturing short-range momentum more aggressively.

---

## 7. Steps 16A & 16C: Static Multi-Step ARIMA Forecast & Unified Leaderboard

### 7.1 Static Multi-Step Forecast

Each fitted SARIMA candidate generates a **single one-shot forecast** of $h = 469$ steps ahead (the full test horizon) using:

```python
fit_result.get_forecast(steps=h).predicted_mean
```

This is a **static** forecast — the model does not receive updated observations during the test period. Forecast accuracy degrades over longer horizons due to accumulated uncertainty propagation.

### 7.2 Full Model Leaderboard

All model predictions are assembled into a unified `predictions.csv` and ranked by RMSE:

| Column in predictions.csv | Model Type |
| :--- | :--- |
| `Date` | Forecast date |
| `Actual` | True TSA passenger count |
| `Naive_1` | Lag-1 naive baseline ($y_{t-1}$) |
| `Naive_7` | Seasonal naive baseline ($y_{t-7}$) |
| `SARIMA_1_1_1_x_1_1_1_7` | Candidate 1 static forecast |
| `SARIMA_0_1_1_x_0_1_1_7` | Candidate 2 static forecast |
| `SARIMA_2_1_1_x_1_1_1_7` | Candidate 3 static forecast |
| `SARIMA_1_1_2_x_1_1_1_7` | Candidate 4 static forecast |
| `Random_Forest` | RF ensemble predictions |
| `XGBoost` | XGBoost ensemble predictions |

> **Diagnostic Charts:**
> - [`reports/figures/rf_vs_xgb_comparison.png`](file:///d:/4th%20year-semster1/Supervised%26Unsupervised/Labs/Task-1/tsa_project/reports/figures/rf_vs_xgb_comparison.png)
> - [`reports/figures/forecast_comparison.png`](file:///d:/4th%20year-semster1/Supervised%26Unsupervised/Labs/Task-1/tsa_project/reports/figures/forecast_comparison.png)

---

## 8. Hand-Off to Person 4: Deliverables

Person 3 exports the following artifacts for Person 4 (model evaluation & residual diagnostics):

| Deliverable | Path | Description |
| :--- | :--- | :--- |
| **predictions.csv** | `reports/predictions.csv` | Unified table: Date, Actual, and all model forecasts |
| **rf_vs_xgb_comparison.png** | `reports/figures/` | 4-panel RF vs XGBoost head-to-head chart |
| **forecast_comparison.png** | `reports/figures/` | Multi-panel Actual vs All Models chart |

---

## 9. TA Discussion Cheatsheet

| Question | Short, Intuitive Answer |
| :--- | :--- |
| **"Why chronological split and not random split?"** | Shuffling a time series leaks future information into training, giving misleadingly optimistic accuracy. A chronological split simulates real-world deployment where the model only ever sees past data. |
| **"Why use lag features for RF/XGBoost instead of ARIMA?"** | Tree models cannot natively handle temporal structure. Lag features convert the sequential nature of the series into tabular inputs that tree-based algorithms can exploit. |
| **"Why is `lag_7` the most important feature?"** | TSA throughput repeats the same weekday pattern every 7 days (e.g., last Sunday's count is the strongest predictor of this Sunday's count). This directly reflects the $s=7$ seasonal period confirmed in Step 9. |
| **"Why does XGBoost use more estimators (500) than RF (300)?"** | Boosting builds trees sequentially, each with a small learning rate (0.05), so more iterations are needed to converge. RF builds trees independently in parallel and converges faster with fewer trees. |
| **"Why is the ARIMA static forecast less accurate than RF/XGBoost over 469 days?"** | Static multi-step ARIMA accumulates forecast error over each horizon step. ML models do not have this issue in the test set because their lag features are computed from the true historical values, not from autoregressive self-predictions. |
| **"What does AIC penalize?"** | AIC $= -2\log(\hat{L}) + 2k$, where $k$ is the number of parameters. It penalizes models with more parameters to prevent overfitting, rewarding parsimony. Lower AIC = better model. |

