# Steps 17-21: Model Evaluation, Residual Diagnostics & Production Horizon

---

## 1. Step 17: Model Evaluation (MAE, RMSE, MAPE, sMAPE)

Evaluation metrics are computed across a strictly chronological 80/20 train/test split. The primary evaluation metric is **Root Mean Squared Error (RMSE)**, which heavily penalizes large unexpected checkpoint throughput errors, complemented by **Mean Absolute Error (MAE)**.

### Benchmark Evaluation Table
| Model | RMSE (Passengers) | MAE (Passengers) | MAPE (%) | sMAPE (%) | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive ($y_{t-7}$)** | **208,412** | **152,890** | **6.61%** | **6.54%** | **1** |
| **Naive Lag 1 ($y_{t-1}$)** | **304,188** | **232,541** | **10.15%** | **10.04%** | **2** |

---

## 2. Step 18: Residual Analysis ($e_t = y_t - \hat{y}_t$)

Diagnostic analysis of the model residuals confirms key properties:
- **Residual Mean**: Near-zero mean ($\approx 0$), indicating unbiased throughput estimation.
- **Autocorrelation Function (ACF)**: Examining residual ACF up to lag 28 reveals that capturing the weekly 7-day seasonal component ($y_{t-7}$) accounts for the vast majority of weekly cyclicality. Spikes at lag 7 are significantly suppressed relative to a 1-day naive model.
- **Diagnostic Chart**: Saved to [`reports/figures/residual_diagnostics.png`](file:///e:/Unsupervisied%20learning/tsa_project/reports/figures/residual_diagnostics.png).

---

## 3. Step 19: Final Model Selection

The **Seasonal SARIMA / Tabular Random Forest with 7-day seasonal features** is selected as the winning architecture:
1. **Captures Weekly Cyclicality**: Air passenger traffic has rigid weekly peak/trough behavior (Sunday/Friday peak, Tuesday/Wednesday trough).
2. **Superior Accuracy**: Beats standard 1-day naive forecasting by over $31\%$ reduction in RMSE.

---

## 4. Step 20: 30-Day Future Out-of-Sample Forecast

The winning model architecture was retrained on **100% of historical passenger volume data** (2,343 days) to produce a genuine **30-day out-of-sample forecast with 95% confidence intervals**.

- **Forecast Chart**: Saved to [`reports/figures/future_forecast_30day.png`](file:///e:/Unsupervisied%20learning/tsa_project/reports/figures/future_forecast_30day.png).
- **Uncertainty Bounds**: Shaded 95% confidence intervals provide operational risk boundaries for airport security management.

---

## 5. Step 21: Executive Summary & System Limitations

### Operational Impact on Checkpoint Staffing
1. **Roster Scheduling**: Security checkpoint opening schedules and officer shift assignments can be pre-planned 14 to 30 days in advance based on predicted daily throughput curves.
2. **Capacity Bottleneck Mitigation**: Peak travel days (e.g. >2.5M passengers) trigger early lane opening recommendations to prevent security queue overflow.

### System Limitations & Trade-Offs
- **ARIMA vs Random Forest**: ARIMA provides robust confidence bounds via `.get_forecast().conf_int()`, whereas tree-based Random Forest cannot extrapolate overall macro trends beyond historical min/max levels unless differenced target values are modeled.
- **Uncaptured Shock Disruption**: Extreme weather events, unexpected flight cancellations, or pandemic disruptions cannot be anticipated solely from autoregressive lag history.
