# TSA Checkpoint Passenger Volume Forecasting

End-to-end Time Series Analysis (TSA) and forecasting pipeline for US Transportation Security Administration checkpoint passenger throughput.

## Project Structure & Roles

- **Person 1 (Steps 1–6)**: Pipeline Architect & Data Ingestion (`run_person1.py`, `docs/steps_1_3.md`)
  - Pulls/loads raw TSA data (`data/raw/tsa_raw.csv`)
  - QA audit, missing date interpolation, daily continuity (`freq='D'`)
  - Exports clean baseline contract: `data/processed/tsa_passenger_clean.csv`
  - Resamples daily to weekly and evaluates noise reduction

- **Person 2 (Steps 7–13)**: Statistical & Time Series Diagnostics (`run_person2.py`, `docs/steps_7_13.md`)
  - Summary statistics and 7-day / 30-day centered rolling trend extraction
  - Intra-week (day-of-week) and intra-year (monthly) seasonality & additive decomposition ($s=7$)
  - Formal stationarity hypothesis testing (ADF & KPSS) across raw and differenced series
  - Regular ($d=1$) and seasonal ($D=1, s=7$) differencing with 28-lag ACF/PACF grids
  - Recommends candidate ARIMA/SARIMAX orders to Person 3

- **Person 3 (Steps 14–16)**: Modeling & ML Forecasting Engine (`run_person3.py`, `src/modeling.py`)
  - Chronological 80/20 train/test split (no data leakage)
  - Fits SARIMA candidate models (from Person 2's ACF/PACF orders) and records AIC scores
  - Builds Random Forest & XGBoost ML pipelines with lag (1,2,3,7,14), rolling stats (mean/std 7-day), and calendar features
  - Generates detailed side-by-side performance and feature importance comparison between Random Forest and XGBoost (`rf_vs_xgb_comparison.png`)
  - Static multi-step forecasts (h=469 days) for all models
  - Exports unified `reports/predictions.csv` for Person 4 (columns: Date, Actual, Naive_1, Naive_7, SARIMA candidates, Random_Forest, XGBoost)

- **Person 4 (Steps 17–21)**: Evaluation, Diagnostics & Production Horizon (`run_person4.py`, `docs/steps_17_21.md`)
  - Benchmarks all models on test set (MAE, RMSE, MAPE, sMAPE)
  - Residual diagnostics ($e_t = y_t - \hat{y}_t$) and white-noise ACF checks on winning model
  - Retrains winning architecture on 100% historical data
  - Generates 30-day out-of-sample forecast with 95% confidence intervals

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Person 1 data ingestion:
   ```bash
   python run_person1.py
   ```

3. Run Person 2 statistical diagnostics:
   ```bash
   python run_person2.py
   ```

4. Run Person 3 modeling & forecasting engine:
   ```bash
   python run_person3.py
   ```
   *Note: The rolling one-step-ahead forecast (~469 SARIMA fits) takes 5–15 minutes.*

5. Run Person 4 evaluation & production forecast:
   ```bash
   python run_person4.py
   ```
