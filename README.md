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

- **Person 3 (Steps 14–16)**: Modeling & ML Forecasting Engine
  - Chronological 80/20 train/test split
  - Evaluates ARIMA / SARIMA candidate grid and tabular Random Forest pipeline

- **Person 4 (Steps 17–21)**: Evaluation, Diagnostics & Production Horizon (`run_person4.py`, `docs/steps_17_21.md`)
  - Benchmarks models on test set (MAE, RMSE, MAPE, sMAPE)
  - Residual diagnostics ($e_t = y_t - \hat{y}_t$) and white-noise ACF checks
  - Retrains winning SARIMA architecture on 100% historical data
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

4. Run Person 4 evaluation & production forecast:
   ```bash
   python run_person4.py
   ```
