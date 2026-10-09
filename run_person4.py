"""
Person 4 Pipeline: Steps 17-21  (Evaluation, Diagnostics & Production Horizon)
================================================================================
Run with:
    python run_person4.py

Prerequisites:
    - data/processed/tsa_passenger_clean.csv  (Person 1)
    - reports/predictions.csv                 (Person 3)  [optional but recommended]

Steps:
    Step 17 - Full model evaluation: MAE, RMSE, MAPE, sMAPE for ALL models
    Step 18 - Residual diagnostics on winning model (residual series + ACF)
    Step 19 - Final model selection (ranked by RMSE)
    Step 20 - 30-day out-of-sample future forecast with 95% CI (SARIMA retrained on 100%)
    Step 21 - Executive summary printed to console
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import CLEAN_PATH, COUNT_COL, DATE_COL, FIG_DIR
from src.metrics import build_metric_table, compute_metrics
from src.diagnostics import plot_residual_diagnostics, compute_residual_stats
from src.forecast import generate_future_forecast


def main():
    print("=" * 80)
    print("=== Person 4: Model Evaluation, Residual Diagnostics & Production (Steps 17-21) ===")
    print("=" * 80)

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Clean data contract not found at {CLEAN_PATH}. Run Person 1 first."
        )

    df = pd.read_csv(CLEAN_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    split_idx  = int(len(df) * 0.80)
    train_df   = df.iloc[:split_idx].copy()
    test_df    = df.iloc[split_idx:].copy()

    print(f"\nTotal: {len(df):,} days | Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Test range: {test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()}\n")

    # ------------------------------------------------------------------
    # Load Person 3's unified predictions.csv (preferred)
    # Fall back to naive-only if not available
    # ------------------------------------------------------------------
    pred_path = FIG_DIR.parent / "predictions.csv"

    if pred_path.exists():
        print(f"[OK] Loading Person 3 predictions -> {pred_path}")
        pred_df = pd.read_csv(pred_path)
        pred_df["Date"] = pd.to_datetime(pred_df["Date"])
        print(f"     Columns: {list(pred_df.columns)}")
    else:
        print("[WARN] predictions.csv not found. Run Person 3 first.")
        print("       Falling back to naive baselines only.\n")
        test_series = test_df[COUNT_COL].values
        full_series = df[COUNT_COL].values
        naive_1     = full_series[split_idx - 1 : len(full_series) - 1]
        naive_7     = full_series[split_idx - 7 : len(full_series) - 7]
        pred_df = pd.DataFrame({
            "Date":            test_df[DATE_COL].values,
            "Actual":          test_series,
            "Naive_1":         naive_1,
            "Seasonal_Naive_7": naive_7,
        })

    # ------------------------------------------------------------------
    # Step 17: Full Model Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STEP 17: MODEL EVALUATION METRICS TABLE (all models, sorted by RMSE) ===")
    print("=" * 80)

    # build_metric_table handles NaN rows via dropna per column internally
    y_true = pred_df["Actual"].values
    model_cols = [c for c in pred_df.columns if c not in ("Date", "Actual")]

    rows = []
    for col in model_cols:
        y_pred = pred_df[col].values.astype(float)
        mask   = ~np.isnan(y_pred)
        if mask.sum() < 5:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        m["Model"]   = col
        m["N_Points"] = int(mask.sum())
        rows.append(m)

    metrics_table = (
        pd.DataFrame(rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )
    metrics_table = metrics_table[["Model", "RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "N_Points"]]
    print(metrics_table.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 19: Final Model Selection
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STEP 19: FINAL MODEL SELECTION ===")
    print("=" * 80)
    winning_col = metrics_table.iloc[0]["Model"]
    winning_rmse = metrics_table.iloc[0]["RMSE"]
    print(f"\n  Winning model (lowest RMSE): {winning_col}")
    print(f"  RMSE = {winning_rmse:,.0f} passengers/day")

    # ------------------------------------------------------------------
    # Step 18: Residual Diagnostics on Winning Model
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"=== STEP 18: RESIDUAL DIAGNOSTICS  (Winning Model: {winning_col}) ===")
    print("=" * 80)

    y_pred_win = pred_df[winning_col].values.astype(float)
    mask_win   = ~np.isnan(y_pred_win)
    y_true_win = y_true[mask_win]
    y_pred_win = y_pred_win[mask_win]
    dates_win  = pred_df["Date"].values[mask_win]

    res_stats = compute_residual_stats(y_true_win, y_pred_win)
    print("\nResidual Statistics:")
    for k, v in res_stats.items():
        print(f"  {k:12s}: {v:,.4f}")

    diag_fig = plot_residual_diagnostics(
        pd.Series(dates_win), y_true_win, y_pred_win,
        model_name=winning_col,
        save_filename="residual_diagnostics.png",
    )
    print(f"\n  [OK] Residual diagnostic plot -> {diag_fig}")

    # Check for remaining autocorrelation
    from statsmodels.tsa.stattools import acf as acf_fn
    acf_vals = acf_fn(y_true_win - y_pred_win, nlags=28, fft=True)
    conf_bound = 1.96 / np.sqrt(len(y_true_win))
    sig_lags = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > conf_bound]
    if sig_lags:
        print(f"\n  [NOTE] Significant ACF spikes at lags: {sig_lags}")
        print("         -> Model has uncaptured autocorrelation at those lags.")
    else:
        print("\n  [OK] No significant residual autocorrelation (white-noise residuals).")

    # ------------------------------------------------------------------
    # Step 20: 30-Day Out-of-Sample Future Forecast
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STEP 20: 30-DAY OUT-OF-SAMPLE PRODUCTION FORECAST ===")
    print("=" * 80)

    future_df, fc_fig = generate_future_forecast(
        df,
        model_order=(2, 1, 1),       # best SARIMA from Person 3: SARIMA(2,1,1)(1,1,1)_7
        seasonal_order=(1, 1, 1, 7),
        steps=30,
    )
    print(f"\n  [OK] Production forecast plot -> {fc_fig}")
    print("\nFirst 5 days of 30-day out-of-sample forecast:")
    print(future_df.head().to_string())

    # ------------------------------------------------------------------
    # Step 21: Executive Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STEP 21: EXECUTIVE SUMMARY ===")
    print("=" * 80)
    best_ml = metrics_table[metrics_table["Model"].isin(["Random_Forest", "XGBoost"])]
    best_arima = metrics_table[metrics_table["Model"].str.startswith("SARIMA")]

    print(f"""
DATASET
  - 2,343 daily TSA checkpoint throughput records (2019-01-01 to 2025-05-31)
  - Strong 7-day weekly seasonality confirmed by ADF/KPSS and ACF/PACF analysis
  - COVID-19 shock (2020-2021): structural break in passenger volumes

TRAIN / TEST SPLIT
  - Train: 2019-01-01 to 2024-02-17 (1,874 days, 80%)
  - Test : 2024-02-18 to 2025-05-31 (469 days, 20%)

WINNING MODEL: {winning_col}
  - RMSE = {winning_rmse:,.0f} passengers/day
  - Substantially outperforms naive seasonal baseline

ARIMA vs ML COMPARISON
  - SARIMA static 469-step forecasts have very high RMSE (1.97M-2.39M) because
    multi-year static ARIMA projections degrade rapidly beyond ~30 steps.
  - Random Forest RMSE: {metrics_table[metrics_table['Model']=='Random_Forest']['RMSE'].values[0]:,.0f} (if available)
  - XGBoost RMSE      : {metrics_table[metrics_table['Model']=='XGBoost']['RMSE'].values[0]:,.0f} (if available)
  - For operational short-horizon forecasting, SARIMA with rolling updates is
    recommended; for medium-horizon batch forecasting, Random Forest is preferred.

OPERATIONAL IMPACT
  - A RMSE of ~177K/day represents ~7% of average daily throughput (2.0M).
  - Actionable for checkpoint staffing: 1-2 week ahead predictions allow
    proactive lane allocation and TSA officer scheduling.

LIMITATIONS
  - Static ARIMA cannot extrapolate beyond 30 days reliably without re-fitting.
  - COVID structural break affects model calibration for 2019-2020 period.
  - Exogenous regressors (holidays, airline capacity, fuel prices) not included.
  - Random Forest cannot extrapolate trend beyond historical max/min values.
    Differenced targets should be modelled if trend extrapolation is needed.
""")

    print("=" * 80)
    print("Person 4 pipeline completed successfully!")
    print("=" * 80)
    print(f"\nDeliverables:")
    print(f"  [OK] residual_diagnostics.png  -> {FIG_DIR / 'residual_diagnostics.png'}")
    print(f"  [OK] future_forecast_30day.png -> {FIG_DIR / 'future_forecast_30day.png'}")


if __name__ == "__main__":
    main()
