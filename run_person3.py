"""
Person 3 Pipeline: Steps 14-16  (Modeling & ML Forecasting Engine)
===================================================================
Run with:
    python run_person3.py

What this script does:
    Step 14  - Chronological 80/20 train/test split (no shuffling)
    Step 15A - Fits 4 SARIMA candidate models + records AIC scores
    Step 15B - Random Forest pipeline (lag/rolling/calendar features)
    Step 15C - XGBoost pipeline     (same features for fair comparison)
    Step 16A - Static multi-step forecasts from all SARIMA candidates
    Step 16B - RF vs XGBoost head-to-head comparison (4-panel chart)
    Step 16C - Full model leaderboard (all models ranked by RMSE)
    Exports  - reports/predictions.csv  (<- Person 4 input)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import CLEAN_PATH, DATE_COL, COUNT_COL, FIG_DIR
from src.modeling import (
    chronological_split,
    fit_arima_models,
    rf_pipeline,
    xgb_pipeline,
    static_multistep_forecast,
    build_prediction_dataframe,
    compare_rf_xgb,
    compare_all_models,
    plot_forecast_comparison,
    ARIMA_CANDIDATES,
)


def main():
    print("=" * 70)
    print(" PERSON 3 -- Steps 14-16: Modeling & ML Forecasting Engine")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 0. Verify data contract from Person 1
    # ------------------------------------------------------------------
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Data contract not found at:\n  {CLEAN_PATH}\n"
            "Please run Person 1 first:  python run_person1.py"
        )

    df = pd.read_csv(CLEAN_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"\n[OK] Loaded {len(df):,} daily records  "
          f"({df[DATE_COL].min().date()} to {df[DATE_COL].max().date()})")

    # ------------------------------------------------------------------
    # Step 14: Chronological Train / Test Split
    # ------------------------------------------------------------------
    train_df, test_df = chronological_split(df, train_ratio=0.80)
    split_idx = len(train_df)

    # ------------------------------------------------------------------
    # Step 15A: ARIMA / SARIMA Candidate Grid
    # ------------------------------------------------------------------
    fitted_models = fit_arima_models(train_df, candidates=ARIMA_CANDIDATES)

    # Best by AIC
    if fitted_models:
        best_name    = min(fitted_models, key=lambda k: fitted_models[k]["aic"])
        best_order   = fitted_models[best_name]["order"]
        best_s_order = fitted_models[best_name]["seasonal_order"]
        print(f"\n  Best ARIMA by AIC -> {fitted_models[best_name]['label']}")
    else:
        best_order, best_s_order = (1, 1, 1), (1, 1, 1, 7)

    # ------------------------------------------------------------------
    # Step 15B: Random Forest Pipeline
    # ------------------------------------------------------------------
    rf_result = rf_pipeline(train_df, test_df)

    # ------------------------------------------------------------------
    # Step 15C: XGBoost Pipeline
    # ------------------------------------------------------------------
    xgb_result = xgb_pipeline(train_df, test_df)

    # ------------------------------------------------------------------
    # Step 16A: Static Multi-Step ARIMA Forecasts
    # ------------------------------------------------------------------
    static_fc_df = static_multistep_forecast(fitted_models, test_df)

    # ------------------------------------------------------------------
    # Step 16B: RF vs XGBoost Head-to-Head Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 16B: RF vs XGBoost HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    ml_metrics = compare_rf_xgb(
        rf_result, xgb_result, test_df,
        save_filename="rf_vs_xgb_comparison.png",
    )

    # ------------------------------------------------------------------
    # Step 16C: Assemble Unified Prediction DataFrame
    # ------------------------------------------------------------------
    pred_df = build_prediction_dataframe(
        test_df=test_df,
        full_df=df,
        split_idx=split_idx,
        fitted_models=fitted_models,
        rf_result=rf_result,
        xgb_result=xgb_result,
        rolling_arima_preds=None,   # set to rolling_forecast_engine() output if needed
        static_fc_df=static_fc_df,
    )

    print("\n" + "=" * 70)
    print("STEP 16C: UNIFIED PREDICTION DATAFRAME  (first 5 rows)")
    print("=" * 70)
    print(pred_df.head().to_string(index=False))
    print(f"\n  Shape: {pred_df.shape}  |  Columns: {list(pred_df.columns)}")

    # ------------------------------------------------------------------
    # Full leaderboard
    # ------------------------------------------------------------------
    leaderboard = compare_all_models(pred_df)
    print("\n" + "=" * 70)
    print("FULL MODEL LEADERBOARD (sorted by RMSE, lower is better)")
    print("=" * 70)
    print(leaderboard.to_string(index=False))

    # ------------------------------------------------------------------
    # Export predictions.csv for Person 4
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred_out = FIG_DIR.parent / "predictions.csv"
    pred_df.to_csv(pred_out, index=False)
    print(f"\n  [Saved] predictions.csv -> {pred_out}")

    # ------------------------------------------------------------------
    # Forecast comparison plot (all models)
    # ------------------------------------------------------------------
    plot_forecast_comparison(pred_df, save_filename="forecast_comparison.png")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    winner = leaderboard.iloc[0]["Model"]
    print("\n" + "=" * 70)
    print(" PERSON 3 -- COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n  Best overall model (RMSE): {winner}")
    print("\nDeliverables produced:")
    print(f"  [OK] predictions.csv         -> {pred_out}")
    print(f"  [OK] rf_vs_xgb_comparison.png-> {FIG_DIR / 'rf_vs_xgb_comparison.png'}")
    print(f"  [OK] forecast_comparison.png -> {FIG_DIR / 'forecast_comparison.png'}")
    print(f"\n  Pass predictions.csv + fitted_models to Person 4.\n")


if __name__ == "__main__":
    main()
