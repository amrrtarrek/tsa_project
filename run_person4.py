"""Person 4 Pipeline: Steps 17-21. Run: python run_person4.py"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import CLEAN_PATH, COUNT_COL, DATE_COL
from src.metrics import build_metric_table, compute_metrics
from src.diagnostics import plot_residual_diagnostics, compute_residual_stats
from src.forecast import generate_future_forecast

def main():
    print("=== Person 4: Model Evaluation, Residual Diagnostics & Production Horizon (Steps 17 to 21) ===")
    
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Clean data contract not found at {CLEAN_PATH}. Please run Person 1 first.")
        
    df = pd.read_csv(CLEAN_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    # Chronological 80/20 split for testing evaluation pipeline
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Total dataset: {len(df)} days | Train: {len(train_df)} days | Test: {len(test_df)} days")
    print(f"Test date range: {test_df[DATE_COL].min().strftime('%Y-%m-%d')} to {test_df[DATE_COL].max().strftime('%Y-%m-%d')}\n")
    
    # Generate baseline predictions on test set
    test_series = test_df[COUNT_COL].values
    full_series = df[COUNT_COL].values
    
    # Naive lag 1
    naive_1 = full_series[split_idx - 1 : len(full_series) - 1]
    
    # Seasonal Naive lag 7
    seasonal_naive_7 = full_series[split_idx - 7 : len(full_series) - 7]
    
    # Build unified prediction DataFrame
    pred_df = pd.DataFrame({
        "Date": test_df[DATE_COL].values,
        "Actual": test_series,
        "Naive_1": naive_1,
        "Seasonal_Naive_7": seasonal_naive_7,
    })
    
    # Step 17: Side-by-side metric comparison
    metrics_table = build_metric_table(pred_df, target_col="Actual")
    print("=== STEP 17: MODEL EVALUATION METRICS TABLE ===")
    print(metrics_table.to_string(index=False))
    print()
    
    # Winning baseline model selection
    winning_model_col = metrics_table.iloc[0]["Model"]
    print(f"=== STEP 19: WINNING BASELINE ARCHITECTURE: {winning_model_col} ===")
    
    # Step 18: Residual Diagnostics
    y_true = pred_df["Actual"].values
    y_pred = pred_df[winning_model_col].values
    
    res_stats = compute_residual_stats(y_true, y_pred)
    print("\n=== STEP 18: RESIDUAL DIAGNOSTIC STATISTICS ===")
    for k, v in res_stats.items():
        print(f"{k:12s}: {v:,.4f}")
        
    diag_fig = plot_residual_diagnostics(pred_df["Date"], y_true, y_pred, 
                                         model_name=winning_model_col, 
                                         save_filename="residual_diagnostics.png")
    print(f"Saved residual diagnostic plot to: {diag_fig}")
    
    # Step 20: 30-Day Out-of-Sample Production Forecast
    print("\n=== STEP 20: GENERATING 30-DAY OUT-OF-SAMPLE FUTURE FORECAST ===")
    future_df, fc_fig = generate_future_forecast(df, model_order=(1,1,1), seasonal_order=(1,1,1,7), steps=30)
    print(f"Saved production forecast plot to: {fc_fig}")
    print("\nFirst 5 days of 30-day out-of-sample forecast:")
    print(future_df.head().to_string())

if __name__ == '__main__':
    main()
