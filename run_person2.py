"""Person 2 Pipeline: Steps 7-13. Run: python run_person2.py"""
import pandas as pd
from pathlib import Path

from src.config import CLEAN_PATH, DATE_COL, COUNT_COL
from src.eda_trend import (
    compute_eda_summary,
    plot_trend_analysis,
    plot_seasonality,
    decompose_series,
)
from src.stationarity import (
    build_stationarity_table,
    plot_differencing_and_acf_pacf,
    get_recommended_arima_orders,
)


def main():
    print("=" * 80)
    print("=== Person 2: Statistical & Time Series Diagnostics (Steps 7 to 13) ===")
    print("=" * 80)

    # 1. Verify and Load Person 1 Data Contract
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Clean data contract not found at {CLEAN_PATH}. Please run Person 1 first.")

    df = pd.read_csv(CLEAN_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    print(f"\n[Contract Verified] Successfully loaded {len(df)} continuous daily records.")
    print(f"Date Span: {df[DATE_COL].min().strftime('%Y-%m-%d')} to {df[DATE_COL].max().strftime('%Y-%m-%d')}\n")

    # -------------------------------------------------------------
    # Step 7: Exploratory Data Analysis & Summary Statistics
    # -------------------------------------------------------------
    print("=== STEP 7: EXPLORATORY DATA ANALYSIS (EDA) & SUMMARY STATS ===")
    eda_stats = compute_eda_summary(df)
    for k, v in eda_stats.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:,.2f}")
        else:
            print(f"  {k:12s}: {v}")

    # -------------------------------------------------------------
    # Step 8: Trend Analysis (Centered Rolling Averages)
    # -------------------------------------------------------------
    print("\n=== STEP 8: TREND ANALYSIS (7-DAY & 30-DAY CENTERED ROLLING AVERAGES) ===")
    fig_trend = plot_trend_analysis(df, save_filename="eda_trend_rolling.png")
    print(f"  -> Generated and saved trend visualization to: {fig_trend.name}")

    # -------------------------------------------------------------
    # Step 9: Seasonality Analysis & Classical Decomposition (s=7)
    # -------------------------------------------------------------
    print("\n=== STEP 9: SEASONALITY ANALYSIS & DECOMPOSITION (s=7) ===")
    fig_season = plot_seasonality(df, save_filename="seasonality_day_month.png")
    print(f"  -> Generated day-of-week & monthly distribution plot to: {fig_season.name}")
    
    fig_decomp = decompose_series(df, model="additive", period=7, save_filename="seasonal_decomposition.png")
    print(f"  -> Generated classical additive decomposition plot (s=7) to: {fig_decomp.name}")

    # -------------------------------------------------------------
    # Steps 10, 12, 13: ACF / PACF & Differencing Analysis
    # -------------------------------------------------------------
    print("\n=== STEPS 10, 12, 13: AUTOCORRELATION & DIFFERENCING DIAGNOSTIC GRID ===")
    fig_acf_pacf = plot_differencing_and_acf_pacf(df, max_lags=28, save_filename="stationarity_differencing_acf_pacf.png")
    print(f"  -> Generated 4x3 Series / ACF / PACF diagnostic grid to: {fig_acf_pacf.name}")

    # -------------------------------------------------------------
    # Step 11: Stationarity Hypothesis Testing (ADF & KPSS)
    # -------------------------------------------------------------
    print("\n=== STEP 11: FORMAL STATIONARITY HYPOTHESIS TESTS (ADF & KPSS) ===")
    stat_table = build_stationarity_table(df)
    print(stat_table.to_string(index=False))

    # -------------------------------------------------------------
    # Hand-Off to Person 3: Recommended ARIMA / SARIMAX Orders
    # -------------------------------------------------------------
    print("\n=== HAND-OFF TO PERSON 3: RECOMMENDED CANDIDATE ARIMA / SARIMAX ORDERS ===")
    rec_orders = get_recommended_arima_orders()
    for _, row in rec_orders.iterrows():
        print(f"  * {row['Candidate']}:")
        print(f"      Non-Seasonal Order (p, d, q) : {row['Order (p, d, q)']}")
        print(f"      Seasonal Order (P, D, Q)_7   : {row['Seasonal (P, D, Q)_7']}")
        print(f"      Rationale                    : {row['Statistical Rationale']}")
        print()

    print("=" * 80)
    print("Person 2 pipeline executed successfully! Deliverables ready for TA discussion.")
    print("=" * 80)


if __name__ == "__main__":
    main()
