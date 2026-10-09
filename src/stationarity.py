"""Steps 10-13: Autocorrelation, Stationarity Tests (ADF/KPSS), Differencing & Model Orders."""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .config import DATE_COL, COUNT_COL, FIG_DIR


def run_stationarity_tests(series: pd.Series, series_name: str) -> dict:
    """Executes both Augmented Dickey-Fuller (ADF) and KPSS tests on a series."""
    clean_s = series.dropna().astype(float)
    
    # 1. Augmented Dickey-Fuller (ADF)
    # H0: Unit root present (Non-stationary)
    # H1: Stationary (Reject H0 if p < 0.05)
    adf_res = adfuller(clean_s, autolag="AIC")
    adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
    
    # 2. KPSS Test
    # H0: Stationary around a constant level
    # H1: Unit root present (Non-stationary) (Reject H0 if p < 0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_res = kpss(clean_s, regression="c", nlags="auto")
    kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
    
    # Joint Interpretation Verdict
    if adf_p < 0.05 and kpss_p >= 0.05:
        status = "Stationary (Both agree)"
    elif adf_p >= 0.05 and kpss_p < 0.05:
        status = "Non-Stationary (Both agree)"
    elif adf_p < 0.05 and kpss_p < 0.05:
        status = "Difference / Structural Break (ADF says Stat, KPSS rejects)"
    else:
        status = "Inconclusive (ADF fails to reject, KPSS fails to reject)"
        
    return {
        "Series": series_name,
        "ADF Stat": round(adf_stat, 4),
        "ADF p-value": round(adf_p, 4),
        "KPSS Stat": round(kpss_stat, 4),
        "KPSS p-value": round(kpss_p, 4),
        "Conclusion": status,
    }


def build_stationarity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a comprehensive comparison table for Raw, Regular Diff, Seasonal Diff, and Combined Diff."""
    s = df[COUNT_COL].astype(float) if COUNT_COL in df.columns else df.iloc[:, 0].astype(float)
    
    diff_reg = s.diff(1)               # First-order regular diff (d=1)
    diff_sea = s.diff(7)               # Seasonal diff (D=1, s=7)
    diff_comb = s.diff(7).diff(1)       # Combined diff (d=1, D=1, s=7)
    
    tests = [
        run_stationarity_tests(s, "Raw Series (X_t)"),
        run_stationarity_tests(diff_reg, "Regular Diff (d=1, X_t - X_{t-1})"),
        run_stationarity_tests(diff_sea, "Seasonal Diff (D=1, X_t - X_{t-7})"),
        run_stationarity_tests(diff_comb, "Combined Diff (d=1, D=1)"),
    ]
    return pd.DataFrame(tests)


def plot_differencing_and_acf_pacf(df: pd.DataFrame, max_lags: int = 28, 
                                   save_filename: str = "stationarity_differencing_acf_pacf.png") -> Path:
    """Step 10, 12, 13: 4-row diagnostic grid displaying Series, ACF, and PACF across transformations."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    s = df[COUNT_COL].astype(float) if COUNT_COL in df.columns else df.iloc[:, 0].astype(float)
    dates = pd.to_datetime(df[DATE_COL]) if DATE_COL in df.columns else df.index
    
    transforms = [
        ("Raw Series (X_t)", s, "#0f172a"),
        ("First Difference (d=1: X_t - X_{t-1})", s.diff(1), "#0284c7"),
        ("Seasonal Difference (D=1: X_t - X_{t-7})", s.diff(7), "#059669"),
        ("Combined Difference (d=1, D=1)", s.diff(7).diff(1), "#7c3aed"),
    ]
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    
    for row_idx, (title, series_data, color) in enumerate(transforms):
        valid_series = series_data.dropna()
        valid_dates = dates[-len(valid_series):]
        
        # Column 1: Time Plot
        axes[row_idx, 0].plot(valid_dates, valid_series.values / 1e6, color=color, lw=0.7)
        axes[row_idx, 0].set_title(f"{title} - Time Series", fontsize=10, fontweight="bold")
        axes[row_idx, 0].set_ylabel("Value (M)", fontsize=9)
        axes[row_idx, 0].grid(True, linestyle="--", alpha=0.3)
        if row_idx == 3:
            axes[row_idx, 0].set_xlabel("Date", fontsize=9)
            
        # Column 2: Autocorrelation Function (ACF)
        plot_acf(valid_series, ax=axes[row_idx, 1], lags=max_lags, color=color, vlines_kwargs={"colors": color})
        axes[row_idx, 1].set_title(f"{title} - ACF (Lags 1-{max_lags})", fontsize=10, fontweight="bold")
        axes[row_idx, 1].set_ylabel("Autocorrelation", fontsize=9)
        axes[row_idx, 1].grid(True, linestyle="--", alpha=0.3)
        if row_idx == 3:
            axes[row_idx, 1].set_xlabel("Lag (Days)", fontsize=9)

        # Column 3: Partial Autocorrelation Function (PACF)
        plot_pacf(valid_series, ax=axes[row_idx, 2], lags=max_lags, method="ywm", color=color, vlines_kwargs={"colors": color})
        axes[row_idx, 2].set_title(f"{title} - PACF (Lags 1-{max_lags})", fontsize=10, fontweight="bold")
        axes[row_idx, 2].set_ylabel("Partial Autocorrelation", fontsize=9)
        axes[row_idx, 2].grid(True, linestyle="--", alpha=0.3)
        if row_idx == 3:
            axes[row_idx, 2].set_xlabel("Lag (Days)", fontsize=9)

    plt.tight_layout()
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def get_recommended_arima_orders() -> pd.DataFrame:
    """Generates the recommended candidate orders for Person 3 based on diagnostic conclusions."""
    recommendations = [
        {
            "Candidate": "Candidate 1 (Parsimonious Baseline)",
            "Order (p, d, q)": "(1, 1, 1)",
            "Seasonal (P, D, Q)_7": "(1, 1, 1, 7)",
            "Statistical Rationale": "Balances AR(1) daily momentum and MA(1) shock absorption with seasonal (1,1,1)_7 to capture the dominant weekly cycle."
        },
        {
            "Candidate": "Candidate 2 (Airline Model Variant)",
            "Order (p, d, q)": "(0, 1, 1)",
            "Seasonal (P, D, Q)_7": "(0, 1, 1, 7)",
            "Statistical Rationale": "Pure Moving Average specification after first + seasonal differencing; highly stable and avoids AR parameter over-fitting."
        },
        {
            "Candidate": "Candidate 3 (Extended AR Component)",
            "Order (p, d, q)": "(2, 1, 1)",
            "Seasonal (P, D, Q)_7": "(1, 1, 1, 7)",
            "Statistical Rationale": "Captures 2-day passenger booking and checkpoint momentum along with the seasonal weekly pattern."
        },
        {
            "Candidate": "Candidate 4 (Extended MA Component)",
            "Order (p, d, q)": "(1, 1, 2)",
            "Seasonal (P, D, Q)_7": "(1, 1, 1, 7)",
            "Statistical Rationale": "Accounts for potential 2-day residual error moving average persistence."
        },
        {
            "Candidate": "Candidate 5 (Non-Seasonal Benchmark)",
            "Order (p, d, q)": "(1, 1, 1)",
            "Seasonal (P, D, Q)_7": "None",
            "Statistical Rationale": "Non-seasonal baseline to formally demonstrate the empirical accuracy boost gained by incorporating s=7."
        },
    ]
    return pd.DataFrame(recommendations)
