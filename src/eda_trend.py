"""Steps 7-9: Exploratory Data Analysis, Trend Analysis & Seasonality."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose

from .config import DATE_COL, COUNT_COL, FIG_DIR


def compute_eda_summary(df: pd.DataFrame) -> dict:
    """Computes summary statistics (mean, std, median, skewness, min, max) for throughput."""
    s = df[COUNT_COL].astype(float)
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
    }


def plot_trend_analysis(df: pd.DataFrame, save_filename: str = "eda_trend_rolling.png") -> Path:
    """Step 8: Plots raw time series overlaid with 7-day and 30-day centered rolling averages."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    series = df.set_index(DATE_COL)[COUNT_COL].astype(float) if DATE_COL in df.columns else df[COUNT_COL].astype(float)
    roll_7 = series.rolling(window=7, center=True).mean()
    roll_30 = series.rolling(window=30, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series.index, series.values / 1e6, label="Daily Throughput (Raw)", color="#94a3b8", alpha=0.55, lw=0.8)
    ax.plot(series.index, roll_7.values / 1e6, label="7-Day Centered Rolling Average (Weekly Trend)", color="#0284c7", lw=1.6)
    ax.plot(series.index, roll_30.values / 1e6, label="30-Day Centered Rolling Average (Monthly Trend)", color="#dc2626", lw=2.2)

    ax.set_title("TSA Checkpoint Passenger Volume: Underlying Trend Extraction (2019–2025)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Passengers (Millions)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_seasonality(df: pd.DataFrame, save_filename: str = "seasonality_day_month.png") -> Path:
    """Step 9: Analyzes intra-week cyclicality and intra-year monthly patterns."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    temp = df.copy()
    if DATE_COL in temp.columns:
        temp[DATE_COL] = pd.to_datetime(temp[DATE_COL])
    else:
        temp[DATE_COL] = temp.index

    temp["Day_Name"] = temp[DATE_COL].dt.day_name()
    temp["Day_Num"] = temp[DATE_COL].dt.dayofweek
    temp["Month_Num"] = temp[DATE_COL].dt.month
    temp["Month_Name"] = temp[DATE_COL].dt.strftime("%b")

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Day of Week Boxplot
    day_data = [temp[temp["Day_Name"] == d][COUNT_COL].values / 1e6 for d in days_order]
    bp1 = axes[0].boxplot(day_data, labels=[d[:3] for d in days_order], patch_artist=True)
    for box in bp1["boxes"]:
        box.set_facecolor("#bae6fd")
        box.set_edgecolor("#0284c7")
    for median in bp1["medians"]:
        median.set_color("#0f172a")
        median.set_linewidth(1.8)

    axes[0].set_title("Weekly Seasonality: Throughput by Day of Week", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Day of Week", fontsize=11)
    axes[0].set_ylabel("Passengers (Millions)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Highlight Friday and Sunday peak
    axes[0].annotate("Weekend Peak", xy=(5, 2.5), xytext=(4, 2.8),
                     arrowprops=dict(facecolor="#0284c7", shrink=0.08, width=1, headwidth=6),
                     fontweight="bold", color="#0369a1")

    # Panel 2: Monthly Mean & Spread Bar Chart
    month_stats = temp.groupby("Month_Num")[COUNT_COL].agg(["mean", "std"]).reindex(range(1, 13))
    x_pos = np.arange(12)
    axes[1].bar(x_pos, month_stats["mean"] / 1e6, yerr=month_stats["std"] / 1e6, capsize=4, 
                color="#cbd5e1", edgecolor="#475569", alpha=0.85, label="Monthly Mean (+/- 1 Std)")
    
    # Highlight summer months (June, July, August -> indices 5, 6, 7)
    for idx in [5, 6, 7]:
        axes[1].bar(x_pos[idx], month_stats.loc[idx + 1, "mean"] / 1e6, 
                    yerr=month_stats.loc[idx + 1, "std"] / 1e6, capsize=4, 
                    color="#fbbf24", edgecolor="#b45309", alpha=0.95)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(months_order)
    axes[1].set_title("Annual Seasonality: Average Monthly Throughput", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Month", fontsize=11)
    axes[1].set_ylabel("Passengers (Millions)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def decompose_series(df: pd.DataFrame, model: str = "additive", period: int = 7, 
                     save_filename: str = "seasonal_decomposition.png") -> Path:
    """Step 9: Runs classical seasonal_decompose (period=7) and saves publication-quality figure."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    series = df.set_index(DATE_COL)[COUNT_COL].astype(float) if DATE_COL in df.columns else df[COUNT_COL].astype(float)
    decomp = seasonal_decompose(series, model=model, period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(series.index, decomp.observed / 1e6, color="#0f172a", lw=0.9)
    axes[0].set_ylabel("Observed (M)")
    axes[0].set_title(f"Classical Time Series Decomposition (Model={model.capitalize()}, Period s={period})", 
                      fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(series.index, decomp.trend / 1e6, color="#dc2626", lw=1.8)
    axes[1].set_ylabel("Trend (M)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Plot seasonal component
    axes[2].plot(series.index, decomp.seasonal / 1e6, color="#0284c7", lw=0.8)
    axes[2].set_ylabel("Seasonal (M)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    # Plot residuals
    axes[3].scatter(series.index, decomp.resid / 1e6, color="#64748b", s=6, alpha=0.6)
    axes[3].axhline(0, color="black", linestyle="--", lw=1)
    axes[3].set_ylabel("Residual (M)")
    axes[3].set_xlabel("Date", fontsize=11)
    axes[3].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
