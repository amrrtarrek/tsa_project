"""Step 18: Residual Analysis & Diagnostic Checks (Plot, ACF, Stats)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from .config import FIG_DIR

def compute_residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "skewness": float(pd.Series(residuals).skew()),
        "kurtosis": float(pd.Series(residuals).kurtosis()),
    }

def plot_residual_diagnostics(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, 
                              model_name: str = "Winning Model", 
                              save_filename: str = "residual_diagnostics.png") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    
    # Panel 1: Residuals over time
    axes[0].plot(dates, residuals, color="tab:purple", lw=0.8, label=f"Residuals ({model_name})")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"Residual Series over Time: e_t = y_t - y_hat_t ({model_name})")
    axes[0].set_ylabel("Error (Passengers)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")
    
    # Panel 2: ACF of Residuals
    n_lags = 28
    acf_vals = acf(residuals, nlags=n_lags, fft=True)
    lags = np.arange(len(acf_vals))
    
    # 95% Confidence Interval for White Noise ACF bounds
    conf_bound = 1.96 / np.sqrt(len(residuals))
    
    axes[1].stem(lags, acf_vals, basefmt=" ")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhline(conf_bound, color="red", linestyle="--", alpha=0.7, label="95% CI Bound")
    axes[1].axhline(-conf_bound, color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Autocorrelation Function (ACF) of Residuals (Lags 0 to 28)")
    axes[1].set_xlabel("Lag (Days)")
    axes[1].set_ylabel("ACF")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
