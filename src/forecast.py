"""Step 20: 100% Historical Data Retraining & Future Out-of-Sample 14 to 30-Day Forecast."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .config import COUNT_COL, DATE_COL, FIG_DIR

def generate_future_forecast(clean_df: pd.DataFrame, 
                             model_order: tuple = (1, 1, 1), 
                             seasonal_order: tuple = (1, 1, 1, 7), 
                             steps: int = 30, 
                             save_filename: str = "future_forecast_30day.png") -> tuple:
    """Retrains SARIMA architecture on 100% historical data and generates a steps-ahead out-of-sample forecast with 95% CI."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_df.copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.set_index(DATE_COL).sort_index()
        
    series = df[COUNT_COL].astype(float)
    
    print(f"Retraining SARIMA{model_order}x{seasonal_order} on 100% historical data ({len(series)} days)...")
    model = SARIMAX(series, order=model_order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False)
    fit_res = model.fit(disp=False)
    
    # Generate future forecast & 95% CI
    forecast_res = fit_res.get_forecast(steps=steps)
    fc_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)
    
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="D")
    
    fc_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": fc_mean.values,
        "Lower_95": conf_int.iloc[:, 0].values,
        "Upper_95": conf_int.iloc[:, 1].values,
    }).set_index("Date")
    
    # Plot historical tail + future forecast with shaded 95% CI
    plt.figure(figsize=(12, 6))
    tail_days = 90
    hist_tail = series.iloc[-tail_days:]
    
    plt.plot(hist_tail.index, hist_tail.values, label="Historical Actual (Last 90 Days)", color="tab:blue", lw=1.5)
    plt.plot(fc_df.index, fc_df["Forecast"], label=f"Future {steps}-Day Forecast", color="tab:red", lw=2, linestyle="--")
    plt.fill_between(fc_df.index, fc_df["Lower_95"], fc_df["Upper_95"], color="tab:red", alpha=0.2, label="95% Confidence Interval")
    
    plt.title(f"Production TSA Checkpoint Throughput: {steps}-Day Out-of-Sample Forecast")
    plt.xlabel("Date")
    plt.ylabel("Passenger Throughput")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return fc_df, out_path
