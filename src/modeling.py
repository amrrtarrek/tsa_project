"""
Person 3 - Steps 14-16: Modeling & ML Forecasting Engine
=========================================================
Modular functions for:
  Step 14  - Chronological Train/Test Split
  Step 15A - ARIMA / SARIMA candidate grid
  Step 15B - Random Forest supervised lag pipeline
  Step 15C - XGBoost supervised lag pipeline  [NEW]
  Step 16  - Static multi-step & rolling one-step-ahead forecasting
  Extra    - RF vs XGBoost head-to-head comparison  [NEW]
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARNING] xgboost not installed. Run: pip install xgboost")

from .config import DATE_COL, COUNT_COL, FIG_DIR

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# STEP 14: Chronological Train / Test Split
# ---------------------------------------------------------------------------

def chronological_split(df: pd.DataFrame, train_ratio: float = 0.80) -> tuple:
    """Strictly chronological split - no shuffling ever."""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    print("=" * 70)
    print("STEP 14: CHRONOLOGICAL TRAIN / TEST SPLIT (80/20)")
    print("=" * 70)
    print(f"  Total rows   : {len(df):,} days")
    print(f"  Train period : {train_df[DATE_COL].min().date()} to "
          f"{train_df[DATE_COL].max().date()}  ({len(train_df):,} days)")
    print(f"  Test period  : {test_df[DATE_COL].min().date()} to "
          f"{test_df[DATE_COL].max().date()}  ({len(test_df):,} days)")
    return train_df, test_df


# ---------------------------------------------------------------------------
# STEP 15A: ARIMA / SARIMA Candidate Grid
# ---------------------------------------------------------------------------

ARIMA_CANDIDATES = [
    {
        "name": "SARIMA_1_1_1_x_1_1_1_7",
        "label": "Candidate 1: SARIMA(1,1,1)(1,1,1)_7",
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 7),
    },
    {
        "name": "SARIMA_0_1_1_x_0_1_1_7",
        "label": "Candidate 2: SARIMA(0,1,1)(0,1,1)_7  [Airline]",
        "order": (0, 1, 1),
        "seasonal_order": (0, 1, 1, 7),
    },
    {
        "name": "SARIMA_2_1_1_x_1_1_1_7",
        "label": "Candidate 3: SARIMA(2,1,1)(1,1,1)_7",
        "order": (2, 1, 1),
        "seasonal_order": (1, 1, 1, 7),
    },
    {
        "name": "SARIMA_1_1_2_x_1_1_1_7",
        "label": "Candidate 4: SARIMA(1,1,2)(1,1,1)_7",
        "order": (1, 1, 2),
        "seasonal_order": (1, 1, 1, 7),
    },
    # Candidate 5 removed: non-seasonal ARIMA on 1,874 days is too slow
    # for interactive use. Included as reference in report only.
]


def fit_arima_models(train_df: pd.DataFrame, candidates: list = None) -> dict:
    """
    Fits each ARIMA/SARIMA candidate on the training series.
    Returns dict: {name -> {label, fit_result, aic, order, seasonal_order}}
    """
    if candidates is None:
        candidates = ARIMA_CANDIDATES

    train_series = (
        train_df.set_index(DATE_COL)[COUNT_COL].astype(float)
        if DATE_COL in train_df.columns
        else train_df.iloc[:, 0].astype(float)
    )

    print("\n" + "=" * 70)
    print("STEP 15A: ARIMA / SARIMA CANDIDATE GRID FITTING")
    print("=" * 70)

    fitted_models = {}
    aic_rows = []

    for cand in candidates:
        name    = cand["name"]
        label   = cand["label"]
        order   = cand["order"]
        s_order = cand.get("seasonal_order")

        print(f"\n  Fitting {label} ...", end=" ", flush=True)
        try:
            mdl = SARIMAX(
                train_series,
                order=order,
                seasonal_order=s_order if s_order else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit_res = mdl.fit(disp=False)
            aic = round(fit_res.aic, 2)
            print(f"AIC = {aic:,.2f}")
            fitted_models[name] = {
                "label": label,
                "fit_result": fit_res,
                "aic": aic,
                "order": order,
                "seasonal_order": s_order,
            }
            aic_rows.append({"Model": label, "AIC": aic})
        except Exception as exc:
            print(f"FAILED ({exc})")

    if aic_rows:
        aic_df = pd.DataFrame(aic_rows).sort_values("AIC").reset_index(drop=True)
        print("\n  AIC Comparison Table (lower is better):")
        print(aic_df.to_string(index=False))

    return fitted_models


# ---------------------------------------------------------------------------
# Feature Engineering (shared by RF and XGBoost)
# ---------------------------------------------------------------------------

def build_supervised_features(series: pd.Series) -> pd.DataFrame:
    """
    Builds the supervised feature matrix for tree-based models.

    Lag features  : lag_1, lag_2, lag_3, lag_7, lag_14
    Rolling stats : rolling_mean_7, rolling_std_7  (shifted by 1 to avoid leakage)
    Calendar      : day_of_week, is_weekend, month

    IMPORTANT: computed on the FULL series before split to avoid data leakage.
    NaN rows from initial lags must be dropped from TRAIN only (not test).
    """
    df = pd.DataFrame({"y": series})
    df.index = pd.to_datetime(df.index)

    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["rolling_mean_7"] = df["y"].shift(1).rolling(7).mean()
    df["rolling_std_7"]  = df["y"].shift(1).rolling(7).std()

    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"]  = (df.index.dayofweek >= 5).astype(int)
    df["month"]       = df.index.month

    return df


FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_std_7",
    "day_of_week", "is_weekend", "month",
]


def _prepare_xy(train_df, test_df):
    """Build X/y arrays for both train and test from the full series."""
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL])
    full_series = full_df.set_index(DATE_COL)[COUNT_COL].astype(float)

    feat_df = build_supervised_features(full_series)

    train_dates = pd.to_datetime(train_df[DATE_COL])
    test_dates  = pd.to_datetime(test_df[DATE_COL])

    train_feat = feat_df.loc[feat_df.index.isin(train_dates)].dropna()
    test_feat  = feat_df.loc[feat_df.index.isin(test_dates)]

    X_train = train_feat[FEATURE_COLS].values
    y_train = train_feat["y"].values
    X_test  = test_feat[FEATURE_COLS].values
    y_test  = test_feat["y"].values

    return X_train, y_train, X_test, y_test, train_feat, test_feat


# ---------------------------------------------------------------------------
# STEP 15B: Random Forest Pipeline
# ---------------------------------------------------------------------------

def rf_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Builds and trains the Random Forest regression pipeline."""
    X_train, y_train, X_test, y_test, train_feat, test_feat = _prepare_xy(
        train_df, test_df
    )

    print("\n" + "=" * 70)
    print("STEP 15B: RANDOM FOREST SUPERVISED LAG PIPELINE")
    print("=" * 70)
    print(f"  Features         : {FEATURE_COLS}")
    print(f"  Training samples : {len(X_train):,}  (after NaN lag drop)")
    print(f"  Test samples     : {len(X_test):,}")

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    mae  = mean_absolute_error(y_test, rf_preds)
    rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    print(f"  RF Test MAE  : {mae:,.0f}")
    print(f"  RF Test RMSE : {rmse:,.0f}")

    return {
        "model": rf,
        "feature_cols": FEATURE_COLS,
        "train_feat": train_feat,
        "test_feat": test_feat,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test,  "y_test": y_test,
        "rf_test_pred": rf_preds,
        "mae": mae, "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# STEP 15C: XGBoost Pipeline  [NEW]
# ---------------------------------------------------------------------------

def xgb_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Builds and trains an XGBoost regression pipeline using the same lag/
    rolling/calendar features as the Random Forest pipeline for a fair comparison.
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    X_train, y_train, X_test, y_test, train_feat, test_feat = _prepare_xy(
        train_df, test_df
    )

    print("\n" + "=" * 70)
    print("STEP 15C: XGBOOST SUPERVISED LAG PIPELINE")
    print("=" * 70)
    print(f"  Features         : {FEATURE_COLS}")
    print(f"  Training samples : {len(X_train):,}  (after NaN lag drop)")
    print(f"  Test samples     : {len(X_test):,}")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    xgb_preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, xgb_preds)
    rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"  XGB Test MAE  : {mae:,.0f}")
    print(f"  XGB Test RMSE : {rmse:,.0f}")

    return {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "train_feat": train_feat,
        "test_feat": test_feat,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test,  "y_test": y_test,
        "xgb_test_pred": xgb_preds,
        "mae": mae, "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# RF vs XGBoost Comparison  [NEW]
# ---------------------------------------------------------------------------

def compare_rf_xgb(
    rf_result: dict,
    xgb_result: dict,
    test_df: pd.DataFrame,
    save_filename: str = "rf_vs_xgb_comparison.png",
) -> pd.DataFrame:
    """
    Produces a comprehensive head-to-head comparison of Random Forest vs XGBoost:
      - Side-by-side metric table  (MAE, RMSE, MAPE)
      - 4-panel comparison chart:
          [0] Actual vs RF vs XGB predictions over time
          [1] Residuals (errors) over time for both models
          [2] RF feature importances
          [3] XGB feature importances
    Returns the metric comparison DataFrame.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    y_test   = rf_result["y_test"]
    rf_preds = rf_result["rf_test_pred"]
    xg_preds = xgb_result["xgb_test_pred"]

    test_dates = pd.to_datetime(test_df[DATE_COL].values)

    # --- Metric table ---
    def _metrics(y_true, y_pred, name):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        return {"Model": name, "MAE": round(mae, 0), "RMSE": round(rmse, 0),
                "MAPE (%)": round(mape, 2)}

    metrics_df = pd.DataFrame([
        _metrics(y_test, rf_preds, "Random Forest"),
        _metrics(y_test, xg_preds, "XGBoost"),
    ])
    # Determine winner
    winner = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]

    print("\n" + "=" * 70)
    print("RF vs XGBoost HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print(f"\n  Winner (lowest RMSE): {winner}")

    # --- 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Random Forest vs XGBoost: Head-to-Head Comparison",
                 fontsize=15, fontweight="bold", y=0.98)

    rf_color  = "#0284c7"   # blue
    xgb_color = "#dc2626"   # red
    act_color = "#374151"   # dark gray

    # Panel 0: Predictions vs Actual
    ax0 = axes[0, 0]
    ax0.plot(test_dates, y_test / 1e6,   color=act_color, lw=1.5, label="Actual",        alpha=0.85)
    ax0.plot(test_dates, rf_preds / 1e6, color=rf_color,  lw=1.5, label="Random Forest", linestyle="--")
    ax0.plot(test_dates, xg_preds / 1e6, color=xgb_color, lw=1.5, label="XGBoost",       linestyle=":")
    ax0.set_title("Actual vs Predicted (Test Set)", fontweight="bold")
    ax0.set_ylabel("Passengers (Millions)")
    ax0.legend(fontsize=9)
    ax0.grid(True, linestyle="--", alpha=0.3)
    ax0.tick_params(axis="x", rotation=30)

    # Panel 1: Residuals over time
    ax1 = axes[0, 1]
    rf_resid  = y_test - rf_preds
    xgb_resid = y_test - xg_preds
    ax1.plot(test_dates, rf_resid / 1e3,  color=rf_color,  lw=1.2, label="RF Residuals",  alpha=0.8)
    ax1.plot(test_dates, xgb_resid / 1e3, color=xgb_color, lw=1.2, label="XGB Residuals", alpha=0.8)
    ax1.axhline(0, color="black", lw=0.8, linestyle="--")
    ax1.set_title("Residuals Over Time  (Actual - Predicted)", fontweight="bold")
    ax1.set_ylabel("Residual (Thousands)")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)

    # Panel 2: RF Feature Importances
    ax2 = axes[1, 0]
    rf_imp = pd.Series(
        rf_result["model"].feature_importances_,
        index=FEATURE_COLS,
    ).sort_values()
    bars2 = ax2.barh(rf_imp.index, rf_imp.values, color=rf_color, edgecolor="white")
    ax2.bar_label(bars2, fmt="{:.3f}", padding=3, fontsize=8)
    ax2.set_title("Random Forest Feature Importances", fontweight="bold")
    ax2.set_xlabel("Mean Decrease in Impurity")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.3)

    # Panel 3: XGBoost Feature Importances
    ax3 = axes[1, 1]
    xgb_imp = pd.Series(
        xgb_result["model"].feature_importances_,
        index=FEATURE_COLS,
    ).sort_values()
    bars3 = ax3.barh(xgb_imp.index, xgb_imp.values, color=xgb_color, edgecolor="white")
    ax3.bar_label(bars3, fmt="{:.3f}", padding=3, fontsize=8)
    ax3.set_title("XGBoost Feature Importances", fontweight="bold")
    ax3.set_xlabel("F-score (Gain)")
    ax3.grid(True, axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved comparison chart -> {out_path}")

    return metrics_df


def compare_all_models(
    pred_df: pd.DataFrame,
    save_filename: str = "all_models_comparison.png",
) -> pd.DataFrame:
    """
    Computes MAE / RMSE / MAPE for every model column in pred_df
    and returns a table sorted by RMSE ascending.
    """
    model_cols = [c for c in pred_df.columns if c not in ("Date", "Actual")]
    y_true = pred_df["Actual"].values
    rows = []
    for col in model_cols:
        y_pred = pred_df[col].dropna().values
        y_act  = pred_df.loc[pred_df[col].notna(), "Actual"].values
        if len(y_pred) == 0:
            continue
        mae  = mean_absolute_error(y_act, y_pred)
        rmse = np.sqrt(mean_squared_error(y_act, y_pred))
        mape = np.mean(np.abs((y_act - y_pred) / np.maximum(y_act, 1))) * 100
        rows.append({"Model": col, "MAE": round(mae), "RMSE": round(rmse),
                     "MAPE (%)": round(mape, 2), "N_Test": len(y_pred)})
    result = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Naive Baselines
# ---------------------------------------------------------------------------

def naive_baseline(full_series: pd.Series, split_idx: int) -> tuple:
    """Naive lag-1 and seasonal lag-7 baselines."""
    vals    = full_series.values
    naive_1 = vals[split_idx - 1 : len(vals) - 1]
    naive_7 = vals[split_idx - 7 : len(vals) - 7]
    return naive_1, naive_7


# ---------------------------------------------------------------------------
# STEP 16A: Static Multi-Step Forecast (ARIMA)
# ---------------------------------------------------------------------------

def static_multistep_forecast(
    fitted_models: dict,
    test_df: pd.DataFrame,
    h: int = None,
) -> pd.DataFrame:
    """One-shot multi-step forecasts from each fitted ARIMA over the test horizon."""
    if h is None:
        h = len(test_df)

    test_dates   = pd.to_datetime(test_df[DATE_COL].values)
    static_fc_df = pd.DataFrame({"Date": test_dates})

    print("\n" + "=" * 70)
    print(f"STEP 16: STATIC MULTI-STEP ARIMA FORECAST  (h = {h} days)")
    print("=" * 70)

    for name, info in fitted_models.items():
        try:
            fc    = info["fit_result"].get_forecast(steps=h)
            preds = fc.predicted_mean.values[:h]
            static_fc_df[name] = preds
            print(f"  [OK] {info['label'][:65]}")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")

    return static_fc_df


# ---------------------------------------------------------------------------
# STEP 16B: Rolling One-Step-Ahead Forecast (optional, slow)
# ---------------------------------------------------------------------------

def rolling_forecast_engine(
    full_df: pd.DataFrame,
    best_order: tuple,
    best_seasonal_order,
    split_idx: int,
    max_steps: int = 50,        # default: only roll first 50 steps for speed
) -> np.ndarray:
    """
    Rolling one-step-ahead SARIMA forecast.
    max_steps limits how many steps to roll (use None for full test set).
    """
    full_df = full_df.copy()
    full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL])
    series   = full_df.set_index(DATE_COL)[COUNT_COL].astype(float)
    history  = series.iloc[:split_idx].tolist()
    test_vals = series.iloc[split_idx:].tolist()

    n = len(test_vals) if max_steps is None else min(max_steps, len(test_vals))
    rolling_preds = []

    print("\n" + "=" * 70)
    print(f"STEP 16B: ROLLING ONE-STEP-AHEAD FORECAST  ({n} steps)")
    print(f"          Order: {best_order}  |  Seasonal: {best_seasonal_order}")
    print("=" * 70)

    for step in range(n):
        actual = test_vals[step]
        hist_s = pd.Series(history)
        try:
            mdl = SARIMAX(
                hist_s,
                order=best_order,
                seasonal_order=best_seasonal_order if best_seasonal_order else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = mdl.fit(disp=False)
            pred = float(fit.get_forecast(steps=1).predicted_mean.iloc[0])
        except Exception:
            pred = history[-1]

        rolling_preds.append(pred)
        history.append(actual)

        if (step + 1) % 10 == 0 or step == n - 1:
            print(f"  Step {step+1:>4}/{n} done")

    return np.array(rolling_preds)


# ---------------------------------------------------------------------------
# STEP 16C: Build Unified Prediction DataFrame
# ---------------------------------------------------------------------------

def build_prediction_dataframe(
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    split_idx: int,
    fitted_models: dict,
    rf_result: dict,
    xgb_result: dict = None,
    rolling_arima_preds: np.ndarray = None,
    static_fc_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Assembles the unified prediction DataFrame for Person 4:
      Date | Actual | Naive_1 | Naive_7 | <ARIMA candidates> |
      Random_Forest | XGBoost | Rolling_ARIMA_Best (if run)
    """
    full_df = full_df.copy()
    full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL])
    full_series  = full_df.set_index(DATE_COL)[COUNT_COL].astype(float)
    test_series  = test_df[COUNT_COL].values
    test_dates   = pd.to_datetime(test_df[DATE_COL].values)
    n_test       = len(test_df)

    naive_1, naive_7 = naive_baseline(full_series, split_idx)

    pred_df = pd.DataFrame({
        "Date":    test_dates,
        "Actual":  test_series,
        "Naive_1": naive_1[:n_test],
        "Naive_7": naive_7[:n_test],
    })

    # Static ARIMA forecasts
    if static_fc_df is not None:
        for col in static_fc_df.columns:
            if col != "Date":
                pred_df[col] = static_fc_df[col].values[:n_test]

    # Random Forest predictions (map by date)
    _add_ml_preds(pred_df, test_dates, rf_result["test_feat"],
                  rf_result["rf_test_pred"], "Random_Forest")

    # XGBoost predictions
    if xgb_result is not None:
        _add_ml_preds(pred_df, test_dates, xgb_result["test_feat"],
                      xgb_result["xgb_test_pred"], "XGBoost")

    # Rolling one-step-ahead (optional)
    if rolling_arima_preds is not None:
        pad = np.full(n_test, np.nan)
        pad[:len(rolling_arima_preds)] = rolling_arima_preds
        pred_df["Rolling_ARIMA_Best"] = pad

    return pred_df


def _add_ml_preds(pred_df, test_dates, feat_df, raw_preds, col_name):
    """Helper: align ML predictions back to test dates by index."""
    preds_full = np.full(len(test_dates), np.nan)
    feat_dates = feat_df.index
    for i, d in enumerate(test_dates):
        mask = feat_dates == d
        if mask.any():
            idx = np.where(mask)[0][0]
            if idx < len(raw_preds):
                preds_full[i] = raw_preds[idx]
    pred_df[col_name] = preds_full


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_forecast_comparison(
    pred_df: pd.DataFrame,
    save_filename: str = "forecast_comparison.png",
) -> Path:
    """Multi-panel chart: Actual vs every model in pred_df."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model_cols = [c for c in pred_df.columns if c not in ("Date", "Actual")]
    n_models   = len(model_cols)
    n_cols     = 2
    n_rows     = (n_models + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharey=False)
    axes = axes.flatten()

    palette = ["#0284c7", "#dc2626", "#16a34a", "#7c3aed",
               "#ea580c", "#0891b2", "#be185d", "#854d0e"]

    dates  = pd.to_datetime(pred_df["Date"])
    actual = pred_df["Actual"].values

    for i, col in enumerate(model_cols):
        ax    = axes[i]
        preds = pred_df[col].values
        mask  = ~np.isnan(preds)
        ax.plot(dates, actual / 1e6, label="Actual", color="#374151", lw=1.5, alpha=0.8)
        ax.plot(dates[mask], preds[mask] / 1e6, label=col,
                color=palette[i % len(palette)], lw=1.5, linestyle="--")
        if mask.sum() > 0:
            mae  = mean_absolute_error(actual[mask], preds[mask])
            rmse = np.sqrt(mean_squared_error(actual[mask], preds[mask]))
            ax.set_title(f"{col}\nMAE={mae/1e3:.1f}K  RMSE={rmse/1e3:.1f}K",
                         fontsize=10, fontweight="bold")
        ax.set_ylabel("Passengers (M)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("TSA Passenger Throughput: Actual vs All Predictions (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved forecast comparison -> {out_path}")
    return out_path


def plot_rf_feature_importance(
    rf_model,
    feature_cols: list,
    save_filename: str = "rf_feature_importance.png",
) -> Path:
    """Horizontal bar chart of RF feature importances."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(imp.index, imp.values, color="#0284c7", edgecolor="white")
    ax.bar_label(bars, fmt="{:.3f}", padding=3, fontsize=9)
    ax.set_title("Random Forest Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path = FIG_DIR / save_filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved RF feature importance -> {out_path}")
    return out_path
