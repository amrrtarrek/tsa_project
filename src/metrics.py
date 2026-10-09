"""Step 17: Reusable evaluation metrics (MAE, RMSE, MAPE, sMAPE)."""
import numpy as np
import pandas as pd

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
        "sMAPE (%)": smape(y_true, y_pred),
    }

def build_metric_table(predictions_df: pd.DataFrame, target_col: str = "Actual") -> pd.DataFrame:
    """Computes evaluation metrics for all model columns in predictions_df against target_col."""
    model_cols = [c for c in predictions_df.columns if c not in [target_col, "Date", "date"]]
    y_true = predictions_df[target_col].values
    
    rows = []
    for m in model_cols:
        y_pred = predictions_df[m].values
        res = compute_metrics(y_true, y_pred)
        res["Model"] = m
        rows.append(res)
        
    res_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    res_df = res_df[["Model", "RMSE", "MAE", "MAPE (%)", "sMAPE (%)"]]
    return res_df
