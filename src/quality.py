"""Steps 3-4: Data Understanding & Data Quality checks."""
import pandas as pd
from .config import DATE_COL, COUNT_COL


def understand(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print(df.dtypes, "\n")
    print(df.head(), "\n")
    print(df.tail(), "\n")
    print(df[COUNT_COL].describe(), "\n")


def quality_report(df: pd.DataFrame) -> dict:
    d = pd.to_datetime(df[DATE_COL], errors="coerce")
    full = pd.date_range(d.min(), d.max(), freq="D")
    missing_dates = full.difference(pd.DatetimeIndex(d.dropna().unique()))

    report = {
        "n_rows": len(df),
        "date_min": d.min(),
        "date_max": d.max(),
        "bad_dates": int(d.isna().sum()),
        "null_counts": int(df[COUNT_COL].isna().sum()),
        "negative_counts": int((df[COUNT_COL] < 0).sum()),
        "zero_counts": int((df[COUNT_COL] == 0).sum()),
        "duplicated_dates": int(d.duplicated().sum()),
        "missing_calendar_days": len(missing_dates),
        "newest_first": bool(d.iloc[0] > d.iloc[-1]),
    }
    print("=== DATA QUALITY REPORT ===")
    for k, v in report.items():
        print(f"{k:24s}: {v}")
    if len(missing_dates):
        print("First missing dates:", list(missing_dates[:10].date))
    return report
