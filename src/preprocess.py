"""Step 5: Date & Time preprocessing -> clean daily series with DatetimeIndex."""
import pandas as pd
import numpy as np
from .config import DATE_COL, COUNT_COL, FREQ


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    # duplicated dates -> keep the last one
    df = df.drop_duplicates(subset=DATE_COL, keep="last")

    # negative counts are impossible -> treat as missing
    df.loc[df[COUNT_COL] < 0, COUNT_COL] = np.nan

    df = df.set_index(DATE_COL).sort_index()          # sort_index: newest-first fix
    df = df.asfreq(FREQ)                              # inserts missing calendar days as NaN

    n_missing = int(df[COUNT_COL].isna().sum())
    df[COUNT_COL] = df[COUNT_COL].interpolate(method="time")   # NOT ffill
    df[COUNT_COL] = df[COUNT_COL].bfill().ffill()     # only for edge NaNs
    print(f"Interpolated {n_missing} missing values")

    df.index.name = DATE_COL
    assert df.index.is_monotonic_increasing
    assert df.index.freqstr == FREQ
    assert df[COUNT_COL].isna().sum() == 0
    return df
