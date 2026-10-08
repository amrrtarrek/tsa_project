"""Step 2: Data Acquisition - load raw TSA throughput data into a DataFrame
with exactly two columns: Date, Passenger_Count (still RAW / uncleaned)."""
import pandas as pd
from .config import DATE_COL, COUNT_COL


def _to_number(s: pd.Series) -> pd.Series:
    # "2,345,678" -> 2345678 ; bad values -> NaN
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(),
                         errors="coerce")


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Case A: long format  (Date, Numbers)  <- Kaggle style
    if df.shape[1] == 2:
        df.columns = [DATE_COL, COUNT_COL]
        df[COUNT_COL] = _to_number(df[COUNT_COL])
        return df

    # Case B: wide format from tsa.gov  (Date, 2026, 2025, 2024, ...)
    year_cols = [c for c in df.columns if c.isdigit()]
    if year_cols:
        long = df.melt(id_vars=df.columns[0], value_vars=year_cols,
                       var_name="Year", value_name=COUNT_COL)
        # first col is like "1/1" or "1/1/2025": rebuild with the year column
        md = long[df.columns[0]].astype(str).str.split("/").str[:2].str.join("/")
        long[DATE_COL] = pd.to_datetime(md + "/" + long["Year"], errors="coerce")
        long[COUNT_COL] = _to_number(long[COUNT_COL])
        return long[[DATE_COL, COUNT_COL]]

    raise ValueError(f"Unknown raw format, columns = {list(df.columns)}")
