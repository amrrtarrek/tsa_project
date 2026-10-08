from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "tsa_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "tsa_passenger_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"

DATE_COL = "Date"
COUNT_COL = "Passenger_Count"
FREQ = "D"
