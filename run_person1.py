"""Person 1 pipeline: Steps 1-6. Run:  python run_person1.py"""
from src.config import RAW_PATH, CLEAN_PATH, COUNT_COL
from src.data_loader import load_raw
from src.quality import understand, quality_report
from src.preprocess import clean_daily
from src.resample import to_weekly, compare_noise, plot_daily_vs_weekly

# Step 2-3: acquire + understand
raw = load_raw(RAW_PATH)
understand(raw)

# Step 4: quality
quality_report(raw)

# Step 5: dates/time
daily = clean_daily(raw)

# Step 6: resample
weekly = to_weekly(daily)
compare_noise(daily, weekly)
plot_daily_vs_weekly(daily, weekly)

# Export the data contract for Persons 2, 3, 4
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
daily.reset_index().to_csv(CLEAN_PATH, index=False)   # columns: Date, Passenger_Count
print("Saved:", CLEAN_PATH, daily.shape)
