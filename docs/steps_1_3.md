# Steps 1-3: Problem, Data, Understanding

## 1. Problem Definition
- **Problem:** forecast daily number of passengers screened at US TSA checkpoints.
- **Business value:** airports/TSA can plan checkpoint staffing, lanes and shifts
  ahead of busy days (holidays, weekends) and avoid long queues or wasted labor.
- **Target variable:** y = daily passenger throughput (`Passenger_Count`).
- **Metrics:** MAE (average error in passengers), RMSE (penalizes big errors).
- **Horizon:** 14-30 days ahead.

## 2. Data Acquisition
- Source: official TSA "Passenger Volumes" (tsa.gov/travel/passenger-volumes)
  or the Kaggle "TSA passenger throughput" CSV. Saved as `data/raw/tsa_raw.csv`.
- Granularity: daily, one row per date, from 2019 until today.

## 3. Data Understanding
- **Raw Dataset:** 2,188 daily entries aggregated from official TSA checkpoint throughput files.
- **Clean Continuous Dataset:** 2,343 days (`freq='D'`) after handling 155 missing calendar days via time-based linear interpolation.
- **Date Range:** 2019-01-01 to 2025-05-31.
- **Mean Daily Throughput:** 1,910,317 passengers.
- **Min Daily Throughput:** 87,708 passengers (April 2020 COVID-19 trough).
- **Max Daily Throughput:** 3,024,622 passengers (All-time peak travel day).
- **Expected Patterns:** Strong 7-day cyclical pattern (Sundays/Fridays peak, Tuesdays/Wednesdays trough), yearly holiday seasonality, and significant noise reduction when resampled to weekly totals (step-to-step mean change drops from 12.21% daily to 5.30% weekly).
