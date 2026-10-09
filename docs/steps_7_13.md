# Steps 7–13: Statistical & Time Series Diagnostics

---

## 1. Executive Summary: Person 2 Role

Person 2 is responsible for the core **statistical diagnostic framework** of the TSA passenger forecasting system. Leveraging the clean continuous data contract from Person 1 (`data/processed/tsa_passenger_clean.csv`), this phase isolates underlying trends, quantifies intra-week and intra-year seasonality, runs formal stationarity hypothesis tests (ADF and KPSS), and identifies optimal differencing orders ($d=1, D=1, s=7$) and candidate autoregressive/moving average orders for Person 3.

---

## 2. Step 7: Exploratory Data Analysis (EDA) & Summary Statistics

Daily TSA checkpoint volume represents the operational heartbeat of US domestic and international aviation. Across the continuous span from January 1, 2019 to May 31, 2025 (2,343 days), the throughput statistics are summarized below:

| Metric | Value (Passengers) | Statistical Interpretation |
| :--- | :---: | :--- |
| **Observation Count** | **2,343 days** | Continuous daily records (`freq='D'`). |
| **Mean Throughput** | **1,922,116** | Baseline expected daily throughput nationwide. |
| **Median Throughput** | **2,096,816** | Higher than mean due to COVID-19 left-skewed shock. |
| **Standard Deviation** | **649,067** | High dispersion driven by travel seasonality and pandemic collapse. |
| **Historical Minimum** | **87,708** | April 14, 2020 (Severe COVID-19 aviation trough). |
| **Historical Maximum** | **3,024,622** | All-time single-day passenger screening peak. |
| **Skewness** | **-1.120** | Negative skewness reflects the acute pandemic drop. |
| **Kurtosis** | **0.430** | Slight leptokurtic concentration with pandemic long tail. |

---

## 3. Step 8: Trend Analysis (Centered Rolling Averages)

Raw daily passenger series exhibit massive intra-week noise (e.g., oscillating by several hundred thousand passengers between Tuesday and Sunday). To isolate the long-term macroeconomic trend, we apply two moving windows:
* **7-Day Centered Rolling Average**: Completely filters out the 7-day cyclical oscillation while retaining weekly trajectory changes.
* **30-Day Centered Rolling Average**: Smooths monthly fluctuations to expose seasonal summer travel peaks and broad macroeconomic cycles.

> **Why Centered Rolling Averages?**  
> Unlike a standard trailing (backward-looking) moving average which introduces a lag of $\frac{window - 1}{2}$ days (e.g., lagging peaks by 15 days on a 30-day window), a centered moving average places the smoothed estimate directly over the actual calendar day of observation.

* **Diagnostic Chart**: Saved to [`reports/figures/eda_trend_rolling.png`](file:///d:/Senior%20typeshit%20Projects/Task%201%20nlp/reports/figures/eda_trend_rolling.png).

---

## 4. Step 9: Seasonality Analysis & Classical Decomposition

### 4.1 Intra-Week (Day-of-Week) Seasonality
Throughput analysis grouped by day of week reveals rigid operational patterns:
1. **Peak Checkpoint Days**: **Sunday** and **Friday** consistently observe the highest passenger volumes (average $>2.2\text{M}$ passengers), driven by weekend leisure travelers and business travel wrap-ups.
2. **Trough Checkpoint Days**: **Tuesday** and **Wednesday** observe the lowest volumes (average $<1.7\text{M}$ passengers).

### 4.2 Intra-Year (Monthly) Seasonality
Monthly aggregation shows distinct seasonal travel quarters:
* **Summer Peak**: June, July, and August observe peak travel volume (summer vacations).
* **Winter Dips & Holiday Spikes**: January and February observe seasonal lows, punctuated by sharp surges around Thanksgiving and Christmas/New Year.

### 4.3 Classical Additive Decomposition ($s=7$)
Applying additive decomposition ($Y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Residual}_t$) with period $s=7$:
* The **Seasonal component** exhibits a steady $\pm 450,000$ passenger intra-week wave.
* The **Additive model** is selected over multiplicative because post-2021 the cyclical amplitude remains roughly stable around baseline throughput.
* **Diagnostic Charts**:
  * [`reports/figures/seasonality_day_month.png`](file:///d:/Senior%20typeshit%20Projects/Task%201%20nlp/reports/figures/seasonality_day_month.png)
  * [`reports/figures/seasonal_decomposition.png`](file:///d:/Senior%20typeshit%20Projects/Task%201%20nlp/reports/figures/seasonal_decomposition.png)

---

## 5. Step 11: Formal Stationarity Testing (ADF & KPSS)

Time series modeling requires covariance stationarity (constant mean, constant variance, and autocovariance dependent only on lag distance). We employ two complementary tests:
1. **Augmented Dickey-Fuller (ADF)**:
   * $H_0$: Unit root present (Non-Stationary).
   * $H_1$: Series is Stationary ($p < 0.05$ rejects $H_0$).
2. **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)**:
   * $H_0$: Series is Level/Trend Stationary.
   * $H_1$: Unit root present ($p < 0.05$ rejects $H_0$, indicating Non-Stationary).

### Joint Stationarity Test Results
| Series Transformation | ADF Statistic | ADF $p$-value | KPSS Statistic | KPSS $p$-value | Diagnostic Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw Series ($X_t$)** | **-1.7543** | **0.4034** | **2.5352** | **< 0.01** | **Non-Stationary (Both agree)** |
| **Regular Diff ($d=1$: $X_t - X_{t-1}$)** | **-9.9058** | **< 0.0001** | **0.0551** | **> 0.10** | **Stationary (Both agree)** |
| **Seasonal Diff ($D=1$: $X_t - X_{t-7}$)** | **-8.9376** | **< 0.0001** | **0.0792** | **> 0.10** | **Stationary (Both agree)** |
| **Combined Diff ($d=1, D=1$)** | **-14.7398** | **< 0.0001** | **0.0118** | **> 0.10** | **Stationary (Strongest Stationarity)** |

> **Key Statistical Insight for TA Discussion:**  
> The raw series is unambiguously non-stationary due to the post-COVID upward trend and severe structural shift. Once regular differencing ($d=1$) or seasonal differencing ($D=1, s=7$) is applied, the series passes both tests with flying colors ($p_{\text{ADF}} < 0.0001$ and $p_{\text{KPSS}} > 0.10$).

---

## 6. Steps 10, 12, 13: Autocorrelation (ACF / PACF) & Differencing Analysis

The 4-stage diagnostic grid evaluates lags 1 through 28:
1. **Raw Series**:
   * **ACF**: Extremely slow decay with massive crests every 7 days (lags 7, 14, 21, 28) — hallmark of strong weekly seasonality and non-stationarity.
   * **PACF**: Dominant spike at lag 1 followed by recurring weekly spikes at lag 7.
2. **First Difference ($d=1$)**:
   * Removes macroeconomic trend.
   * **ACF & PACF**: Strongly preserves 7-day cyclical oscillations (negative lag-1 rebound, sharp positive spikes at lag 7 and lag 14). Confirms that regular differencing alone is insufficient for seasonal data.
3. **Seasonal Difference ($D=1, s=7$)**:
   * Completely eliminates the 7-day cycle. Lags 7, 14, 21 collapse into the white-noise band.
4. **Combined Difference ($d=1, D=1$)**:
   * Achieves optimal stationarity.
   * **ACF**: Negative autocorrelation at lag 1 and lag 7 (suggests non-seasonal MA(1) and seasonal MA(1) components).
   * **PACF**: Cuts off after lag 1 or 2.

* **Diagnostic Chart**: Saved to [`reports/figures/stationarity_differencing_acf_pacf.png`](file:///d:/Senior%20typeshit%20Projects/Task%201%20nlp/reports/figures/stationarity_differencing_acf_pacf.png).

---

## 7. Hand-Off to Person 3: Recommended Candidate Orders

Based on Box-Jenkins identification principles applied to the differenced series:
* **Integration Orders**: $d=1$ (regular diff) and $D=1$ with seasonal period $s=7$.
* **Autoregressive Terms**: $p \in \{0, 1, 2\}$, $P \in \{0, 1\}$.
* **Moving Average Terms**: $q \in \{1, 2\}$, $Q \in \{1\}$.

### Recommended 5 Candidate Specifications
| Candidate Name | Order $(p, d, q)$ | Seasonal Order $(P, D, Q)_7$ | Statistical Justification |
| :--- | :---: | :---: | :--- |
| **Candidate 1 (Parsimonious Baseline)** | **(1, 1, 1)** | **(1, 1, 1, 7)** | Balances AR(1) daily momentum and MA(1) shock absorption with seasonal $(1,1,1)_7$ cycle. |
| **Candidate 2 (Airline Model Variant)** | **(0, 1, 1)** | **(0, 1, 1, 7)** | Pure Moving Average specification on differenced series; avoids AR parameter over-fitting. |
| **Candidate 3 (Extended AR Component)** | **(2, 1, 1)** | **(1, 1, 1, 7)** | Models 2-day booking and travel inertia alongside weekly seasonality. |
| **Candidate 4 (Extended MA Component)** | **(1, 1, 2)** | **(1, 1, 1, 7)** | Captures 2-day residual error moving-average persistence. |
| **Candidate 5 (Non-Seasonal Benchmark)** | **(1, 1, 1)** | **None** | Non-seasonal baseline to quantify accuracy gains from modeling $s=7$. |

---

## 8. TA Discussion Cheatsheet

| Question | Short, Intuitive Answer |
| :--- | :--- |
| **"Why centered rolling average instead of trailing?"** | Trailing averages lag peaks and troughs by $(k-1)/2$ days. Centered averages place smoothed points right at the true calendar event, providing an unbiased visual trend. |
| **"Why is raw data non-stationary?"** | It has an upward recovery trend and a massive structural break in April 2020 due to COVID-19, violating the constant mean assumption. |
| **"Why period $s=7$?"** | Commercial aviation is rigidly structured on 7-day work/weekend cycles (peak on Friday/Sunday, trough on Tuesday/Wednesday). |
| **"Why additive instead of multiplicative decomposition?"** | The seasonal oscillations maintain a roughly constant magnitude ($\approx \pm 450\text{k}$) rather than expanding proportionally with passenger counts. |
| **"How did you choose candidate orders for Person 3?"** | Regular diff ($d=1$) removes trend, seasonal diff ($D=1, s=7$) removes weekly cycles. ACF/PACF of combined diff show negative correlation spikes at lag 1 and lag 7, indicating $(1,1,1)\times(1,1,1)_7$ or $(0,1,1)\times(0,1,1)_7$. |
