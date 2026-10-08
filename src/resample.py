"""Step 6: Resampling daily -> weekly and noise comparison."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .config import COUNT_COL, FIG_DIR


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    weekly = daily[COUNT_COL].resample("W").sum().to_frame()
    # drop first/last week if incomplete (fewer than 7 days)
    days = daily[COUNT_COL].resample("W").count()
    return weekly[days == 7]


def compare_noise(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    cv = lambda s: s.std() / s.mean()
    pc = lambda s: s.pct_change().abs().mean()
    out = pd.DataFrame({
        "Daily":  [cv(daily[COUNT_COL]), pc(daily[COUNT_COL])],
        "Weekly": [cv(weekly[COUNT_COL]), pc(weekly[COUNT_COL])],
    }, index=["Coef. of variation", "Mean |% change| step to step"])
    print(out.round(4))
    return out


def plot_daily_vs_weekly(daily, weekly) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(daily.index, daily[COUNT_COL], lw=0.7)
    ax[0].set_title("Daily TSA passenger throughput")
    ax[1].plot(weekly.index, weekly[COUNT_COL], color="tab:orange")
    ax[1].set_title("Weekly total (resample 'W')")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "daily_vs_weekly.png", dpi=150)
    plt.close()
