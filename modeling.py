"""Forecast-safe feature construction shared by training and inference."""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "Outage_Hours",
    "Production_Meters",
    "Power_Lag_1",
    "Power_Lag_7",
    "Rolling_Mean_7",
]


def make_supervised_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Create one-step-ahead training rows without using the target row in features."""
    work = df.copy()
    if "Date" in work.columns:
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.sort_values("Date")
    for col in ["Power_kWh", "Production_Meters", "Outage_Hours"]:
        if col not in work.columns:
            raise ValueError(f"Required training column '{col}' is missing.")
        work[col] = pd.to_numeric(work[col], errors="coerce")
    power = work["Power_kWh"]
    work["Power_Lag_1"] = power.shift(1)
    work["Power_Lag_7"] = power.shift(7)
    # Shift before rolling so a row's target value never enters its own feature.
    work["Rolling_Mean_7"] = power.shift(1).rolling(window=7, min_periods=7).mean()
    work = work.dropna(subset=FEATURE_COLUMNS + ["Power_kWh", "Production_Meters", "Outage_Hours"])
    work = work[(work["Power_kWh"] >= 0) & (work["Production_Meters"] >= 0) & (work["Outage_Hours"] >= 0)]
    return work.reset_index(drop=True)
