"""Shared schema validation and cleaning helpers for factory CSV files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "power_kwh": "Power_kWh",
    "electricity_kwh": "Power_kWh",
    "energy_kwh": "Power_kWh",
    "usage_kwh": "Power_kWh",
    "production_meters": "Production_Meters",
    "meters_produced": "Production_Meters",
    "output_meters": "Production_Meters",
    "fabric_meters": "Production_Meters",
    "outage_hours": "Outage_Hours",
    "power_cut_hours": "Outage_Hours",
    "load_shedding_hours": "Outage_Hours",
    "diesel_cost_pkr": "Diesel_Cost_PKR",
    "gross_revenue_pkr": "Gross_Revenue_PKR",
}


def _key(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column aliases but do not guess production/outage values."""
    df = df.copy()
    rename: dict[object, str] = {}
    for col in df.columns:
        normalized = _key(col)
        rename[col] = COLUMN_ALIASES.get(normalized, str(col).strip())
    df = df.rename(columns=rename)
    # Avoid silently accepting duplicate canonical columns after alias mapping.
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate columns after standardization: {duplicated}")
    return df


def clean_factory_data(
    raw_df: pd.DataFrame,
    source_name: str = "uploaded CSV",
    require_production: bool = True,
    require_outage: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Return cleaned data and non-fatal warnings.

    Power_kWh is required. Production_Meters is normally required for textile
    efficiency benchmarking. Outage_Hours is optional for inference but required
    to train the project's five-feature outage-aware model.
    """
    if raw_df.empty:
        raise ValueError(f"{source_name} contains no rows.")
    df = standardize_columns(raw_df)
    warnings: list[str] = []

    required = ["Power_kWh"]
    if require_production:
        required.append("Production_Meters")
    if require_outage:
        required.append("Outage_Hours")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required column(s): {', '.join(missing)}. "
            "See README.md for the expected CSV schema."
        )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        bad_dates = int(df["Date"].isna().sum())
        if bad_dates:
            warnings.append(f"Dropped {bad_dates} row(s) with unparseable/missing Date values.")
            df = df[df["Date"].notna()].copy()
        df = df.sort_values("Date")
    else:
        warnings.append("No Date column found; row order is treated as chronological.")

    numeric_cols = [
        c for c in ["Power_kWh", "Production_Meters", "Outage_Hours", "Diesel_Cost_PKR", "Gross_Revenue_PKR"]
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mandatory_numeric = ["Power_kWh"] + (["Production_Meters"] if require_production else [])
    before = len(df)
    df = df.dropna(subset=mandatory_numeric).copy()
    dropped = before - len(df)
    if dropped:
        warnings.append(f"Dropped {dropped} row(s) with missing/non-numeric required values.")

    for col in numeric_cols:
        if (df[col].dropna() < 0).any():
            raise ValueError(f"{col} contains negative values; please check the source data.")

    if require_production and "Production_Meters" in df.columns:
        zero_prod = int((df["Production_Meters"] <= 0).sum())
        if zero_prod:
            warnings.append(
                f"{zero_prod} row(s) have zero production; those rows are excluded from kWh/metre efficiency metrics."
            )
    if "Outage_Hours" not in df.columns:
        warnings.append("Outage_Hours is absent; outage-related analysis is incomplete and the ML forecast will assume 0 outage hours.")
    elif df["Outage_Hours"].isna().any():
        missing_outages = int(df["Outage_Hours"].isna().sum())
        warnings.append(
            f"{missing_outages} row(s) have missing/non-numeric Outage_Hours; values were not invented and those rows are excluded from model training."
        )

    if len(df) < 1:
        raise ValueError(f"No usable rows remain in {source_name} after cleaning.")
    return df.reset_index(drop=True), warnings


def load_reference_files(reference_dir: str | Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load compatible textile reference CSVs placed in data/reference/."""
    folder = Path(reference_dir)
    result: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    if not folder.exists():
        return result, warnings
    for path in sorted(folder.glob("*.csv")):
        try:
            raw = pd.read_csv(path)
            df, row_warnings = clean_factory_data(raw, path.name, require_production=True, require_outage=False)
            if "Outage_Hours" not in df.columns:
                # Keep for benchmarking; training script will skip this source.
                pass
            result[path.stem.replace("_", " ").title()] = df
            warnings.extend([f"{path.name}: {warning}" for warning in row_warnings])
        except Exception as exc:  # one malformed file should not prevent loading other references
            warnings.append(f"Could not load reference file {path.name}: {exc}")
    return result, warnings


def numeric_summary(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")
