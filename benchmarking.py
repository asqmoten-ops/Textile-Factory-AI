"""Peer benchmarking and explainable factory-stage assessment."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _specific_energy(df: pd.DataFrame) -> pd.Series:
    if not {"Power_kWh", "Production_Meters"}.issubset(df.columns):
        return pd.Series(dtype=float)
    energy = pd.to_numeric(df["Power_kWh"], errors="coerce")
    output = pd.to_numeric(df["Production_Meters"], errors="coerce")
    ratio = energy / output.where(output > 0)
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def _weighted_intensity(df: pd.DataFrame) -> float:
    power = pd.to_numeric(df.get("Power_kWh", pd.Series(dtype=float)), errors="coerce")
    production = pd.to_numeric(df.get("Production_Meters", pd.Series(dtype=float)), errors="coerce")
    mask = power.notna() & production.notna() & (production > 0)
    if not mask.any():
        return float("nan")
    return float(power[mask].sum() / production[mask].sum())


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def assess_factory(
    target_name: str,
    target_df: pd.DataFrame,
    peer_factories: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Compute recent KPIs and a relative energy-intensity stage.

    Benchmark results are only comparable when peers measure energy and output in
    equivalent intervals and units. The stage is a relative ranking, not a universal
    industry certification.
    """
    peer_factories = peer_factories or {}
    df = target_df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")
    recent = df.tail(7)
    previous = df.iloc[max(0, len(df) - 14):max(0, len(df) - 7)]

    intensity = _weighted_intensity(recent)
    overall_intensity = _weighted_intensity(df)
    recent_production = _safe_mean(recent.get("Production_Meters", pd.Series(dtype=float)))
    previous_production = _safe_mean(previous.get("Production_Meters", pd.Series(dtype=float)))
    recent_power = _safe_mean(recent.get("Power_kWh", pd.Series(dtype=float)))
    previous_power = _safe_mean(previous.get("Power_kWh", pd.Series(dtype=float)))
    recent_outage = _safe_mean(recent.get("Outage_Hours", pd.Series(dtype=float)))

    if pd.notna(previous_production) and previous_production > 0 and pd.notna(recent_production):
        production_change_pct = (recent_production / previous_production - 1.0) * 100
    else:
        production_change_pct = float("nan")

    peers: list[tuple[str, float]] = []
    for peer_name, peer_df in peer_factories.items():
        if peer_name == target_name:
            continue
        peer_intensity = _weighted_intensity(peer_df.tail(7))
        if math.isfinite(peer_intensity) and peer_intensity >= 0:
            peers.append((peer_name, peer_intensity))

    # Compare against each peer's recent seven-row weighted kWh/metre. This expects
    # all datasets to use compatible reporting intervals and production units.
    intensity_values = [v for _, v in peers]
    percentile_rank = float(100 * np.mean(np.asarray(intensity_values) <= intensity)) if intensity_values and math.isfinite(intensity) else float("nan")
    peer_median = float(np.median(intensity_values)) if intensity_values else float("nan")

    if len(df) < 14:
        stage = "Insufficient history"
        stage_reason = "At least 14 usable observations are recommended for a recent-versus-prior trend check."
    elif not peers:
        stage = "Baseline established — peers needed"
        stage_reason = "No compatible peer benchmark is available yet; upload comparable factories or add verified files to data/reference/."
    elif not math.isfinite(intensity):
        stage = "Production data needed"
        stage_reason = "Energy per metre cannot be calculated because valid production measurements are unavailable."
    elif percentile_rank <= 25:
        stage = "Leading relative efficiency"
        stage_reason = "Recent kWh per metre is in the lowest-intensity quartile of the available peer sample."
    elif percentile_rank <= 50:
        stage = "Better than peer median"
        stage_reason = "Recent kWh per metre is at or below the middle of the available peer sample."
    elif percentile_rank <= 75:
        stage = "Needs monitoring"
        stage_reason = "Recent kWh per metre is higher than the median of the available peer sample."
    else:
        stage = "High energy intensity"
        stage_reason = "Recent kWh per metre is in the highest-intensity quartile of the available peer sample. Investigate context before drawing conclusions."

    if not math.isfinite(production_change_pct):
        production_trend = "Not enough prior-period production data"
    elif production_change_pct <= -5:
        production_trend = "Production down materially"
    elif production_change_pct >= 5:
        production_trend = "Production up materially"
    else:
        production_trend = "Production broadly stable"

    return {
        "factory_name": target_name,
        "observations": int(len(df)),
        "recent_avg_power_kwh": round(recent_power, 2) if math.isfinite(recent_power) else None,
        "recent_avg_production_m": round(recent_production, 2) if math.isfinite(recent_production) else None,
        "recent_avg_outage_hours": round(recent_outage, 2) if math.isfinite(recent_outage) else None,
        "recent_kwh_per_meter": round(intensity, 6) if math.isfinite(intensity) else None,
        "overall_kwh_per_meter": round(overall_intensity, 6) if math.isfinite(overall_intensity) else None,
        "production_change_pct": round(production_change_pct, 2) if math.isfinite(production_change_pct) else None,
        "production_trend": production_trend,
        "peer_count": len(peers),
        "peer_median_kwh_per_meter": round(peer_median, 6) if math.isfinite(peer_median) else None,
        "peer_percentile_rank": round(percentile_rank, 1) if math.isfinite(percentile_rank) else None,
        "stage": stage,
        "stage_reason": stage_reason,
        "peer_values": [{"factory": name, "kwh_per_meter": round(val, 6)} for name, val in peers],
        "benchmark_caveat": "Peer ranking is relative to the supplied sample. Compare only factories with compatible processes, production units, reporting intervals and product mixes.",
    }
