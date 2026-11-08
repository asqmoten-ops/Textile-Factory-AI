"""Baseline statistical forecasting and synthetic demo telemetry."""
from __future__ import annotations

import numpy as np
import pandas as pd


class TextileForecaster:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_historical_dataset(self, days: int = 30) -> pd.DataFrame:
        """Generate DEMO-ONLY telemetry. It is not real-world training data."""
        if days < 1:
            raise ValueError("days must be at least 1")
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
        base_kwh = 2800.0
        base_production = 9600.0
        fabric_rate_pkr = 450.0
        kwh_noise = self.rng.normal(0, 180, days)
        outage_hours = self.rng.choice(
            [0.0, 1.5, 3.0, 4.5, 6.0], size=days, p=[0.2, 0.3, 0.3, 0.15, 0.05]
        )
        daily_kwh = np.maximum(1000, base_kwh - (outage_hours * 250) + kwh_noise)
        daily_production = np.maximum(3000, base_production - (outage_hours * 800) + (kwh_noise * 2))
        return pd.DataFrame({
            "Date": dates,
            "Power_kWh": np.round(daily_kwh, 2),
            "Production_Meters": np.round(daily_production, 2),
            "Outage_Hours": outage_hours,
            "Gross_Revenue_PKR": np.round(daily_production * fabric_rate_pkr, 2),
            "Diesel_Cost_PKR": np.round(outage_hours * 250 * 0.25 * 280, 2),
        })

    def predict_7day_demand(self, df_history: pd.DataFrame) -> pd.DataFrame:
        if "Power_kWh" not in df_history or df_history.empty:
            raise ValueError("History must contain at least one Power_kWh value.")
        power = pd.to_numeric(df_history["Power_kWh"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(power) == 0:
            raise ValueError("No valid Power_kWh values are available.")
        recent = power[-min(7, len(power)):]
        level = float(np.mean(recent))
        trend = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 2 else 0.0
        dates = pd.date_range(start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=7, freq="D")
        forecasts = [max(0.0, level + trend * (i + 1)) for i in range(7)]

        prod = pd.to_numeric(df_history.get("Production_Meters", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(prod):
            recent_prod = prod.tail(7).to_numpy(dtype=float)
            prod_level = float(np.mean(recent_prod))
            prod_trend = float(np.polyfit(np.arange(len(recent_prod)), recent_prod, 1)[0]) if len(recent_prod) >= 2 else 0.0
            prod_forecast = [max(0.0, prod_level + prod_trend * (i + 1)) for i in range(7)]
        else:
            prod_forecast = [np.nan] * 7
        return pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Forecast_Power_kWh": np.round(forecasts, 2),
            "Forecast_Production_Meters": np.round(prod_forecast, 2),
        })

    def predict_multi_factory_benchmark(self, factory_dict: dict, target_factory: str) -> pd.DataFrame:
        if target_factory not in factory_dict:
            raise KeyError(f"Unknown target factory: {target_factory}")
        target = factory_dict[target_factory]
        if "Power_kWh" not in target:
            raise ValueError("Target factory must contain Power_kWh.")
        target_avg = pd.to_numeric(target["Power_kWh"], errors="coerce").tail(7).mean()
        means = [pd.to_numeric(df["Power_kWh"], errors="coerce").tail(7).mean()
                 for df in factory_dict.values() if "Power_kWh" in df]
        means = [float(v) for v in means if pd.notna(v)]
        peer_avg = float(np.mean(means)) if means else float(target_avg)
        dates = pd.date_range(start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=7, freq="D")
        return pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Target_Factory_Forecast_kWh": [round(float(target_avg + i * 10), 2) for i in range(7)],
            "Industry_Peer_Benchmark_kWh": [round(float(peer_avg + i * 12), 2) for i in range(7)],
        })

    def project_30day_financial_risk(self, daily_net_loss_pkr: float, expected_outage_days_per_month: int) -> dict:
        if expected_outage_days_per_month < 0:
            raise ValueError("expected_outage_days_per_month cannot be negative")
        monthly = float(daily_net_loss_pkr) * int(expected_outage_days_per_month)
        return {
            "expected_outage_days": int(expected_outage_days_per_month),
            "daily_loss_pkr": round(float(daily_net_loss_pkr), 2),
            "monthly_projected_loss_pkr": round(monthly, 2),
        }
