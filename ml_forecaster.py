"""Load the trained model and recursively forecast the next seven daily observations."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modeling import FEATURE_COLUMNS


class MLForecaster:
    def __init__(self, model_path: str = "models/textile_rf_model.pkl"):
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.with_name("model_metadata.json")
        self.model = self._load_model()
        self.metadata = self._load_metadata()

    def _load_model(self):
        if self.model_path.exists():
            try:
                return joblib.load(self.model_path)
            except Exception:
                return None
        return None

    def _load_metadata(self) -> dict:
        if self.metadata_path.exists():
            try:
                return json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    @property
    def model_available(self) -> bool:
        return self.model is not None

    def predict_7day_ml(self, df_history: pd.DataFrame) -> pd.DataFrame:
        if "Power_kWh" not in df_history or df_history.empty:
            raise ValueError("History must contain Power_kWh observations.")
        history = df_history.copy()
        if "Date" in history.columns:
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history.sort_values("Date")
        powers = pd.to_numeric(history["Power_kWh"], errors="coerce").dropna().astype(float).tolist()
        if not powers:
            raise ValueError("No valid Power_kWh values found.")
        productions = pd.to_numeric(history.get("Production_Meters", pd.Series(dtype=float)), errors="coerce").dropna()
        outages = pd.to_numeric(history.get("Outage_Hours", pd.Series(dtype=float)), errors="coerce").dropna()
        current_prod = float(productions.tail(7).mean()) if len(productions) else 0.0
        current_outage = float(outages.tail(7).mean()) if len(outages) else 0.0
        # This is an assumption, not a measured future schedule.
        future_dates = pd.date_range(start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=7, freq="D")
        forecast_power: list[float] = []
        model_used = bool(self.model_available and len(powers) >= 7 and len(productions) > 0)
        if model_used:
            method = "random_forest"
        elif not self.model_available:
            method = "statistical_fallback_no_model"
        else:
            method = "statistical_fallback_insufficient_history_or_production"

        if model_used:
            recursive_history = list(powers)
            for day_idx in range(7):
                feature_row = {
                    "Outage_Hours": current_outage,
                    "Production_Meters": max(0.0, current_prod),
                    "Power_Lag_1": recursive_history[-1],
                    "Power_Lag_7": recursive_history[-7],
                    "Rolling_Mean_7": float(np.mean(recursive_history[-7:])),
                }
                x_input = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
                predicted = max(0.0, float(self.model.predict(x_input)[0]))
                forecast_power.append(round(predicted, 2))
                recursive_history.append(predicted)
        else:
            recent = np.asarray(powers[-min(7, len(powers)):], dtype=float)
            level = float(recent.mean())
            trend = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 2 else 0.0
            forecast_power = [round(max(0.0, level + trend * (i + 1)), 2) for i in range(7)]

        return pd.DataFrame({
            "Date": future_dates.strftime("%Y-%m-%d"),
            "ML_Forecast_Power_kWh": forecast_power,
            "ML_Model_Loaded": [self.model_available] * 7,
            "ML_Model_Used": [model_used] * 7,
            "ML_Forecast_Method": [method] * 7,
        })
