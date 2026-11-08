"""Transparent statistical + Random Forest forecast combination."""
from __future__ import annotations

import pandas as pd
from forecaster import TextileForecaster
from ml_forecaster import MLForecaster


class HybridForecaster:
    def __init__(self, model_path: str = "models/textile_rf_model.pkl"):
        self.stat_forecaster = TextileForecaster(seed=42)
        self.ml_forecaster = MLForecaster(model_path=model_path)

    def generate_hybrid_forecast(self, df_history: pd.DataFrame) -> pd.DataFrame:
        df_stat = self.stat_forecaster.predict_7day_demand(df_history)
        df_ml = self.ml_forecaster.predict_7day_ml(df_history)
        model_loaded = self.ml_forecaster.model_available
        model_used = bool(df_ml["ML_Model_Used"].iloc[0])

        if model_used:
            # Keep this weight explicit until reliable holdout evaluation demonstrates
            # that another weight is superior for this particular deployment.
            hybrid = 0.5 * df_stat["Forecast_Power_kWh"].to_numpy() + 0.5 * df_ml["ML_Forecast_Power_kWh"].to_numpy()
            method_note = "50/50 statistical + Random Forest; weight is configurable and not claimed to be optimal"
        else:
            # Do not blend two statistical fallbacks and present it as ML.
            hybrid = df_stat["Forecast_Power_kWh"].to_numpy()
            if model_loaded:
                method_note = "Statistical forecast only; the saved Random Forest was not used because sufficient history/production features were unavailable"
            else:
                method_note = "Statistical forecast only; no trained Random Forest model is loaded"

        return pd.DataFrame({
            "Date": df_stat["Date"],
            "Statistical_Forecast_kWh": df_stat["Forecast_Power_kWh"],
            "ML_Forecast_kWh": df_ml["ML_Forecast_Power_kWh"],
            "Hybrid_Forecast_kWh": [round(max(0.0, float(value)), 2) for value in hybrid],
            "ML_Model_Loaded": [model_loaded] * len(df_stat),
            "ML_Model_Used": [model_used] * len(df_stat),
            "Forecast_Method_Note": [method_note] * len(df_stat),
        })
