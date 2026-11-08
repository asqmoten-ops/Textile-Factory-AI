import numpy as np
import pandas as pd

from benchmarking import assess_factory
from engine import TextileFactoryEngine
from forecaster import TextileForecaster
from modeling import make_supervised_rows
from optimizer import MachineLoadOptimizer
from hybrid_forecaster import HybridForecaster


def test_engine_outage_impact():
    engine = TextileFactoryEngine()
    res = engine.calculate_outage_impact(350.0, 1200.0, 4.0, 150.0)
    assert res["power_deficit_kw"] == 200.0
    assert res["power_supplied_kw"] == 150.0
    assert res["diesel_liters_used"] == 150.0
    assert res["saved_grid_cost_pkr"] == 84000.0
    assert res["net_financial_loss_pkr"] > 0


def test_forecaster_data_shape():
    forecaster = TextileForecaster(seed=42)
    history = forecaster.generate_historical_dataset(days=30)
    assert len(history) == 30
    assert {"Power_kWh", "Production_Meters", "Outage_Hours"}.issubset(history.columns)
    forecast = forecaster.predict_7day_demand(history)
    assert len(forecast) == 7
    assert "Forecast_Power_kWh" in forecast.columns
    assert (forecast["Forecast_Power_kWh"] >= 0).all()


def test_optimizer_capacity_limit():
    result = MachineLoadOptimizer().optimize_generator_allocation(available_gen_kw=250.0)
    assert result["total_allocated_kw"] <= 250.0
    assert result["unutilized_gen_capacity_kw"] >= 0.0
    assert "schedule_table" in result


def test_rolling_feature_does_not_include_target_row():
    df = pd.DataFrame({
        "Power_kWh": np.arange(1.0, 16.0),
        "Production_Meters": np.full(15, 100.0),
        "Outage_Hours": np.zeros(15),
    })
    supervised = make_supervised_rows(df)
    first = supervised.iloc[0]
    # First target is power=8; its feature mean must use 1..7 only.
    assert first["Power_kWh"] == 8.0
    assert first["Rolling_Mean_7"] == 4.0
    assert first["Power_Lag_1"] == 7.0
    assert first["Power_Lag_7"] == 1.0


def test_assessment_no_peers_does_not_invent_benchmark():
    df = TextileForecaster(seed=11).generate_historical_dataset(days=30)
    result = assess_factory("Target", df, {})
    assert result["peer_count"] == 0
    assert result["peer_percentile_rank"] is None
    assert "peers needed" in result["stage"].lower()


def test_hybrid_forecast_has_seven_days_without_model():
    df = TextileForecaster(seed=5).generate_historical_dataset(days=30)
    result = HybridForecaster(model_path="models/not_a_real_model.pkl").generate_hybrid_forecast(df)
    assert len(result) == 7
    assert not bool(result["ML_Model_Loaded"].iloc[0])
    assert result["Hybrid_Forecast_kWh"].notna().all()


def test_training_loader_reads_multiple_sources_from_reference_folder(tmp_path):
    from train_model import load_training_sources

    for index in range(3):
        frame = TextileForecaster(seed=20 + index).generate_historical_dataset(days=60)
        frame["Power_kWh"] = frame["Power_kWh"] * (1 + index * 0.03)
        frame.to_csv(tmp_path / f"factory_{index}.csv", index=False)
    sources, warnings = load_training_sources(tmp_path)
    assert len(sources) == 3
    assert all(len(frame) > 30 for frame in sources.values())


def test_missing_outage_values_are_not_silently_filled():
    from data_utils import clean_factory_data

    frame = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=3),
        "Power_kWh": [100, 110, 120],
        "Production_Meters": [400, 420, 430],
        "Outage_Hours": [0.0, np.nan, 1.0],
    })
    cleaned, warnings = clean_factory_data(frame, "quality.csv")
    assert pd.isna(cleaned.loc[1, "Outage_Hours"])
    assert any("values were not invented" in warning for warning in warnings)
