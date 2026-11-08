"""Train an outage-aware Random Forest from multiple compatible factory CSVs.

Training inputs: data/reference/*.csv
Expected columns per file: Power_kWh, Production_Meters, Outage_Hours; Date is recommended.
Each file should represent one textile factory or one comparable production process at a
consistent daily interval. Do not put synthetic demo files in data/reference/.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import clean_factory_data
from modeling import FEATURE_COLUMNS, make_supervised_rows

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "data" / "reference"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "textile_rf_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def load_training_sources(reference_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if reference_dir is None:
        reference_dir = REFERENCE_DIR
    sources: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    if not reference_dir.exists():
        return sources, warnings
    for path in sorted(reference_dir.glob("*.csv")):
        try:
            raw = pd.read_csv(path)
            clean, row_warnings = clean_factory_data(
                raw,
                source_name=path.name,
                require_production=True,
                require_outage=True,
            )
            supervised = make_supervised_rows(clean)
            if len(supervised) < 8:
                warnings.append(f"Skipped {path.name}: only {len(supervised)} usable rows remain after lag features; provide more history.")
                continue
            sources[path.stem] = supervised
            warnings.extend([f"{path.name}: {warning}" for warning in row_warnings])
        except Exception as exc:
            warnings.append(f"Skipped {path.name}: {exc}")
    return sources, warnings


def _mae_report(model, test_df: pd.DataFrame) -> dict:
    if test_df.empty:
        return {"test_rows": 0, "mae_kwh": None, "rmse_kwh": None, "r2": None}
    pred = model.predict(test_df[FEATURE_COLUMNS])
    actual = test_df["Power_kWh"].to_numpy(dtype=float)
    return {
        "test_rows": int(len(test_df)),
        "mae_kwh": round(float(mean_absolute_error(actual, pred)), 3),
        "rmse_kwh": round(float(np.sqrt(mean_squared_error(actual, pred))), 3),
        "r2": round(float(r2_score(actual, pred)), 4) if len(actual) >= 2 else None,
    }


def train_and_save_model() -> dict:
    sources, warnings = load_training_sources()
    if not sources:
        raise SystemExit(
            "No compatible training CSVs found. Add real, compatible daily factory CSVs to "
            f"{REFERENCE_DIR} with Power_kWh, Production_Meters and Outage_Hours columns."
        )

    all_rows = pd.concat(
        [df.assign(_source=name) for name, df in sources.items()],
        ignore_index=True,
    )
    if len(all_rows) < 30:
        raise SystemExit(
            f"Only {len(all_rows)} usable training rows were found. Add more real data; "
            "30 rows is merely a minimal technical floor, not a recommendation for a reliable model."
        )

    # Validation policy:
    # - With >=3 factory sources, hold one entire factory out to check transfer to an
    #   unseen source. This is stronger than randomly mixing rows from every factory.
    # - With 1-2 sources, hold out the latest 20% of rows within each source.
    if len(sources) >= 3:
        validation_source = sorted(sources)[-1]
        train_parts = [df for name, df in sources.items() if name != validation_source]
        test_df = sources[validation_source]
        validation_method = "held-out factory/source"
        validation_sources = [validation_source]
        validation_train_df = pd.concat(train_parts, ignore_index=True)
    else:
        validation_method = "chronological tail split within each source"
        train_parts, test_parts = [], []
        for name, df in sources.items():
            split = max(1, int(len(df) * 0.8))
            if split >= len(df):
                train_parts.append(df)
            else:
                train_parts.append(df.iloc[:split])
                test_parts.append(df.iloc[split:])
        validation_train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=all_rows.columns)
        validation_sources = sorted(sources)

    if len(validation_train_df) < 20:
        raise SystemExit(
            f"Only {len(validation_train_df)} rows are available for validation training. "
            "Add more history or more compatible real factory datasets."
        )

    validation_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    validation_model.fit(validation_train_df[FEATURE_COLUMNS], validation_train_df["Power_kWh"])
    validation_metrics = _mae_report(validation_model, test_df)

    # After reporting honest holdout metrics, refit the deployment model on all approved
    # reference sources so the deployed artifact uses all supplied data.
    final_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(all_rows[FEATURE_COLUMNS], all_rows["Power_kWh"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestRegressor",
        "target": "Power_kWh (next daily observation)",
        "feature_columns": FEATURE_COLUMNS,
        "reference_directory": str(REFERENCE_DIR.relative_to(ROOT)),
        "source_count": len(sources),
        "source_rows_after_feature_engineering": {name: int(len(df)) for name, df in sources.items()},
        "training_rows": int(len(all_rows)),
        "validation_method": validation_method,
        "validation_sources": validation_sources,
        "validation_metrics": validation_metrics,
        "warnings": warnings,
        "notes": [
            "Rolling_Mean_7 uses only prior observations; current target is not included.",
            "Metrics are from a holdout evaluated before final refit. The final deployment model is refit on all approved reference rows.",
            "Performance on new factories depends on whether the reference sources represent the target factory's process and operating conditions.",
            "Future production and outage values are assumed from recent history by MLForecaster; they are not known future measurements.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Reference sources used: {len(sources)}")
    print(f"Rows after feature engineering: {len(all_rows)}")
    print(f"Validation method: {validation_method}")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return metadata


if __name__ == "__main__":
    train_and_save_model()
