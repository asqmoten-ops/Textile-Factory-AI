# Textile Factory Energy & Operations Intelligence System.

The **Textile Factory Energy & Operations Intelligence System** is a Python-based decision-support application designed to help textile manufacturers analyze energy consumption, estimate financial losses caused by power outages, forecast electricity requirements, assess operational efficiency, and optimize generator power allocation.

Built using Streamlit, the system integrates a **multi-factory reference library**, a forecast-safe Random Forest training pipeline, statistical and machine learning forecasting, factory health assessments, peer energy-intensity comparisons, and an improved machine-load optimizer. These features support data-driven decision-making to help factories reduce operational costs, manage electricity outages, and improve overall efficiency.

> **Important:** The Random Forest model can be trained using compatible factory datasets. A trustworthy model must be trained on verified compatible real-world CSVs. The app explicitly identifies its built-in data as synthetic demo data when no factory CSV is uploaded.

## Features

- Outage financial-impact estimate: power deficit, reduced production, diesel expense, avoided baseline grid bill, and net modeled financial impact.
- Seven-day statistical and Random Forest forecasts; the dashboard clearly identifies when the model is not trained and falls back to statistical forecasting.
- Training from **multiple compatible reference factory CSVs** in `data/reference/`.
- Holdout evaluation. With three or more reference files, one source is held out to estimate whether the model generalizes to an unseen factory; with one or two sources, the latest chronological tail is held out.
- Health assessment based on recent kWh per metre, production trend, outage history, and relative peer position when compatible peers exist.
- Generator-load allocation from an uploaded machine registry or an illustrative built-in registry. Allocation is a transparent greedy heuristic, not a guaranteed global optimum.
- CSV validation, common column aliases, warnings about missing data, and automated tests for feature leakage and capacity constraints.

## 1. Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Run the dashboard

```bash
streamlit run app.py
```

Upload one CSV for the factory you want to assess. You may also upload multiple comparable factory CSVs; choose one as the target and the others become peer examples for the session. For valid peer comparisons, their time intervals and units must match.

If no CSV is uploaded, the dashboard uses synthetic demo telemetry. This is only to let you explore the interface and is **not** evidence of real-world model performance.

## 3. Standard factory CSV schema

A daily factory record should contain:

| Column | Required for model training | Meaning |
|---|---:|---|
| `Date` | Strongly recommended | Date of the daily record; supports chronological ordering |
| `Power_kWh` | Yes | Electricity consumed during the record interval, in kWh |
| `Production_Meters` | Yes | Fabric produced during the same interval, in metres |
| `Outage_Hours` | Yes | Power outage duration during the same interval, in hours |
| `Diesel_Cost_PKR` | No | Recorded generator fuel expense, if available |
| `Gross_Revenue_PKR` | No | Recorded gross revenue, if available |

For the dashboard's energy-intensity peer comparison, at minimum `Power_kWh` and `Production_Meters` must both be present and represent the same period. Without `Outage_Hours`, the app can still show some factory KPIs, but outage-based training is unavailable and forecast outage assumptions are incomplete. Missing fields are not silently invented.

Common aliases including `Usage_kWh`, `Energy_kWh`, `Output_Meters`, `Meters_Produced`, and `Power_Cut_Hours` are normalized, but you must check that their definitions and units really match the canonical fields.

## 4. Add real reference factories and train the model

1. Put one verified factory/process CSV per file in `data/reference/`.
2. Ensure each row uses the same daily interval and units, and that `Power_kWh`, `Production_Meters`, and `Outage_Hours` are measured values for the same interval.
3. Ideally collect many months (or years) of data from several distinct, comparable factories. Small samples are supported technically but will not make a robust model by themselves.
4. Run:

```bash
python train_model.py
```

The script saves:

- `models/textile_rf_model.pkl` — the trained model.
- `models/model_metadata.json` — source counts, rows, validation methodology, MAE/RMSE/R², warnings, and notes.

The model predicts the next daily `Power_kWh` using recent energy lags, a prior-seven-observation average, current/assumed production, and outage hours. The rolling-average feature is built using only prior observations, avoiding the target leakage present in the original code.

### Forecast assumptions and limitations

The future seven-day values of production and outages are not known when the forecast is made. This implementation assumes recent average production and outage hours for the future, then forecasts demand recursively. Treat the results as scenarios/estimates, not as guaranteed predictions. Update the model only with reviewed data and compare reported validation metrics over time.

With three or more reference sources, the script evaluates on a held-out source before fitting the saved deployment model on all approved reference sources. With fewer than three sources, it evaluates a chronological tail split. These metrics are useful checks but do not prove performance for every new factory.

## 5. Current factory stage

The stage is rule-based and intentionally explainable. It uses the current factory's recent weighted `kWh per metre`, its production trend versus the previous seven rows, and peer energy intensity when peer data is supplied. 

Peer comparisons can be misleading when factories differ in fabric type, process (for example spinning vs weaving vs dyeing), product quality, shifts, machine mix, production units, or reporting intervals. Use peer groups with compatible processes and keep those limitations in mind.

## 6. Machine registry CSV

Optional columns:

- `Machine` (or `Machine_Name`)
- `Power_kW` (or `Power_kW_Required`)
- `Hourly_Profit_PKR` (or `Profit_per_hour_PKR`)

The built-in machine list contains illustrative assumptions. Replace it with verified machine power and production/profit data before operational use. Safety requirements, minimum stable loads, machine dependencies, startup surges and critical loads are not modeled by the greedy heuristic.

## 7. Run tests

```bash
pytest -q
```

## 8. Public data sources and licences

See [`DATASET_SOURCES.md`](DATASET_SOURCES.md). Public datasets must be reviewed and transformed carefully; a textile machine-level dataset or a steel plant's energy-only records cannot automatically become a daily textile kWh/metre/outage reference. 

## Project structure

```text
app.py                     Streamlit dashboard
engine.py                  Outage financial calculations
forecaster.py              Statistical forecast and synthetic demo data
modeling.py                Leakage-safe feature engineering
train_model.py             Multi-source Random Forest training/evaluation
ml_forecaster.py           Recursive ML forecast
hybrid_forecaster.py       Statistical + ML forecast integration
benchmarking.py            Factory KPIs and explainable stage assessment
data_utils.py              CSV normalization, validation and reference loading
optimizer.py               Generator machine-load allocation
charts.py                  Plotly visualizations
 data/reference/            Verified compatible factory CSV library
 data/public_industrial/    Reserved for raw public research datasets
 models/                    Model artifacts created after training
tests/test_simulator.py     Automated tests
```
