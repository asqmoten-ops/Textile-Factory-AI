"""Streamlit dashboard for textile outage economics, forecasting and peer assessment."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from benchmarking import assess_factory
from charts import (
    create_expense_pie_chart,
    create_peer_efficiency_chart,
    create_sensitivity_line_chart,
    create_waterfall_chart,
)
from data_utils import clean_factory_data, load_reference_files
from engine import TextileFactoryEngine
from forecaster import TextileForecaster
from hybrid_forecaster import HybridForecaster
from optimizer import MachineLoadOptimizer


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "data" / "reference"
MODEL_PATH = ROOT / "models" / "textile_rf_model.pkl"


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Textile Factory Energy & Outage Simulator",
    layout="wide",
)

st.title("Textile Factory Energy & Outage Impact Simulator")

st.caption(
    "Factory health assessment · peer benchmarking · demand forecasting "
    "· outage economics · generator allocation"
)


# ---------------------------------------------------------
# CUSTOM DISPLAY FUNCTION
# Used for long status text instead of st.metric().
# This prevents text from being cut off or overlapping.
# ---------------------------------------------------------

def show_wrapped_status(label: str, value: str) -> None:
    """Display a status label and a wrapping text value."""

    st.markdown(f"**{label}**")

    safe_value = html.escape(str(value))

    st.markdown(
        f"""
        <div style="
            font-size: 1.4rem;
            font-weight: 500;
            line-height: 1.4;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: normal;
            margin-top: 4px;
            margin-bottom: 12px;
        ">
            {safe_value}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# INITIALIZE ENGINES
# ---------------------------------------------------------

factory_engine = TextileFactoryEngine()

forecaster = TextileForecaster(seed=42)

optimizer = MachineLoadOptimizer()

hybrid_forecaster = HybridForecaster(
    model_path=str(MODEL_PATH)
)


# ---------------------------------------------------------
# SIDEBAR: FACTORY DATA UPLOAD
# ---------------------------------------------------------

st.sidebar.header("Factory and Peer Data")

uploaded_files = st.sidebar.file_uploader(
    "Upload factory CSVs (you may include comparable peer factories)",
    type=["csv"],
    accept_multiple_files=True,
    help=(
        "Use the same reporting interval, energy unit and production unit "
        "for the selected factory and its peers."
    ),
)


factory_data_dict: dict[str, pd.DataFrame] = {}

quality_messages: list[str] = []


if uploaded_files:

    for file in uploaded_files:

        factory_name = (
            Path(file.name)
            .stem
            .replace("_", " ")
            .title()
        )

        try:

            raw_df = pd.read_csv(file)

            cleaned, warnings = clean_factory_data(
                raw_df,
                source_name=file.name,
                require_production=True,
                require_outage=False,
            )

            if factory_name in factory_data_dict:

                factory_name = (
                    f"{factory_name} "
                    f"({len(factory_data_dict) + 1})"
                )

            factory_data_dict[factory_name] = cleaned

            quality_messages.extend(
                [
                    f"{file.name}: {msg}"
                    for msg in warnings
                ]
            )

        except Exception as exc:

            st.sidebar.error(
                f"Could not use {file.name}: {exc}"
            )


# ---------------------------------------------------------
# SYNTHETIC DEMO DATA OR UPLOADED FACTORY DATA
# ---------------------------------------------------------

if not factory_data_dict:

    demo_df = forecaster.generate_historical_dataset(days=30)

    factory_data_dict["Demo Factory (SYNTHETIC)"] = demo_df

    selected_factory = "Demo Factory (SYNTHETIC)"

    df_history = demo_df

    st.sidebar.warning(
        "Using synthetic demo data. It is not real-world evidence "
        "and is not included in training."
    )

else:

    selected_factory = st.sidebar.selectbox(
        "Select factory to assess",
        list(factory_data_dict.keys()),
    )

    df_history = factory_data_dict[selected_factory]

    st.sidebar.success(
        f"Loaded {len(factory_data_dict)} factory CSV(s)."
    )


# ---------------------------------------------------------
# LOAD VERIFIED REFERENCE FACTORIES
# ---------------------------------------------------------

reference_factories, reference_warnings = load_reference_files(
    REFERENCE_DIR
)

for warning in reference_warnings:

    quality_messages.append(
        f"Reference library: {warning}"
    )


# ---------------------------------------------------------
# BUILD PEER FACTORY COLLECTION
# ---------------------------------------------------------

peer_factories: dict[str, pd.DataFrame] = {}


# Uploaded factories other than the selected factory

for name, data in factory_data_dict.items():

    if name != selected_factory:

        peer_factories[name] = data


# Reference factories

for name, data in reference_factories.items():

    if name not in peer_factories and name != selected_factory:

        label = name

    else:

        label = f"Reference: {name}"

    if label != selected_factory:

        peer_factories[label] = data


# ---------------------------------------------------------
# DATA QUALITY NOTES
# ---------------------------------------------------------

if quality_messages:

    with st.expander(
        "Data quality notes",
        expanded=False,
    ):

        for msg in quality_messages:

            st.warning(msg)


# ---------------------------------------------------------
# SIDEBAR: OUTAGE SIMULATION INPUTS
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.header("Outage Simulation Inputs")


req_power = st.sidebar.number_input(
    "Factory required power (kW)",
    min_value=0.0,
    value=350.0,
    step=10.0,
)

prod_rate = st.sidebar.number_input(
    "Production rate (metres/hour)",
    min_value=0.0,
    value=1200.0,
    step=50.0,
)

outage_hrs = st.sidebar.slider(
    "Outage duration (hours)",
    min_value=0.0,
    max_value=12.0,
    value=4.0,
    step=0.5,
)

gen_cap = st.sidebar.number_input(
    "Backup generator capacity (kW)",
    min_value=0.0,
    value=150.0,
    step=10.0,
)

st.sidebar.caption(
    "To train/update the Random Forest: add verified, compatible daily "
    "factory CSVs to data/reference/ and run python train_model.py."
)


# ---------------------------------------------------------
# CALCULATE OUTAGE IMPACT
# ---------------------------------------------------------

impact = factory_engine.calculate_outage_impact(
    required_power_kw=req_power,
    prod_rate_meters_hr=prod_rate,
    outage_hours=outage_hrs,
    gen_capacity_kw=gen_cap,
)


# ---------------------------------------------------------
# FACTORY HEALTH ASSESSMENT
# ---------------------------------------------------------

assessment = assess_factory(
    selected_factory,
    df_history,
    peer_factories,
)


# ---------------------------------------------------------
# DASHBOARD NAVIGATION
# ---------------------------------------------------------

tab_health, tab_finance, tab_forecast, tab_optimizer = st.tabs(
    [
        "Factory Health & Benchmarks",
        "Financial Impact",
        "Demand Forecasting",
        "Machine Load Optimizer",
    ]
)


# =========================================================
# TAB 1: FACTORY HEALTH & BENCHMARKS
# =========================================================

with tab_health:

    st.subheader(
        f"Factory Intelligence Report — {selected_factory}"
    )

    st.caption(
        "Stage labels are rule-based and relative to the supplied "
        "peer sample; they are not a regulatory certification."
    )


    # -----------------------------------------------------
    # IMPROVED DISPLAY FOR LONG TEXT VALUES
    # -----------------------------------------------------

    # First row: two wide areas for long text

    stage_col, trend_col = st.columns(2)

    with stage_col:

        show_wrapped_status(
            "Current assessment stage",
            assessment["stage"],
        )

    with trend_col:

        show_wrapped_status(
            "Recent production trend",
            assessment["production_trend"],
        )


    # Second row: normal numeric metrics

    energy_col, peers_col = st.columns(2)

    with energy_col:

        energy_value = assessment["recent_kwh_per_meter"]

        energy_display = (
            f"{energy_value:.4f} kWh/m"
            if energy_value is not None
            else "N/A"
        )

        st.metric(
            "Recent energy intensity",
            energy_display,
        )

    with peers_col:

        st.metric(
            "Comparable peers",
            assessment["peer_count"],
        )


    # Stage explanation

    st.info(
        assessment["stage_reason"]
    )


    # -----------------------------------------------------
    # RECENT FACTORY PERFORMANCE METRICS
    # -----------------------------------------------------

    st.divider()

    k1, k2, k3, k4 = st.columns(4)


    # Average energy

    k1.metric(
        "Recent avg energy",
        (
            f"{assessment['recent_avg_power_kwh']:,.1f} kWh/day"
            if assessment["recent_avg_power_kwh"] is not None
            else "N/A"
        ),
    )


    # Average production

    k2.metric(
        "Recent avg output",
        (
            f"{assessment['recent_avg_production_m']:,.1f} m/day"
            if assessment["recent_avg_production_m"] is not None
            else "N/A"
        ),
    )


    # Average outage

    k3.metric(
        "Recent avg outage",
        (
            f"{assessment['recent_avg_outage_hours']:.2f} hr/day"
            if assessment["recent_avg_outage_hours"] is not None
            else "Unknown"
        ),
    )


    # Production change

    k4.metric(
        "Output change vs prior 7 rows",
        (
            f"{assessment['production_change_pct']:+.1f}%"
            if assessment["production_change_pct"] is not None
            else "N/A"
        ),
    )


    # -----------------------------------------------------
    # PEER BENCHMARKING
    # -----------------------------------------------------

    if assessment["peer_count"] == 0:

        st.warning(
            "No compatible reference peers are loaded. Upload other "
            "comparable factory CSVs or add verified CSV files to "
            "data/reference/. The app will not fabricate a real-world comparison."
        )

    else:

        st.plotly_chart(
            create_peer_efficiency_chart(assessment),
            use_container_width=True,
        )

        st.caption(
            assessment["benchmark_caveat"]
        )

        peer_rows = [
            {
                "Factory/reference": selected_factory,
                "Recent kWh per metre": assessment[
                    "recent_kwh_per_meter"
                ],
            }
        ]

        peer_rows.extend(
            {
                "Factory/reference": row["factory"],
                "Recent kWh per metre": row["kwh_per_meter"],
            }
            for row in assessment["peer_values"]
        )

        st.dataframe(
            pd.DataFrame(peer_rows),
            use_container_width=True,
            hide_index=True,
        )


    # -----------------------------------------------------
    # FACTORY HISTORY
    # -----------------------------------------------------

    with st.expander(
        "Factory history (latest 14 observations)"
    ):

        st.dataframe(
            df_history.tail(14),
            use_container_width=True,
            hide_index=True,
        )


    # -----------------------------------------------------
    # HOW THE STAGE IS CALCULATED
    # -----------------------------------------------------

    with st.expander(
        "How the stage is calculated"
    ):

        st.markdown(
            "- **Energy intensity:** recent total kWh divided by "
            "recent total fabric metres.\n"
            "- **Peer rank:** compares the selected factory's recent "
            "kWh/metre with each supplied peer's recent kWh/metre; "
            "lower is better.\n"
            "- **Production trend:** compares the latest seven rows "
            "with the previous seven rows; ±5% is used as a simple "
            "trend threshold.\n"
            "- **Limits:** factory size, fabric type, machine mix, "
            "shifts, product quality, and reporting interval affect "
            "comparisons. Use peers with compatible processes and units."
        )


# =========================================================
# TAB 2: FINANCIAL IMPACT
# =========================================================

with tab_finance:

    st.subheader(
        f"Outage Financial Impact — {selected_factory}"
    )


    # -----------------------------------------------------
    # FINANCIAL SUMMARY METRICS
    # -----------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)


    m1.metric(
        "Net financial impact",
        f"PKR {impact['net_financial_loss_pkr']:,.0f}",
    )

    m2.metric(
        "Lost fabric output",
        f"{impact['lost_fabric_meters']:,.1f} metres",
    )

    m3.metric(
        "Generator fuel cost",
        f"PKR {impact['generator_fuel_cost_pkr']:,.0f}",
    )

    m4.metric(
        "Power deficit",
        f"{impact['power_deficit_kw']:,.1f} kW",
    )


    st.caption(
        impact["financial_model_note"]
    )


    # -----------------------------------------------------
    # FINANCIAL CHARTS
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            create_waterfall_chart(impact),
            use_container_width=True,
        )

        st.plotly_chart(
            create_expense_pie_chart(impact),
            use_container_width=True,
        )

    with right:

        st.plotly_chart(
            create_sensitivity_line_chart(
                factory_engine,
                req_power,
                prod_rate,
                gen_cap,
            ),
            use_container_width=True,
        )


    # -----------------------------------------------------
    # MONTHLY RISK PROJECTION
    # -----------------------------------------------------

    risk_days = st.number_input(
        "Expected outage days per month",
        min_value=0,
        max_value=31,
        value=8,
        step=1,
    )

    risk = forecaster.project_30day_financial_risk(
        impact["net_financial_loss_pkr"],
        risk_days,
    )

    st.metric(
        "Projected monthly net impact",
        f"PKR {risk['monthly_projected_loss_pkr']:,.0f}",
    )


# =========================================================
# TAB 3: DEMAND FORECASTING
# =========================================================

with tab_forecast:

    st.subheader(
        f"Seven-Day Demand Forecast — {selected_factory}"
    )


    # -----------------------------------------------------
    # GENERATE HYBRID FORECAST
    # -----------------------------------------------------

    df_hybrid = hybrid_forecaster.generate_hybrid_forecast(
        df_history
    )


    # -----------------------------------------------------
    # CHECK WHETHER A TRAINED ML MODEL WAS USED
    # -----------------------------------------------------

    if bool(df_hybrid["ML_Model_Used"].iloc[0]):

        metrics = (
            hybrid_forecaster
            .ml_forecaster
            .metadata
            .get("validation_metrics", {})
        )

        st.success(
            "Trained Random Forest model used for this forecast."
        )

        if metrics.get("mae_kwh") is not None:

            st.caption(
                f"Saved validation MAE: "
                f"{metrics['mae_kwh']:,.2f} kWh; "
                f"validation method: "
                f"{hybrid_forecaster.ml_forecaster.metadata.get('validation_method', 'not recorded')}. "
                "Metrics are from reference data, not a guarantee for this factory."
            )


    elif hybrid_forecaster.ml_forecaster.model_available:

        st.warning(
            "A trained model is present, but this upload does not "
            "provide enough usable history/production features for "
            "the ML forecast. The chart uses the statistical forecast instead."
        )


    else:

        st.warning(
            "No trained Random Forest model is loaded. The ML column "
            "is a statistical fallback, not a trained AI prediction. "
            "Add multiple verified reference CSVs and run "
            "python train_model.py."
        )


    # -----------------------------------------------------
    # FORECAST CHART
    # -----------------------------------------------------

    fig = px.line(
        df_hybrid,
        x="Date",
        y=[
            "Statistical_Forecast_kWh",
            "ML_Forecast_kWh",
            "Hybrid_Forecast_kWh",
        ],
        title="7-Day Forecast: Statistical vs. ML vs. Hybrid",
        markers=True,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Daily electricity demand (kWh)",
        legend_title="Forecast method",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # FORECAST METHOD EXPLANATION
    # -----------------------------------------------------

    method_note = (
        str(df_hybrid["Forecast_Method_Note"].iloc[0])
        if "Forecast_Method_Note" in df_hybrid
        else ""
    )

    st.caption(
        method_note
    )


    # -----------------------------------------------------
    # HISTORICAL DATA AND FORECAST TABLE
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.write(
            "Recent historical telemetry"
        )

        st.dataframe(
            df_history.tail(7),
            use_container_width=True,
            hide_index=True,
        )

    with right:

        st.write(
            "Seven-day forecast breakdown"
        )

        st.dataframe(
            df_hybrid,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# TAB 4: MACHINE LOAD OPTIMIZER
# =========================================================

with tab_optimizer:

    st.subheader(
        "Generator Load Allocation"
    )


    # -----------------------------------------------------
    # OPTIONAL MACHINE REGISTRY UPLOAD
    # -----------------------------------------------------

    uploaded_machines_file = st.file_uploader(
        "Upload machine registry CSV (optional)",
        type=["csv"],
        key="machine_registry",
    )

    custom_machines_list = None


    if uploaded_machines_file is not None:

        try:

            custom_df = pd.read_csv(
                uploaded_machines_file
            )

            custom_machines_list = custom_df.to_dict(
                orient="records"
            )

            st.success(
                "Machine registry loaded."
            )

        except Exception as exc:

            st.error(
                f"Could not load machine registry: {exc}"
            )


    # -----------------------------------------------------
    # GENERATOR LOAD OPTIMIZATION
    # -----------------------------------------------------

    try:

        result = optimizer.optimize_generator_allocation(
            available_gen_kw=gen_cap,
            custom_machines_list=custom_machines_list,
        )


        # Summary metrics

        a, b, c = st.columns(3)


        a.metric(
            "Generator capacity",
            f"{result['available_generator_kw']:,.1f} kW",
        )

        b.metric(
            "Allocated load",
            f"{result['total_allocated_kw']:,.1f} kW",
        )

        c.metric(
            "Estimated retained profit",
            f"PKR {result['retained_hourly_profit_pkr']:,.0f}/hr",
        )


        # Machine allocation table

        st.dataframe(
            result["schedule_table"],
            use_container_width=True,
            hide_index=True,
        )


        # Optimization explanation

        st.caption(
            result["optimization_note"]
        )


    except Exception as exc:

        st.error(
            str(exc)
        )