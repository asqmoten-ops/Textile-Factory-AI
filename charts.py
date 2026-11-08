"""Plotly charts used by the Streamlit dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_expense_pie_chart(impact_data: dict) -> go.Figure:
    # A pie should contain only cost components. Avoid treating an avoided grid bill
    # as a third expense because that distorts expense percentages.
    labels = ["Generator Fuel Cost", "Lost Profit Margin"]
    values = [
        max(0.0, float(impact_data.get("generator_fuel_cost_pkr", 0))),
        max(0.0, float(impact_data.get("lost_profit_margin_pkr", 0))),
    ]
    if sum(values) <= 0:
        fig = go.Figure()
        fig.add_annotation(text="No positive cost components in this scenario", showarrow=False)
    else:
        fig = px.pie(names=labels, values=values, title="Modeled Cost Components (PKR)", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True)
    return fig


def create_waterfall_chart(impact_data: dict) -> go.Figure:
    lost_profit = float(impact_data.get("lost_profit_margin_pkr", 0))
    fuel = float(impact_data.get("generator_fuel_cost_pkr", 0))
    saved_grid = float(impact_data.get("saved_grid_cost_pkr", 0))
    net = float(impact_data.get("net_financial_loss_pkr", lost_profit + fuel - saved_grid))
    fig = go.Figure(go.Waterfall(
        name="Financial impact",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Lost Profit Margin", "Generator Fuel Expense", "Avoided Baseline Grid Bill", "Net Financial Impact"],
        y=[lost_profit, fuel, -saved_grid, net],
        text=[f"PKR {lost_profit:,.0f}", f"PKR {fuel:,.0f}", f"−PKR {saved_grid:,.0f}", f"PKR {net:,.0f}"],
        textposition="outside",
        connector={"line": {"color": "rgba(100,100,100,0.5)"}},
    ))
    fig.update_layout(title="Financial Impact Bridge (PKR)", xaxis_title="Component", yaxis_title="PKR")
    return fig


def create_sensitivity_line_chart(factory_engine, req_power: float, prod_rate: float, gen_cap: float) -> go.Figure:
    hours_range = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    losses = [factory_engine.calculate_outage_impact(req_power, prod_rate, h, gen_cap)["net_financial_loss_pkr"] for h in hours_range]
    df_sens = pd.DataFrame({"Outage_Hours": hours_range, "Net_Financial_Loss_PKR": losses})
    fig = px.line(df_sens, x="Outage_Hours", y="Net_Financial_Loss_PKR", title="Sensitivity: Outage Duration vs. Net Financial Impact", markers=True)
    fig.update_layout(xaxis_title="Outage Duration (hours)", yaxis_title="Net financial impact (PKR)")
    return fig


def create_forecast_area_chart(df_forecast: pd.DataFrame) -> go.Figure:
    fig = px.area(df_forecast, x="Date", y="Forecast_Power_kWh", title="7-Day Forecasted Energy Demand (kWh)", markers=True)
    fig.update_layout(xaxis_title="Future date", yaxis_title="Energy demand (kWh)")
    return fig


def create_multi_factory_comparison_chart(df_forecast: pd.DataFrame, target_factory_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Target_Factory_Forecast_kWh"], mode="lines+markers", name=f"{target_factory_name} (target)"))
    fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Industry_Peer_Benchmark_kWh"], mode="lines+markers", name="Peer benchmark (average)", line={"dash": "dash"}))
    fig.update_layout(title=f"7-Day Energy Forecast: {target_factory_name} vs Peers", xaxis_title="Date", yaxis_title="Power demand (kWh)")
    return fig


def create_peer_efficiency_chart(assessment: dict) -> go.Figure:
    rows = assessment.get("peer_values", [])
    if assessment.get("recent_kwh_per_meter") is not None:
        rows = list(rows) + [{"factory": f"{assessment.get('factory_name', 'Selected factory')} (you)", "kwh_per_meter": assessment["recent_kwh_per_meter"]}]
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No compatible peer benchmark is loaded", showarrow=False)
        fig.update_layout(title="Energy Intensity Peer Comparison")
        return fig
    df = pd.DataFrame(rows).sort_values("kwh_per_meter", ascending=True)
    fig = px.bar(df, x="factory", y="kwh_per_meter", title="Recent Energy Intensity: kWh per Metre (lower is better)")
    fig.update_layout(xaxis_title="Factory/reference dataset", yaxis_title="kWh per metre", xaxis_tickangle=-25)
    return fig
