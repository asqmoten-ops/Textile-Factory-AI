"""Financial and production-impact calculations for textile-factory outages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TextileFactoryEngine:
    grid_tariff: float = 60.0       # PKR / kWh
    diesel_price: float = 280.0     # PKR / litre
    gen_fuel_rate: float = 0.25     # litres / kWh generated
    fabric_price: float = 450.0    # PKR / metre (selling price assumption)
    profit_margin: float = 0.35    # fraction of selling price treated as lost contribution/profit

    def __post_init__(self) -> None:
        if self.grid_tariff < 0 or self.diesel_price < 0 or self.gen_fuel_rate < 0:
            raise ValueError("Tariff, diesel price and fuel rate cannot be negative.")
        if self.fabric_price < 0:
            raise ValueError("Fabric price cannot be negative.")
        if not 0 <= self.profit_margin <= 1:
            raise ValueError("profit_margin must be between 0 and 1.")

    def calculate_outage_impact(
        self,
        required_power_kw: float,
        prod_rate_meters_hr: float,
        outage_hours: float,
        gen_capacity_kw: float,
    ) -> dict[str, Any]:
        values = {
            "required_power_kw": required_power_kw,
            "prod_rate_meters_hr": prod_rate_meters_hr,
            "outage_hours": outage_hours,
            "gen_capacity_kw": gen_capacity_kw,
        }
        for name, value in values.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")

        power_supplied = min(required_power_kw, gen_capacity_kw)
        power_deficit = max(0.0, required_power_kw - power_supplied)
        surplus_capacity = max(0.0, gen_capacity_kw - required_power_kw)
        unpowered_ratio = power_deficit / required_power_kw if required_power_kw > 0 else 0.0

        total_potential_yield = prod_rate_meters_hr * outage_hours
        actual_yield = total_potential_yield * (1.0 - unpowered_ratio)
        lost_fabric_meters = total_potential_yield - actual_yield

        lost_gross_revenue = lost_fabric_meters * self.fabric_price
        lost_profit_margin = lost_gross_revenue * self.profit_margin

        gen_kwh_generated = power_supplied * outage_hours
        diesel_liters_used = gen_kwh_generated * self.gen_fuel_rate
        generator_fuel_cost = diesel_liters_used * self.diesel_price

        # Counterfactual utility bill that would have applied to the full requested
        # load in the no-outage scenario. This is an avoided bill, not generator income.
        baseline_grid_kwh = required_power_kw * outage_hours
        saved_grid_cost = baseline_grid_kwh * self.grid_tariff
        net_financial_loss = lost_profit_margin + generator_fuel_cost - saved_grid_cost

        return {
            "outage_hours": round(float(outage_hours), 2),
            "required_power_kw": round(float(required_power_kw), 2),
            "power_supplied_kw": round(float(power_supplied), 2),
            "surplus_capacity": round(float(surplus_capacity), 2),
            "power_deficit_kw": round(float(power_deficit), 2),
            "lost_fabric_meters": round(float(lost_fabric_meters), 2),
            "actual_fabric_meters": round(float(actual_yield), 2),
            "diesel_liters_used": round(float(diesel_liters_used), 2),
            "generator_fuel_cost_pkr": round(float(generator_fuel_cost), 2),
            "saved_grid_cost_pkr": round(float(saved_grid_cost), 2),
            "lost_profit_margin_pkr": round(float(lost_profit_margin), 2),
            "net_financial_loss_pkr": round(float(net_financial_loss), 2),
            "financial_model_note": (
                "Estimate based on the configured fabric price, margin, fuel rate and tariff. "
                "A negative net value means the avoided baseline grid bill exceeds modeled lost profit plus fuel cost."
            ),
        }
