"""Greedy, transparent generator-load allocation for whole machines."""
from __future__ import annotations

from typing import Any
import math
import pandas as pd


DEFAULT_MACHINES = [
    {"Machine": "Ring Spinning Line A", "Power_kW": 75.0, "Hourly_Profit_PKR": 4200.0},
    {"Machine": "Compressor", "Power_kW": 30.0, "Hourly_Profit_PKR": 900.0},
    {"Machine": "Dyeing Pump Set", "Power_kW": 45.0, "Hourly_Profit_PKR": 1800.0},
    {"Machine": "Finishing Line", "Power_kW": 60.0, "Hourly_Profit_PKR": 3100.0},
    {"Machine": "Lighting and Controls", "Power_kW": 12.0, "Hourly_Profit_PKR": 250.0},
    {"Machine": "Winding Unit", "Power_kW": 25.0, "Hourly_Profit_PKR": 1200.0},
]


class MachineLoadOptimizer:
    """Prioritize complete machines by estimated hourly profit per kW.

    This is a transparent greedy heuristic, not a mathematically guaranteed global
    optimum. It assumes a machine either runs at its listed load or is switched off.
    """

    def optimize_generator_allocation(
        self,
        available_gen_kw: float,
        custom_machines_list: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(float(available_gen_kw)) or available_gen_kw < 0:
            raise ValueError("available_gen_kw must be a finite, non-negative number.")
        rows = custom_machines_list if custom_machines_list else DEFAULT_MACHINES
        if not rows:
            rows = []

        normalized = []
        for i, machine in enumerate(rows):
            name = str(machine.get("Machine", machine.get("Machine_Name", machine.get("name", f"Machine {i + 1}"))))
            power = machine.get("Power_kW", machine.get("Power_kW_Required", machine.get("power_kw")))
            profit = machine.get("Hourly_Profit_PKR", machine.get("Profit_per_hour_PKR", machine.get("profit_per_hour_pkr")))
            if power is None or profit is None:
                raise ValueError(
                    "Machine registry needs Power_kW and Hourly_Profit_PKR columns "
                    "(aliases Power_kW_Required and Profit_per_hour_PKR are also supported)."
                )
            power, profit = float(power), float(profit)
            if not math.isfinite(power) or not math.isfinite(profit) or power < 0:
                raise ValueError(f"Invalid power/profit value for {name}.")
            if profit < 0:
                # Negative profit machines can still be operationally essential, but the
                # current objective is profit-retaining allocation, so they are not run.
                efficiency = float("-inf")
            else:
                efficiency = profit / power if power > 0 else float("inf")
            normalized.append({
                "Machine": name,
                "Required_Power_kW": power,
                "Estimated_Profit_PKR_hr": profit,
                "Profit_per_kW": efficiency,
                "_order": i,
            })

        # Higher profit/kW first. Stable ordering makes the schedule reproducible.
        normalized.sort(key=lambda x: (-x["Profit_per_kW"], x["_order"]))
        remaining = float(available_gen_kw)
        schedule = []
        for item in normalized:
            pwr = item["Required_Power_kW"]
            can_run = pwr <= remaining + 1e-9 and item["Estimated_Profit_PKR_hr"] >= 0
            if can_run:
                remaining -= pwr
            schedule.append({
                "Machine": item["Machine"],
                "Required_Power_kW": round(pwr, 2),
                "Estimated_Profit_PKR_hr": round(item["Estimated_Profit_PKR_hr"], 2),
                "Profit_per_kW": round(item["Profit_per_kW"], 2) if math.isfinite(item["Profit_per_kW"]) else None,
                "Status": "RUN" if can_run else "OFF — capacity/profit priority",
            })
        table = pd.DataFrame(schedule)
        running = table[table["Status"] == "RUN"] if not table.empty else table
        allocated = float(running["Required_Power_kW"].sum()) if not running.empty else 0.0
        retained = float(running["Estimated_Profit_PKR_hr"].sum()) if not running.empty else 0.0
        return {
            "available_generator_kw": round(float(available_gen_kw), 2),
            "total_allocated_kw": round(allocated, 2),
            "unutilized_gen_capacity_kw": round(max(0.0, float(available_gen_kw) - allocated), 2),
            "retained_hourly_profit_pkr": round(retained, 2),
            "schedule_table": table,
            "optimization_note": "Greedy ranking by estimated hourly profit per kW; confirm safety, machine dependencies and minimum stable loads with plant engineers.",
        }
