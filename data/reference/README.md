# Real reference factory data library

Place **verified real-world** factory CSV files in this directory. Every file should describe one factory or one comparable production process, with one row per consistent time interval (the project assumes daily rows unless you adapt the code).

Required columns for model training:

- `Power_kWh`: electricity consumed in that interval, in kWh.
- `Production_Meters`: fabric produced in that same interval, in metres.
- `Outage_Hours`: outage duration during that same interval, in hours.

Recommended column:

- `Date`: ISO date such as `2026-01-31`. It is strongly recommended for proper chronological ordering.

Column aliases such as `Usage_kWh`, `Energy_kWh`, `Output_Meters`, and `Power_Cut_Hours` are normalized by the application, but names and units must be checked by a human. Do not convert unknown fields just to make a CSV pass validation.

Use only factories that are reasonably comparable (textile process, product mix, production unit, measurement interval, and energy scope). Do not add demo/synthetic files here. The training script uses all compatible files in this folder; it saves a model and validation metadata under `models/`.

A useful starting goal is many months of daily data per factory and several independent factories. The small technical minimum enforced by the script is **not** a claim that a model trained on a few dozen rows will be reliable.
