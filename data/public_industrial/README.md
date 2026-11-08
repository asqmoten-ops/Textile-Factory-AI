# Public industrial datasets (raw research inputs)

This folder is reserved for publicly sourced industrial datasets after you download them. Raw data here is **not automatically used** for textile-factory model training or the peer benchmark.

A dataset should only enter `data/reference/` after you confirm that its energy, production, outage, and time-interval definitions align with the project's canonical schema. For example, an energy-only steel-plant dataset cannot provide fabric metres or textile outage hours simply by renaming columns.

See `DATASET_SOURCES.md` in the project root for the public sources reviewed when this project scaffold was prepared.
