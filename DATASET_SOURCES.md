# Public dataset research notes

The following public datasets are useful starting points, but they do **not** all represent compatible textile factories. Always read the source description and licence before using or redistributing any data.

## 1. Textile rotor-spinning machine dataset

- Source: Mendeley Data, “Data for: Forecasting the power consumption of a rotor spinning machine from imbalanced data with an adaptive SE-CNN”.
- Link: https://data.mendeley.com/datasets/xmfp5bt3wb/1
- Source description: measurements at eight-hour intervals from 36 rotor-spinning machines; 897 records with 16 input parameters and measured specific power consumption.
- Licence shown by the source: CC BY-NC 3.0 (attribution and non-commercial restriction).
- Limitation: machine-level specific-power records are not the same as factory-level daily `Power_kWh`, daily `Production_Meters`, and `Outage_Hours`. Do not use it to train the project's five-feature model until a defensible mapping and matching target definitions are established. Check the licence before commercial use.

## 2. UCI Steel Industry Energy Consumption

- Source: UCI Machine Learning Repository, “Steel Industry Energy Consumption”.
- Link: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption
- Source description: 35,040 observations from a South Korean steel industry setting, including electricity usage and electrical/time features; CC BY 4.0 according to the repository.
- Limitation: this is not textile production data and it does not provide fabric metres or textile outage hours. It can support a separate generic industrial energy-profile study, but its electricity numbers should not be treated as a comparable textile kWh/metre benchmark.

## How to use public datasets safely in this project

1. Keep each raw source and its original column names in a separate folder.
2. Read the dataset documentation and licence; record source URL and any required attribution.
3. Build and review a conversion script that maps raw readings to the standard interval and units. Do not infer missing fabric production/outages.
4. Validate that `Power_kWh`, `Production_Meters`, and `Outage_Hours` refer to the same interval before putting the transformed file in `data/reference/`.
5. Keep a list of dataset sources and conversion decisions so results can be reproduced.

The project intentionally ships without claimed “real-world training examples.” The model is only trained after you place verified, compatible data in the reference library and run `python train_model.py`.
