# Phoenix Crime Hotspot Forecasting

A reproducible academic project exploring monthly crime-count forecasting across Phoenix Police Department grid areas. It compares a Random Forest regression model with per-grid SARIMAX time-series forecasts and evaluates the Random Forest's ability to identify the five highest-count grids each month.

> **Ethics and intended use:** This is an academic forecasting study, not a police deployment recommendation. Historical incident counts reflect reporting, enforcement, and other structural factors. Models may reproduce or amplify those patterns. Outputs should not be used to make decisions about individuals or justify enforcement without rigorous independent review, community oversight, and impact assessment.

## Project status

This repository contains a reproducible code implementation and a preliminary local reproduction. Previously saved notebook outputs and narrative metrics are not treated as final until regenerated with this pipeline and reconciled. See [the reproduction and audit notes](reports/REPRODUCTION_AUDIT.md) for the current metric comparison and known protocol differences. Results are preliminary; no claim is made that the new pipeline reproduces the submitted PDF exactly.

## Study design

- **Geography:** Phoenix police-grid areas, using the `GRID` field.
- **Study window:** January 2020 through December 2024.
- **Target:** monthly incident count per grid.
- **Random Forest features:** grid, season, year, month, and lagged monthly counts/rolling averages.
- **Random Forest model selection:** train on 2020–2022, select parameters on 2023 validation data, refit on 2020–2023, and evaluate once on the 2024 holdout.
- **SARIMAX:** fixed seasonal specification fitted separately for recurring grids selected using training-period hotspot frequency; forecasts 12 months of 2024 from data through 2023.

The Random Forest uses a rolling one-month-ahead interpretation: lag features for a month use observed counts from preceding months. SARIMAX forecasts the full 2024 horizon from data through 2023. Since the protocols differ, their scores are not a perfectly controlled head-to-head comparison.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Keep the source incident CSV local; do not commit it.
4. Run: `python run_analysis.py --input /path/to/crime-data-2026.csv`.
5. Review generated files in `results/generated/` before sharing them.

The input file must include `OCCURRED ON` and `GRID` columns. The pipeline uses records from 2020–2024 and creates a complete month-by-grid panel for grids present in the input.

## Outputs

- `phoenix_monthly_hotspots.csv`: monthly grid counts and lagged features.
- `RandomForestPredictions.csv`: 2024 grid-level predictions and hotspot flags.
- `RandomForestHotspotAccuracy.csv`: monthly top-five overlap and accuracy.
- `RandomForestValidationTuning.csv`: candidate model scores on validation data.
- `SARIMAX_Grid_Metrics.csv` and `SARIMAX_Grid_Predictions.csv`: per-grid forecast evaluation and predictions.

Generated outputs are ignored by Git by default. Check data permissions and disclosure risks before intentionally publishing any outputs.

## Evaluation definitions

- **Top-5 accuracy:** for each month, the number of grids shared by the actual and predicted top-five sets divided by five; the reported mean is the average across months.
- **MAE / RMSE / R²:** regression metrics on held-out observations; a negative R² indicates performance worse than predicting the test-set mean under that metric's definition.
- **Validation versus test:** hyperparameters are selected using 2023 validation results. The 2024 holdout is reserved for the final evaluation.

## Limitations

- Crime counts are not equivalent to crime incidence; they are affected by reporting, enforcement, and data collection practices.
- Geographic aggregation can conceal within-grid variation and does not support conclusions about individuals.
- Historical patterns can encode structural inequities and can be reinforced by predictive systems.
- Forecast accuracy alone does not establish that a model should guide resource allocation or policing.
- Reproduce the pipeline and document any changes before reporting results.

## Repository layout

```text
src/forecasting.py       Data preparation, feature engineering, and model evaluation
run_analysis.py          Command-line pipeline runner
tests/                   Automated tests
data/README.md           Local data and privacy guidance
results/README.md        Generated output guidance
requirements.txt         Python dependencies
```
