# Reproduction and audit notes

**Status:** preliminary local reproduction, 2026-10-24  
**Data:** local `crime-data-2026.csv`; raw source data is not committed.

## What was run

The pipeline in `run_analysis.py` was executed against the supplied CSV. It aggregated incidents from January 2020 through December 2024 into a complete monthly panel for 1,745 grids (104,700 grid-month rows), engineered lagged features, selected Random Forest hyperparameters using 2023 validation data, refit through 2023, and evaluated on 2024. It also fit the documented SARIMAX specification for five recurring training-period hotspot grids.

Automated tests: **2 passed** using `python -m pytest -q`. The end-to-end pipeline completed successfully.

## Preliminary results from the current repository code

### Random Forest

| Metric | Current pipeline |
|---|---:|
| MAE | 1.6183 |
| RMSE | 2.4245 |
| R² | 0.7031 |
| Mean correct actual Top-5 grids per month | 3.00 / 5 |
| Mean Top-5 overlap accuracy | 60.0% |

The current pipeline selected 100 trees, maximum depth 20, minimum leaf size 1, and `max_features='sqrt'` using 2023 validation scores. The 2024 holdout was not used for hyperparameter selection.

### SARIMAX

The pipeline evaluated five grids and generated 60 grid-month forecasts. Across the five grid-level metrics, the mean MAE was 10.8489, mean RMSE was 12.6726, and mean R² was -0.5457. The negative mean R² indicates that performance varied and was worse than the test-mean baseline under that metric for the evaluated grids. Do not characterize this result as strong performance.

## Reconciliation with supplied artifacts

The previously supplied `RandomForestPredictions.csv` recomputes to MAE 1.6615, RMSE 2.4832, and R² 0.6886. The supplied `RandomForestHotspotAccuracy.csv` averages 3.00 correct Top-5 grids per month, or 60.0%.

The original notebook's narrative reports R² = 0.693, 3.09 correct grids, 61.7% Top-5 accuracy, and cross-validated R² = 0.675. Those figures are **not supported by the supplied saved CSV outputs**. The notebook also evaluates multiple hyperparameter candidates directly on the 2024 test set, then reports a separately configured model. This makes the reported holdout evaluation less independent. Its row-wise `TimeSeriesSplit` validation also needs a careful audit to ensure folds are grouped by month and preserve the intended forecasting protocol.

The new pipeline's Random Forest results differ from the older saved predictions because the implementation and model-selection protocol differ, including the validation-based tuning and the added numeric month feature. These results should not be compared as if they came from the same fitted model. The new numbers are preliminary and must be described as results of the current repository implementation, not as a replacement for the original submitted PDF until the research workflow and reporting are reconciled.

## Important forecasting protocol distinction

Random Forest lag features for a 2024 month can use observed counts from preceding 2024 months. That is a rolling one-month-ahead evaluation. SARIMAX forecasts all of 2024 from data through December 2023. These protocols are not a perfectly matched head-to-head comparison. A future revision should explicitly define whether the intended deployment scenario is rolling one-month-ahead prediction or a full-year forecast, then align both models accordingly.

## Privacy and publication

The source CSV contains incident-level records and sensitive location fields. It remains local and is excluded from Git. Generated CSVs are also ignored by default. Before any public release, review dataset redistribution terms and inspect every proposed derived output for disclosure risk. Aggregation reduces but does not automatically eliminate privacy or ethical concerns.

## Next steps

1. Decide and document the forecasting protocol.
2. Audit chronological validation by whole month, not arbitrary rows.
3. Re-run all metrics under one frozen protocol.
4. Update the final narrative and any PDF tables only after results are verified.
5. Keep raw incident-level data and unreviewed outputs out of the public repository.
