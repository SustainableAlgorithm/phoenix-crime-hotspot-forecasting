"""Reproducible monthly crime-count forecasting utilities.

Raw incident identifiers and addresses are not included in aggregate outputs.
This is an academic analysis scaffold, not a police deployment tool.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
TRAIN_END_YEAR = 2023
TEST_YEAR = 2024
RANDOM_STATE = 42
TARGET = "CrimeCount"
CATEGORICAL_FEATURES = ["GRID", "Season"]
NUMERIC_FEATURES = ["Year", "MonthNumber", "PreviousMonthCount", "Previous3MonthAverage", "Previous6MonthAverage", "Previous12MonthAverage"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
CANDIDATE_PARAMS = [
    {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2, "max_features": "sqrt"},
]


def load_and_aggregate(input_csv: str | Path) -> pd.DataFrame:
    """Aggregate incident records to a complete month-by-grid panel."""
    raw = pd.read_csv(input_csv)
    required = {"OCCURRED ON", "GRID"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    dates = pd.to_datetime(raw["OCCURRED ON"].replace("", np.nan), errors="coerce")
    clean = pd.DataFrame({"DATE": dates, "GRID": raw["GRID"]}).dropna(subset=["DATE", "GRID"])
    clean = clean.loc[clean["DATE"].between(START_DATE, END_DATE)].copy()
    if clean.empty:
        raise ValueError(f"No usable records found between {START_DATE} and {END_DATE}.")
    clean["Month"] = clean["DATE"].dt.to_period("M")
    counts = clean.groupby(["Month", "GRID"], observed=True).size().rename(TARGET)
    months = pd.period_range("2020-01", "2024-12", freq="M")
    grids = np.sort(clean["GRID"].astype(str).unique())
    index = pd.MultiIndex.from_product([months, grids], names=["Month", "GRID"])
    panel = counts.reindex(index, fill_value=0).reset_index()
    panel["GRID"] = panel["GRID"].astype(str)
    return panel.sort_values(["GRID", "Month"]).reset_index(drop=True)


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and lagged features; rolling windows exclude current month."""
    required = {"Month", "GRID", TARGET}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")
    out = panel.copy()
    out["Month"] = pd.PeriodIndex(out["Month"], freq="M")
    out = out.sort_values(["GRID", "Month"]).reset_index(drop=True)
    out["Year"] = out["Month"].map(lambda p: p.year).astype(int)
    out["MonthNumber"] = out["Month"].map(lambda p: p.month).astype(int)
    month = out["MonthNumber"]
    out["Season"] = np.select([month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])], ["Spring", "Summer", "Fall"], default="Winter")
    grouped = out.groupby("GRID", sort=False)[TARGET]
    out["PreviousMonthCount"] = grouped.shift(1).fillna(0)
    for window in (3, 6, 12):
        name = f"Previous{window}MonthAverage"
        out[name] = grouped.transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()).fillna(0)
    return out


def make_model(params: dict) -> Pipeline:
    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ])
    return Pipeline([("preprocess", prep), ("model", RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1))])


def top_k_accuracy(frame: pd.DataFrame, actual_col: str, predicted_col: str, k: int = 5) -> float:
    """Average monthly overlap of actual and predicted top-k sets, divided by k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    scores = []
    for _, group in frame.groupby("Month", sort=True):
        actual = set(group.nlargest(k, actual_col)["GRID"])
        predicted = set(group.nlargest(k, predicted_col)["GRID"])
        scores.append(len(actual & predicted) / k)
    return float(np.mean(scores)) if scores else float("nan")


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {"MAE": float(mean_absolute_error(actual, predicted)), "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))), "R2": float(r2_score(actual, predicted))}


def run_random_forest(panel: pd.DataFrame, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tune on 2023, then evaluate once on the 2024 holdout year."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = panel.loc[panel["Year"] <= 2022]
    validation = panel.loc[panel["Year"] == 2023]
    final_train = panel.loc[panel["Year"] <= TRAIN_END_YEAR]
    test = panel.loc[panel["Year"] == TEST_YEAR]
    if any(frame.empty for frame in (train, validation, final_train, test)):
        raise ValueError("Need usable records in 2020–2023 for training/validation and 2024 for testing.")
    tuning_rows = []
    for params in CANDIDATE_PARAMS:
        candidate = make_model(params).fit(train[FEATURES], train[TARGET])
        predicted = np.clip(candidate.predict(validation[FEATURES]), 0, None)
        scored = validation[["Month", "GRID", TARGET]].copy()
        scored["PredictedCrimeCount"] = predicted
        tuning_rows.append({**params, "ValidationTop5Accuracy": top_k_accuracy(scored, TARGET, "PredictedCrimeCount"), "ValidationRMSE": float(np.sqrt(mean_squared_error(validation[TARGET], predicted)))})
    tuning = pd.DataFrame(tuning_rows).sort_values(["ValidationTop5Accuracy", "ValidationRMSE"], ascending=[False, True]).reset_index(drop=True)
    best = {key: tuning.loc[0, key] for key in CANDIDATE_PARAMS[0]}
    for key in ("n_estimators", "max_depth", "min_samples_leaf"):
        best[key] = int(best[key])
    model = make_model(best).fit(final_train[FEATURES], final_train[TARGET])
    predicted = np.clip(model.predict(test[FEATURES]), 0, None)
    results = test[["Month", "GRID", TARGET]].copy()
    results["PredictedCrimeCount"] = predicted
    results["ActualRank"] = results.groupby("Month")[TARGET].rank(method="first", ascending=False)
    results["PredictedRank"] = results.groupby("Month")["PredictedCrimeCount"].rank(method="first", ascending=False)
    results["ActualHotspot"] = (results["ActualRank"] <= 5).astype(int)
    results["PredictedHotspot"] = (results["PredictedRank"] <= 5).astype(int)
    monthly_rows = []
    for period, group in results.groupby("Month", sort=True):
        actual = set(group.loc[group["ActualHotspot"].eq(1), "GRID"])
        predicted_set = set(group.loc[group["PredictedHotspot"].eq(1), "GRID"])
        correct = len(actual & predicted_set)
        monthly_rows.append({"Month": str(period), "CorrectTop5": correct, "Top5Accuracy": correct / 5})
    monthly = pd.DataFrame(monthly_rows)
    summary = regression_metrics(test[TARGET], predicted)
    summary.update({"MeanCorrectTop5": float(monthly["CorrectTop5"].mean()), "MeanTop5Accuracy": float(monthly["Top5Accuracy"].mean()), "BestParameters": best})
    results["Month"] = results["Month"].astype(str)
    results.to_csv(output_dir / "RandomForestPredictions.csv", index=False)
    monthly.to_csv(output_dir / "RandomForestHotspotAccuracy.csv", index=False)
    tuning.to_csv(output_dir / "RandomForestValidationTuning.csv", index=False)
    return results, monthly, summary


def run_sarimax(panel: pd.DataFrame, output_dir: str | Path, min_historical_top5_months: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast recurring historical hotspot grids with a fixed SARIMAX model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = panel.loc[panel["Year"] <= TRAIN_END_YEAR].copy()
    test = panel.loc[panel["Year"] == TEST_YEAR].copy()
    hotspot_counts = train.loc[train["Hotspot"].eq(1)].groupby("GRID").size()
    grids = hotspot_counts.loc[hotspot_counts >= min_historical_top5_months].index.tolist()
    metrics, predictions = [], []
    for grid in grids:
        train_series = train.loc[train["GRID"].eq(grid)].set_index("Month")[TARGET].sort_index()
        actual = test.loc[test["GRID"].eq(grid)].set_index("Month")[TARGET].sort_index()
        if len(actual) != 12 or len(train_series) < 24:
            warnings.warn(f"Skipping grid {grid}: insufficient monthly history.")
            continue
        try:
            fitted = SARIMAX(train_series.astype(float), order=(1, 1, 0), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            forecast = fitted.forecast(steps=12).clip(lower=0)
            forecast.index = actual.index
            metrics.append({"GRID": grid, **regression_metrics(actual, forecast.to_numpy()), "AIC": float(fitted.aic)})
            for period, actual_value, forecast_value in zip(actual.index, actual.to_numpy(), forecast.to_numpy()):
                predictions.append({"Month": str(period), "GRID": grid, "ActualCrimeCount": int(actual_value), "PredictedCrimeCount": int(round(float(forecast_value)))})
        except Exception as exc:
            warnings.warn(f"SARIMAX failed for grid {grid}: {exc}")
    metrics_df = pd.DataFrame(metrics, columns=["GRID", "MAE", "RMSE", "R2", "AIC"])
    predictions_df = pd.DataFrame(predictions, columns=["Month", "GRID", "ActualCrimeCount", "PredictedCrimeCount"])
    metrics_df.to_csv(output_dir / "SARIMAX_Grid_Metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "SARIMAX_Grid_Predictions.csv", index=False)
    return metrics_df, predictions_df


def run_pipeline(input_csv: str | Path, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = engineer_features(load_and_aggregate(input_csv))
    panel["Rank"] = panel.groupby("Month")[TARGET].rank(method="first", ascending=False).astype(int)
    panel["Hotspot"] = (panel["Rank"] <= 5).astype(int)
    panel.to_csv(output_dir / "phoenix_monthly_hotspots.csv", index=False)
    _, _, rf_summary = run_random_forest(panel, output_dir)
    sarimax_metrics, sarimax_predictions = run_sarimax(panel, output_dir)
    return {"rows_monthly_panel": int(len(panel)), "unique_grids": int(panel["GRID"].nunique()), "random_forest": rf_summary, "sarimax_grids_evaluated": int(len(sarimax_metrics)), "sarimax_predictions": int(len(sarimax_predictions)), "output_dir": str(output_dir)}
