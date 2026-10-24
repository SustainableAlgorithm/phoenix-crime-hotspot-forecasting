import pandas as pd
import pytest
from src.forecasting import engineer_features, top_k_accuracy


def test_lag_features_use_only_previous_months():
    panel = pd.DataFrame({"Month": pd.period_range("2020-01", periods=4, freq="M"), "GRID": ["A"] * 4, "CrimeCount": [10, 20, 30, 40]})
    result = engineer_features(panel)
    assert result.loc[0, "PreviousMonthCount"] == 0
    assert result.loc[1, "PreviousMonthCount"] == 10
    assert result.loc[2, "PreviousMonthCount"] == 20
    assert result.loc[3, "Previous3MonthAverage"] == pytest.approx(20.0)


def test_top_k_accuracy_is_monthly_overlap_fraction():
    frame = pd.DataFrame({"Month": ["2024-01"] * 5, "GRID": list("ABCDE"), "actual": [10, 9, 8, 7, 6], "predicted": [10, 9, 1, 2, 3]})
    assert top_k_accuracy(frame, "actual", "predicted", k=2) == 1.0


def test_top_k_accuracy_empty_input_is_nan():
    frame = pd.DataFrame(columns=["Month", "GRID", "actual", "predicted"])
    assert pd.isna(top_k_accuracy(frame, "actual", "predicted"))
