"""Evaluation maths: metrics, deciles, coverage, tail analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longstay import config
from longstay.evaluate import (
    core_metrics,
    interval_calibration,
    metrics_by_decile,
    metrics_by_group,
    predicted_deciles,
    tail_analysis,
    worst_predictions,
    characterise_failures,
)
from tests.test_model import synthetic_frame


# ------------------------------------------------------------------ metrics


def test_perfect_predictions():
    y = np.array([1.0, 10.0, 100.0])
    metrics = core_metrics(y, y)
    assert metrics["mae_days"] == 0.0
    assert metrics["median_absolute_error_days"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["within_7_days_rate"] == 1.0


def test_mae_is_in_days():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 26.0])
    assert core_metrics(y_true, y_pred)["mae_days"] == pytest.approx(4.0)


def test_within_tolerance_rate_counts_the_boundary_as_inside():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([7.0, 7.01, 14.0])
    metrics = core_metrics(y_true, y_pred)
    assert metrics["within_7_days_rate"] == pytest.approx(1 / 3)
    assert metrics["within_14_days_rate"] == 1.0


def test_bias_sign_means_over_prediction():
    y_true = np.array([10.0])
    y_pred = np.array([15.0])
    assert core_metrics(y_true, y_pred)["mean_bias_days"] == 5.0


def test_metrics_by_group_splits_correctly():
    rows = metrics_by_group(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([12.0, 22.0, 30.0, 40.0]),
        pd.Series(["Dog", "Dog", "Cat", "Cat"]),
    )
    by_name = {row["group"]: row for row in rows}
    assert by_name["Dog"]["mae_days"] == pytest.approx(2.0)
    assert by_name["Cat"]["mae_days"] == pytest.approx(0.0)


# ------------------------------------------------------------------ deciles


def test_deciles_split_into_ten_bins():
    predictions = np.arange(1000, dtype=float)
    assert predicted_deciles(predictions).nunique() == 10


def test_constant_predictions_collapse_to_one_bin_not_a_crash():
    """Baseline 1 predicts one number for everyone."""
    rows = metrics_by_decile(
        np.arange(100, dtype=float), np.full(100, 6.0)
    )
    assert len(rows) >= 1
    assert sum(row["n"] for row in rows) == 100


def test_decile_rows_account_for_every_row():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 300, 500)
    y_pred = rng.uniform(0, 300, 500)
    rows = metrics_by_decile(y_true, y_pred)
    assert sum(row["n"] for row in rows) == 500


def test_deciles_are_ordered_by_prediction():
    rng = np.random.default_rng(1)
    y_pred = rng.uniform(0, 300, 1000)
    rows = metrics_by_decile(rng.uniform(0, 300, 1000), y_pred)
    means = [row["mean_predicted_days"] for row in rows]
    assert means == sorted(means)


# -------------------------------------------------------------- calibration


def test_coverage_is_measured_not_assumed():
    y_true = np.array([5.0, 50.0, 500.0])
    quantiles = {
        0.1: np.array([1.0, 1.0, 1.0]),
        0.5: np.array([10.0, 10.0, 10.0]),
        0.9: np.array([100.0, 100.0, 100.0]),
    }
    result = interval_calibration(y_true, quantiles)
    assert result["empirical_coverage"] == pytest.approx(2 / 3)
    assert result["above_interval_rate"] == pytest.approx(1 / 3)
    assert result["below_interval_rate"] == 0.0


def test_perfectly_covered_intervals_report_one():
    y_true = np.array([5.0, 6.0])
    quantiles = {
        0.1: np.array([0.0, 0.0]),
        0.5: np.array([5.0, 6.0]),
        0.9: np.array([10.0, 10.0]),
    }
    assert interval_calibration(y_true, quantiles)["empirical_coverage"] == 1.0


def test_coverage_counts_the_boundary_as_inside():
    y_true = np.array([1.0, 10.0])
    quantiles = {
        0.1: np.array([1.0, 1.0]),
        0.5: np.array([5.0, 5.0]),
        0.9: np.array([10.0, 10.0]),
    }
    assert interval_calibration(y_true, quantiles)["empirical_coverage"] == 1.0


# ----------------------------------------------------------- failure analysis


def test_worst_predictions_returns_the_largest_errors():
    frame = synthetic_frame(100)
    y_pred = frame[config.TARGET].to_numpy().copy()
    y_pred[7] += 300.0  # one deliberate disaster

    worst = worst_predictions(frame, y_pred, n=5)
    assert len(worst) == 5
    assert worst["absolute_error_days"].iloc[0] == pytest.approx(300.0)
    assert worst["absolute_error_days"].is_monotonic_decreasing


def test_signed_error_distinguishes_under_from_over_prediction():
    frame = synthetic_frame(50)
    y_pred = frame[config.TARGET].to_numpy() - 100.0  # always under-predicts
    summary = characterise_failures(worst_predictions(frame, y_pred, n=10))
    assert summary["under_predicted"] == 10
    assert summary["over_predicted"] == 0


def test_tail_analysis_reports_under_prediction():
    """The disclosure that matters most: does it miss the long stays?"""
    frame = synthetic_frame(200)
    frame[config.TARGET] = np.where(
        np.arange(200) < 20, 150.0, 5.0
    )  # 20 genuine long stays
    y_pred = np.full(200, 6.0)  # a model that never predicts a long stay

    result = tail_analysis(frame, y_pred)
    assert result["n"] == 20
    assert result["under_predicted_rate"] == 1.0
    assert result["predicted_over_threshold_rate"] == 0.0
    assert result["mean_bias_days"] < 0


def test_tail_analysis_handles_no_long_stays():
    frame = synthetic_frame(50)
    frame[config.TARGET] = 3.0
    assert tail_analysis(frame, np.full(50, 3.0))["n"] == 0


def test_tail_recall_is_computed_from_the_ranking():
    frame = synthetic_frame(1000)
    # Long stays are exactly the animals the model ranks highest.
    frame[config.TARGET] = np.where(np.arange(1000) < 100, 200.0, 4.0)
    y_pred = np.where(np.arange(1000) < 100, 90.0, 4.0)

    result = tail_analysis(frame, y_pred)
    assert result["recall_in_top_decile"] == pytest.approx(1.0)
