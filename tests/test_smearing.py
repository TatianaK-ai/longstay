"""Duan smearing — the retransformation bias correction.

Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method."
JASA 78(383), 605-610.

The property that makes this a correction rather than tuning: it is strictly
monotone, so it moves the level and cannot move the ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from longstay import config
from longstay.evaluate import is_rank_preserving
from longstay.model import GradientBoostingModel
from tests.test_model import synthetic_frame


@pytest.fixture(scope="module")
def fitted():
    frame = synthetic_frame(800)
    model = GradientBoostingModel(max_iter=60).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    return model, frame


def test_uncorrected_model_under_predicts_the_mean(fitted):
    """The defect being corrected — naive expm1 is biased low."""
    model, frame = fitted
    actual_mean = frame[config.TARGET].mean()
    naive_mean = model.predict_uncorrected(frame).mean()
    assert naive_mean < actual_mean


def test_smearing_factor_is_above_one_for_a_skewed_target(fitted):
    model, frame = fitted
    model.fit_smearing(frame, frame[config.TARGET].to_numpy())
    assert model.smearing_factor_ > 1.0


def test_smearing_reduces_mean_bias(fitted):
    model, frame = fitted
    y = frame[config.TARGET].to_numpy()
    model.fit_smearing(frame, y)

    bias_before = abs(model.predict_uncorrected(frame).mean() - y.mean())
    bias_after = abs(model.predict(frame).mean() - y.mean())
    assert bias_after < bias_before


def test_smearing_does_not_change_the_ranking(fitted):
    """The single most important property. If this fails, something is wrong."""
    model, frame = fitted
    model.fit_smearing(frame, frame[config.TARGET].to_numpy())

    before = model.predict_uncorrected(frame)
    after = model.predict(frame)

    assert np.array_equal(
        np.argsort(before, kind="mergesort"),
        np.argsort(after, kind="mergesort"),
    )


def test_smearing_preserves_spearman_correlation_exactly(fitted):
    model, frame = fitted
    model.fit_smearing(frame, frame[config.TARGET].to_numpy())
    correlation = np.corrcoef(
        np.argsort(np.argsort(model.predict_uncorrected(frame))),
        np.argsort(np.argsort(model.predict(frame))),
    )[0, 1]
    assert correlation == pytest.approx(1.0)


def test_closed_form_matches_the_definition(fitted):
    """exp(f)*S - 1 must equal the mean of expm1(f + e_i) over residuals.

    The closed form is what the code uses; this checks it against Duan's
    definition applied literally, so the algebra is verified, not assumed.
    """
    model, frame = fitted
    y = frame[config.TARGET].to_numpy()
    model.fit_smearing(frame, y)

    log_predictions = model._log_prediction(frame)
    residuals = np.log1p(y) - log_predictions

    # Literal definition, on a handful of rows.
    literal = np.array(
        [np.mean(np.expm1(f + residuals)) for f in log_predictions[:20]]
    )
    closed_form = (
        np.exp(log_predictions[:20]) * model.smearing_factor_ - 1.0
    )
    np.testing.assert_allclose(literal, closed_form, rtol=1e-9)


def test_uncorrected_model_is_unchanged_by_fitting_smearing(fitted):
    """The fit is untouched; only the back-transform changes."""
    model, frame = fitted
    before = model.predict_uncorrected(frame).copy()
    model.fit_smearing(frame, frame[config.TARGET].to_numpy())
    np.testing.assert_array_equal(before, model.predict_uncorrected(frame))


def test_raw_absolute_model_refuses_smearing():
    """No transform means no retransformation bias and nothing to correct."""
    frame = synthetic_frame(200)
    model = GradientBoostingModel(mode="raw_absolute", max_iter=20).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    with pytest.raises(ValueError, match="no transform"):
        model.fit_smearing(frame, frame[config.TARGET].to_numpy())


def test_raw_absolute_predicts_directly_without_a_back_transform():
    frame = synthetic_frame(400)
    model = GradientBoostingModel(mode="raw_absolute", max_iter=30).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    np.testing.assert_array_equal(
        model.predict(frame), model.predict_uncorrected(frame)
    )


def test_raw_absolute_optimises_the_median_so_it_beats_log1p_on_median_error():
    """Why raw_absolute was chosen: the loss fitted is the loss reported."""
    frame = synthetic_frame(1200)
    y = frame[config.TARGET].to_numpy()

    raw = GradientBoostingModel(mode="raw_absolute", max_iter=80).fit(frame, y)
    log = GradientBoostingModel(mode="log1p", max_iter=80).fit(frame, y)
    log.fit_smearing(frame, y)

    raw_error = np.median(np.abs(raw.predict(frame) - y))
    smeared_error = np.median(np.abs(log.predict(frame) - y))
    assert raw_error < smeared_error


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown mode"):
        GradientBoostingModel(mode="magic")


def test_quantile_models_refuse_smearing():
    """Quantiles are transform-invariant; rescaling them would be wrong."""
    frame = synthetic_frame(200)
    model = GradientBoostingModel(quantile=0.5, max_iter=20).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    with pytest.raises(ValueError, match="Quantiles"):
        model.fit_smearing(frame, frame[config.TARGET].to_numpy())


def test_rank_check_tolerates_ties_in_the_input():
    """Predictions are clipped at 0, so ties in the BEFORE array are normal.

    A tied block was never ordered, so no ordering within it can be inverted.
    The naive sort-and-diff check reads that block as an inversion; this is
    the case that produced a false "ranking changed" report.
    """
    before = np.array([0.0, 0.0, 0.0, 5.0, 9.0])
    after = np.array([3.0, 1.0, 2.0, 14.0, 25.0])  # tied block, any order
    assert is_rank_preserving(before, after)


def test_rank_check_still_catches_a_real_inversion_across_tied_blocks():
    before = np.array([0.0, 0.0, 5.0, 9.0])
    after = np.array([1.0, 2.0, 30.0, 10.0])  # 5 -> 30 but 9 -> 10: inverted
    assert not is_rank_preserving(before, after)


def test_smearing_on_the_real_shape_reports_no_inversions():
    frame = synthetic_frame(600)
    model = GradientBoostingModel(max_iter=40).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    model.fit_smearing(frame, frame[config.TARGET].to_numpy())
    assert is_rank_preserving(
        model.predict_uncorrected(frame), model.predict(frame)
    )


def test_untouched_model_has_a_neutral_factor():
    frame = synthetic_frame(200)
    model = GradientBoostingModel(max_iter=20).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    assert model.smearing_factor_ == 1.0
    np.testing.assert_allclose(
        model.predict(frame), model.predict_uncorrected(frame), atol=1e-9
    )
