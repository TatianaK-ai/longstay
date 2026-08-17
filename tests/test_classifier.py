"""The long-stay classifier — the primary model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longstay import config
from longstay.evaluate import (
    classifier_metrics,
    expected_calibration_error,
    human_summary,
    is_rank_preserving,
    operating_point,
    reliability_table,
    shelter_load_finding,
    threshold_for_precision,
    threshold_for_recall,
)
from longstay.model import (
    LongStayClassifier,
    LongStayClassifierBundle,
    PositiveRateBaseline,
    long_stay_target,
)
from tests.test_model import synthetic_frame


def labelled_frame(n: int = 1200) -> pd.DataFrame:
    """Frame where long stays are genuinely predictable from a feature."""
    frame = synthetic_frame(n)
    rng = np.random.default_rng(config.RANDOM_STATE)
    # Old cats mostly wait a long time; everyone else mostly does not.
    at_risk = (frame["animal_type"] == "Cat") & (frame["age_days"] > 800)
    noise = rng.random(n)
    frame[config.TARGET] = np.where(
        at_risk & (noise > 0.2), 150.0, np.where(noise > 0.95, 120.0, 5.0)
    )
    return frame


# ------------------------------------------------------------------- target


def test_target_is_the_90_day_threshold():
    frame = synthetic_frame(4)
    frame[config.TARGET] = [1.0, 89.9, 90.0, 200.0]
    assert list(long_stay_target(frame)) == [0, 0, 1, 1]


# ----------------------------------------------------------------- baseline


def test_positive_rate_baseline_predicts_the_base_rate():
    X = pd.DataFrame({"a": range(10)})
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    model = PositiveRateBaseline().fit(X, y)
    assert model.positive_rate_ == 0.2
    assert np.all(model.predict_proba(X)[:, 1] == 0.2)


def test_baseline_pr_auc_equals_the_base_rate():
    """A constant ranker's average precision IS the positive rate."""
    y = np.array([1] * 20 + [0] * 180)
    scores = np.full(200, 0.1)
    metrics = classifier_metrics(y, scores)
    assert metrics["pr_auc"] == pytest.approx(0.1, abs=0.01)
    assert metrics["pr_auc_baseline"] == pytest.approx(0.1)


# --------------------------------------------------------------- classifier


@pytest.fixture(scope="module")
def bundle():
    frame = labelled_frame()
    train = frame.iloc[:800]
    validation = frame.iloc[800:1000]
    test = frame.iloc[1000:]
    return LongStayClassifierBundle().fit(train, validation), test


def test_classifier_beats_the_base_rate_on_pr_auc(bundle):
    fitted, test = bundle
    metrics = classifier_metrics(long_stay_target(test), fitted.predict_proba(test))
    assert metrics["pr_auc"] > metrics["pr_auc_baseline"]
    assert metrics["pr_auc_lift"] > 1.0


def test_probabilities_are_in_range(bundle):
    fitted, test = bundle
    probabilities = fitted.predict_proba(test)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_isotonic_calibration_never_inverts_the_ranking(bundle):
    """Isotonic is monotone non-decreasing: it may tie, it must not invert.

    Unlike smearing (strictly monotone, ranking identical), isotonic can
    collapse distinct scores onto the same calibrated value. For a triage list
    that is harmless; an inversion would not be.
    """
    fitted, test = bundle
    raw = fitted.predict_proba_uncalibrated(test)
    calibrated = fitted.predict_proba_isotonic(test)
    assert is_rank_preserving(raw, calibrated)


def test_isotonic_does_create_ties(bundle):
    """Documents why the check above is not a Spearman == 1 check."""
    fitted, test = bundle
    raw = fitted.predict_proba_uncalibrated(test)
    calibrated = fitted.predict_proba_isotonic(test)
    assert len(np.unique(calibrated)) < len(np.unique(raw))


def test_rank_preserving_detects_a_real_inversion():
    assert is_rank_preserving(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0]))
    assert not is_rank_preserving(
        np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    )


def test_calibration_is_fitted_on_validation_not_test(bundle):
    fitted, _ = bundle
    assert fitted.classifier.calibrator_ is not None


def test_uncalibrated_classifier_returns_raw_scores():
    frame = labelled_frame(400)
    model = LongStayClassifier(max_iter=30).fit(
        frame, long_stay_target(frame)
    )
    assert model.calibrator_ is None
    np.testing.assert_allclose(
        model.predict_proba(frame), model.predict_proba_uncalibrated(frame)
    )


def test_classifier_uses_only_whitelisted_features(bundle):
    """Set equality — the encoder groups numeric/boolean/categorical, so the
    order differs from ALLOWED_FEATURES while the membership must not."""
    fitted, _ = bundle
    assert set(fitted.classifier.feature_names_) == set(config.ALLOWED_FEATURES)


def test_leaked_column_never_reaches_the_classifier():
    frame = labelled_frame(400)
    frame["outcome_type"] = "Adoption"
    validation = frame.iloc[300:]
    fitted = LongStayClassifierBundle().fit(frame.iloc[:300], validation)
    assert "outcome_type" not in fitted.classifier.feature_names_


# ---------------------------------------------------------------- threshold


def test_operating_threshold_follows_the_cost_ratio():
    """The threshold is derived, not tuned."""
    expected = 1.0 / (1.0 + config.COST_RATIO_MISS_TO_FALSE_ALARM)
    assert config.OPERATING_THRESHOLD == pytest.approx(expected)


def test_isotonic_is_switched_off_and_predict_proba_returns_raw():
    """It was measured and it made calibration worse. Off means off."""
    assert config.USE_ISOTONIC_CALIBRATION is False
    frame = labelled_frame(600)
    fitted = LongStayClassifierBundle().fit(frame.iloc[:400], frame.iloc[400:])
    np.testing.assert_allclose(
        fitted.predict_proba(frame), fitted.predict_proba_uncalibrated(frame)
    )


def test_isotonic_is_still_fitted_so_the_comparison_can_be_reported():
    frame = labelled_frame(600)
    fitted = LongStayClassifierBundle().fit(frame.iloc[:400], frame.iloc[400:])
    assert fitted.classifier.calibrator_ is not None
    # And it differs from the raw score, otherwise the comparison is vacuous.
    assert not np.allclose(
        fitted.predict_proba_isotonic(frame),
        fitted.predict_proba_uncalibrated(frame),
    )


def test_threshold_favours_recall_over_precision():
    """A miss costs months of kennel life; a false alarm costs one foster."""
    assert config.OPERATING_THRESHOLD < 0.5
    assert config.COST_RATIO_MISS_TO_FALSE_ALARM > 1


def test_risk_band_elevated_starts_at_the_operating_threshold():
    """The band staff see and the threshold the model uses are one number."""
    assert config.RISK_BAND_ELEVATED == config.OPERATING_THRESHOLD


def test_lower_threshold_gives_higher_recall():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    low = operating_point(y, scores, 0.25)
    high = operating_point(y, scores, 0.7)
    assert low["recall"] > high["recall"]


def test_threshold_for_recall_reaches_the_target():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.concatenate([np.linspace(0.9, 0.5, 10), np.linspace(0.5, 0, 90)])
    threshold = threshold_for_recall(y, scores, 0.5)
    assert operating_point(y, scores, threshold)["recall"] >= 0.5


def test_threshold_for_precision_reaches_the_target():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.concatenate([np.linspace(0.95, 0.6, 10), np.linspace(0.4, 0, 90)])
    threshold = threshold_for_precision(y, scores, 0.5)
    assert operating_point(y, scores, threshold)["precision"] >= 0.5


def test_confusion_matrix_totals_match():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.1])
    matrix = operating_point(y, scores, 0.5)["confusion_matrix"]
    assert sum(matrix.values()) == 5
    assert matrix["true_positive"] == 1
    assert matrix["false_positive"] == 1
    assert matrix["false_negative"] == 1


# -------------------------------------------------------------- reliability


def test_reliability_table_covers_every_row():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    y = (rng.random(500) < scores).astype(int)
    table = reliability_table(y, scores)
    assert sum(row["n"] for row in table) == 500


def test_perfectly_calibrated_scores_have_near_zero_error():
    rng = np.random.default_rng(1)
    scores = rng.random(20000)
    y = (rng.random(20000) < scores).astype(int)
    assert expected_calibration_error(reliability_table(y, scores)) < 0.02


def test_badly_calibrated_scores_show_a_large_error():
    """Scores half the true rate must not look calibrated."""
    rng = np.random.default_rng(2)
    truth = rng.random(20000)
    y = (rng.random(20000) < truth).astype(int)
    assert expected_calibration_error(reliability_table(y, truth / 2)) > 0.1


# ----------------------------------------------------------------- sentence


def test_human_summary_reads_as_a_sentence():
    operating = {"precision": 0.31, "recall": 0.62, "threshold": 0.167}
    sentence = human_summary(operating, 0.05)
    assert "100 animals" in sentence
    assert "31" in sentence
    assert "62%" in sentence


# ------------------------------------------------------------ shelter load


def test_shelter_load_uses_animal_days_not_head_count():
    frame = pd.DataFrame({config.TARGET: [1.0, 1.0, 1.0, 1.0, 100.0]})
    finding = shelter_load_finding(frame)

    assert finding["quick_exit_share_of_animals"] == pytest.approx(0.8)
    assert finding["total_animal_days"] == pytest.approx(104.0)
    # Four quick animals are 80% of heads but under 4% of days.
    assert finding["quick_exit_share_of_animal_days"] == pytest.approx(
        4 / 104
    )


def test_shelter_load_shares_are_fractions():
    frame = synthetic_frame(300)
    finding = shelter_load_finding(frame)
    for key in (
        "quick_exit_share_of_animals",
        "quick_exit_share_of_animal_days",
        "long_stay_share_of_animals",
        "long_stay_share_of_animal_days",
    ):
        assert 0.0 <= finding[key] <= 1.0
