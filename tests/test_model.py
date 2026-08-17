"""Models: the baselines, the log transform, and the leakage gate at fit time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longstay import config
from longstay.model import (
    GlobalMedianBaseline,
    GradientBoostingModel,
    GroupMedianBaseline,
    LongstayModels,
    age_bucket,
)


def synthetic_frame(n: int = 600) -> pd.DataFrame:
    """A frame with the real column set and a learnable signal."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    animal_type = rng.choice(["Dog", "Cat"], size=n)
    age_days = rng.choice([30.0, 200.0, 900.0, 3000.0], size=n)

    # Cats and old animals wait longer, plus noise. Something to find.
    base = np.where(animal_type == "Cat", 20.0, 8.0) + age_days / 200.0
    target = np.clip(base * rng.lognormal(0, 0.4, size=n), 0, 365)

    return pd.DataFrame(
        {
            "animal_id": [f"A{i}" for i in range(n)],
            "intake_datetime": pd.date_range("2020-01-01", periods=n, freq="D"),
            "outcome_datetime": pd.date_range("2020-02-01", periods=n, freq="D"),
            config.TARGET: target,
            "age_days": age_days,
            "animal_type": animal_type,
            "sex": rng.choice(["Male", "Female"], size=n),
            "sterilization_status": rng.choice(["Fixed", "Intact"], size=n),
            "intake_type": rng.choice(["Stray", "Owner Surrender"], size=n),
            "intake_condition": rng.choice(["Normal", "Injured"], size=n),
            "primary_breed": rng.choice(
                ["Beagle", "Pit Bull", "Domestic Shorthair"], size=n
            ),
            "primary_color": rng.choice(["Black", "White", "Brown"], size=n),
            "intake_month": rng.integers(1, 13, size=n),
            "intake_day_of_week": rng.integers(0, 7, size=n),
            "intake_season": rng.choice(["Winter", "Summer"], size=n),
            "is_mix": rng.choice([True, False], size=n),
            "is_black": rng.choice([True, False], size=n),
            "has_name": rng.choice([True, False], size=n),
        }
    )


# ------------------------------------------------------------------ buckets


def test_age_bucket_labels():
    ages = pd.Series([10.0, 90.0, 250.0, 700.0, 2000.0, 4000.0])
    assert list(age_bucket(ages)) == [
        "under 2mo", "2-6mo", "6-12mo", "1-3y", "3-7y", "7y+"
    ]


def test_age_bucket_handles_nulls_explicitly():
    """Unknown age is its own bucket, not an imputed guess."""
    assert list(age_bucket(pd.Series([np.nan]))) == [config.AGE_BUCKET_UNKNOWN]


# ---------------------------------------------------------------- baseline 1


def test_global_median_baseline_predicts_the_training_median():
    X = pd.DataFrame({"animal_type": ["Dog"] * 5})
    y = np.array([1.0, 2.0, 3.0, 100.0, 200.0])
    model = GlobalMedianBaseline().fit(X, y)
    assert model.median_ == 3.0
    assert np.all(model.predict(X) == 3.0)


def test_median_is_invariant_under_the_log_transform():
    """Why the baselines need no special handling for the skewed target."""
    y = np.array([1.0, 2.0, 3.0, 100.0, 200.0])
    direct = np.median(y)
    via_log = np.expm1(np.median(np.log1p(y)))
    assert direct == pytest.approx(via_log)


# ---------------------------------------------------------------- baseline 2


def test_group_median_baseline_uses_the_right_cell():
    X = pd.DataFrame(
        {
            "animal_type": ["Dog", "Dog", "Cat", "Cat"],
            "age_days": [30.0, 30.0, 30.0, 30.0],
        }
    )
    y = np.array([4.0, 6.0, 40.0, 60.0])
    model = GroupMedianBaseline().fit(X, y)

    predictions = model.predict(X)
    assert predictions[0] == 5.0   # Dog / under 2mo
    assert predictions[2] == 50.0  # Cat / under 2mo


def test_group_median_baseline_falls_back_for_unseen_cells():
    train_X = pd.DataFrame({"animal_type": ["Dog"], "age_days": [30.0]})
    model = GroupMedianBaseline().fit(train_X, np.array([5.0]))

    unseen = pd.DataFrame({"animal_type": ["Bird"], "age_days": [3000.0]})
    assert model.predict(unseen)[0] == model.fallback_
    assert model.unseen_cell_rate(unseen) == 1.0


def test_group_median_baseline_never_returns_nan():
    frame = synthetic_frame()
    model = GroupMedianBaseline().fit(frame, frame[config.TARGET].to_numpy())
    unseen = frame.copy()
    unseen["animal_type"] = "Livestock"
    assert np.isfinite(model.predict(unseen)).all()


# ---------------------------------------------------------------------- GBM


def test_gbm_predictions_are_in_days_not_log_space():
    frame = synthetic_frame()
    y = frame[config.TARGET].to_numpy()
    model = GradientBoostingModel(max_iter=30).fit(frame, y)
    predictions = model.predict(frame)

    # Log-space predictions would sit around 2-4; days sit around 10-40.
    assert predictions.mean() > 5
    assert predictions.max() <= config.MAX_LOS_DAYS
    assert predictions.min() >= config.MIN_LOS_DAYS


def test_gbm_beats_the_global_median_on_data_with_real_signal():
    """A sanity check on the plumbing, not a claim about the shelter data."""
    frame = synthetic_frame()
    y = frame[config.TARGET].to_numpy()

    gbm = GradientBoostingModel(max_iter=60).fit(frame, y)
    baseline = GlobalMedianBaseline().fit(frame, y)

    gbm_error = np.abs(gbm.predict(frame) - y).mean()
    baseline_error = np.abs(baseline.predict(frame) - y).mean()
    assert gbm_error < baseline_error


def test_quantile_models_are_ordered():
    frame = synthetic_frame()
    y = frame[config.TARGET].to_numpy()

    low = GradientBoostingModel(quantile=0.1, max_iter=40).fit(frame, y)
    high = GradientBoostingModel(quantile=0.9, max_iter=40).fit(frame, y)

    assert low.predict(frame).mean() < high.predict(frame).mean()


def test_unseen_categories_do_not_break_prediction():
    """A breed first seen in the test period maps to Other, not a crash."""
    frame = synthetic_frame()
    model = GradientBoostingModel(max_iter=30).fit(
        frame, frame[config.TARGET].to_numpy()
    )

    future = frame.copy()
    future["primary_breed"] = "Norwegian Lundehund"  # never seen in training
    predictions = model.predict(future)
    assert np.isfinite(predictions).all()


def test_encoder_categories_come_from_training_only(monkeypatch):
    """Category sets are frozen at fit time — test composition must not leak in."""
    frame = synthetic_frame()
    model = GradientBoostingModel(max_iter=20).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    learned = set(model.encoder_.categories_["primary_breed"])

    future = frame.copy()
    future["primary_breed"] = "Brand New Breed"
    model.predict(future)

    assert set(model.encoder_.categories_["primary_breed"]) == learned


def test_category_cap_respects_the_hgb_bin_limit():
    """primary_breed has thousands of levels; native categoricals allow 255."""
    frame = synthetic_frame(400)
    frame["primary_breed"] = [f"Breed {i}" for i in range(len(frame))]
    model = GradientBoostingModel(max_iter=20).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    for column, categories in model.encoder_.categories_.items():
        assert len(categories) <= config.MAX_MODEL_CATEGORIES + 2, column


def test_no_one_hot_explosion():
    """The whole point of native categorical support."""
    frame = synthetic_frame()
    model = GradientBoostingModel(max_iter=20).fit(
        frame, frame[config.TARGET].to_numpy()
    )
    assert len(model.feature_names_) == len(config.ALLOWED_FEATURES)


# ------------------------------------------------------------------ leakage


def test_fitting_rejects_a_leaked_column():
    """LongstayModels.fit goes through build_feature_matrix, which asserts."""
    frame = synthetic_frame()
    frame["outcome_type"] = "Adoption"
    # build_feature_matrix selects the whitelist, so this must still be clean:
    models = LongstayModels().fit(frame)
    assert "outcome_type" not in models.gbm.feature_names_


def test_fitting_fails_when_a_required_feature_is_missing():
    frame = synthetic_frame().drop(columns=["is_mix"])
    with pytest.raises(AssertionError, match="missing"):
        LongstayModels().fit(frame)
