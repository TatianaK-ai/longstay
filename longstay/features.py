"""Feature definitions and encoders — and the gate the leakage check runs at.

`build_feature_matrix` is the ONLY supported way to hand data to a model. It
selects the whitelist from config and asserts against it, so a forbidden
column cannot reach an estimator even if someone adds one upstream.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config


def assert_only_allowed_features(frame: pd.DataFrame) -> None:
    """Fail if the model-bound frame holds anything outside ALLOWED_FEATURES.

    Mechanical enforcement of CLAUDE.md principle 3. This is deliberately a
    hard failure: a model trained on leaked columns looks excellent and is
    worthless, and that is not a failure mode you catch by reading a metric.
    """
    columns = list(frame.columns)

    forbidden = [c for c in columns if c in config.FORBIDDEN_COLUMNS]
    if forbidden:
        raise AssertionError(
            f"LEAKAGE: post-outcome columns reached the model: {forbidden}"
        )

    extra = [c for c in columns if c not in config.ALLOWED_FEATURES]
    if extra:
        raise AssertionError(
            f"Columns not in ALLOWED_FEATURES reached the model: {extra}"
        )

    missing = [c for c in config.ALLOWED_FEATURES if c not in columns]
    if missing:
        raise AssertionError(f"Expected features are missing: {missing}")


def build_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the allowed features, in a fixed order, and verify."""
    # Check before selecting: a bare frame[...] on a missing column raises a
    # pandas KeyError, which says less about what actually went wrong.
    missing = [c for c in config.ALLOWED_FEATURES if c not in frame.columns]
    if missing:
        raise AssertionError(f"Expected features are missing: {missing}")

    matrix = frame[config.ALLOWED_FEATURES].copy()
    assert_only_allowed_features(matrix)
    return matrix


def build_encoder() -> ColumnTransformer:
    """One-hot the categoricals, impute the numerics, pass booleans through.

    `primary_breed` and `primary_color` are high cardinality (thousands of
    values, a long tail seen once). `max_categories` folds the tail into a
    single infrequent bucket rather than exploding the matrix; the cap is
    TOP_BREEDS_N from config, the same number `clean` reports on.

    Missing `age_days` is imputed with the MEDIAN and flagged by an indicator
    column, so the model can distinguish "young" from "we do not know" instead
    of quietly treating an unknown age as the average one.
    """
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=config.TOP_BREEDS_N,
                    sparse_output=False,
                    min_frequency=1,
                ),
            ),
        ]
    )

    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    )

    boolean = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
            ("bool", boolean, config.BOOLEAN_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def temporal_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on INTAKE date into train / validation / test, in time order.

    Never randomly — CLAUDE.md principle 2. The validation period sits between
    train and test so that any choice made on validation is still blind to the
    test period. Boundaries are dates, not shuffles, because in production the
    model only ever sees the past.

    Returns (train, validation, test). The three are disjoint and, given
    contiguous config dates, exhaustive.
    """
    train_end = pd.Timestamp(config.TRAIN_END_DATE)
    val_start = pd.Timestamp(config.VAL_START_DATE)
    val_end = pd.Timestamp(config.VAL_END_DATE)
    test_start = pd.Timestamp(config.TEST_START_DATE)

    if not (train_end < val_start <= val_end < test_start):
        raise ValueError(
            "Split dates must be strictly ordered "
            "TRAIN_END < VAL_START <= VAL_END < TEST_START; got "
            f"{config.TRAIN_END_DATE}, {config.VAL_START_DATE}, "
            f"{config.VAL_END_DATE}, {config.TEST_START_DATE}"
        )

    # End dates are inclusive of the whole day, not of midnight only.
    day = pd.Timedelta(days=1)
    when = frame["intake_datetime"]

    train = frame[when < train_end + day]
    validation = frame[(when >= val_start) & (when < val_end + day)]
    test = frame[when >= test_start]
    return train, validation, test
