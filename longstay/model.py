"""Models: two baselines and a gradient boosting regressor.

The baselines are not decoration. Baseline 1 is the number every other model
must beat. Baseline 2 — a lookup table of medians by animal_type and age
bucket — is the one that quietly defeats a lot of "AI projects" whose authors
never built it and so never found out.

Everything here predicts in DAYS. The gradient boosting model fits on
log1p(days) internally and inverts with expm1 before returning, so callers
never handle log space. The target is heavily right-skewed, but "0.4 log days"
is not a thing a shelter worker can act on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression

from . import config
from .features import build_feature_matrix

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def age_bucket(age_days: pd.Series) -> pd.Series:
    """Coarse age bands for the lookup-table baseline.

    Nulls become an explicit "unknown" bucket rather than being dropped or
    imputed — "we do not know how old this animal is" is itself informative
    and, at 0.01% of rows, not worth guessing about.
    """
    values = pd.to_numeric(age_days, errors="coerce")
    edges = [-np.inf] + list(config.AGE_BUCKET_EDGES[1:]) + [np.inf]
    bucketed = pd.cut(
        values,
        bins=edges,
        labels=config.AGE_BUCKET_LABELS,
        right=False,
    )
    return (
        bucketed.astype(object)
        .where(values.notna(), config.AGE_BUCKET_UNKNOWN)
        .fillna(config.AGE_BUCKET_UNKNOWN)
        .astype(str)
    )


def _as_days(predictions: np.ndarray) -> np.ndarray:
    """Clip to the physically possible range. A stay is never negative."""
    return np.clip(predictions, config.MIN_LOS_DAYS, config.MAX_LOS_DAYS)


# --------------------------------------------------------------------------
# Baseline 1 — the training median, for everybody
# --------------------------------------------------------------------------


class GlobalMedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the training median length of stay for every animal.

    This is the floor. A model that cannot beat it has learned nothing, and
    reporting it is what makes every other number mean something.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GlobalMedianBaseline":
        self.median_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _as_days(np.full(len(X), self.median_, dtype=float))


# --------------------------------------------------------------------------
# Baseline 2 — a lookup table
# --------------------------------------------------------------------------


class GroupMedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the training median for this animal_type x age_bucket cell.

    No learning, no gradients, no library — a group-by and a dictionary. It is
    genuinely hard to beat, which is exactly why it is here.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GroupMedianBaseline":
        frame = pd.DataFrame(
            {
                "animal_type": X["animal_type"].astype(str),
                "age_bucket": age_bucket(X["age_days"]),
                "y": np.asarray(y, dtype=float),
            }
        )
        self.table_ = frame.groupby(["animal_type", "age_bucket"])["y"].median()
        self.cell_counts_ = frame.groupby(
            ["animal_type", "age_bucket"]
        )["y"].size()
        self.fallback_ = float(np.median(frame["y"]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays(
            [X["animal_type"].astype(str), age_bucket(X["age_days"])]
        )
        # Unseen combinations fall back to the global median rather than NaN.
        predictions = self.table_.reindex(keys).to_numpy(dtype=float)
        predictions = np.where(
            np.isnan(predictions), self.fallback_, predictions
        )
        return _as_days(predictions)

    def unseen_cell_rate(self, X: pd.DataFrame) -> float:
        """Share of rows falling in a cell the training data never contained."""
        keys = pd.MultiIndex.from_arrays(
            [X["animal_type"].astype(str), age_bucket(X["age_days"])]
        )
        return float(self.table_.reindex(keys).isna().mean())


# --------------------------------------------------------------------------
# Model 3 — histogram gradient boosting with native categorical support
# --------------------------------------------------------------------------


class _CategoryEncoder:
    """Cap category cardinality using TRAINING frequencies only.

    HistGradientBoostingRegressor supports categorical features natively, so
    we do not one-hot anything — the breed column would explode into thousands
    of columns. But a native categorical feature may not exceed max_bins (255)
    levels, and primary_breed has more than that. So the top
    MAX_MODEL_CATEGORIES levels by training frequency are kept and the tail is
    folded into "Other".

    Categories are learned on train and then frozen: a breed that appears for
    the first time in the test period maps to "Other" rather than shifting the
    encoding, which would be leakage of test-set composition into the encoder.
    """

    OTHER = "Other"
    MISSING = "Missing"

    def fit(self, X: pd.DataFrame) -> "_CategoryEncoder":
        self.categories_: dict[str, list[str]] = {}
        for column in config.CATEGORICAL_FEATURES:
            values = X[column].astype("string").fillna(self.MISSING)
            counts = values.value_counts()
            keep = list(counts.head(config.MAX_MODEL_CATEGORIES).index)
            if self.OTHER not in keep:
                keep.append(self.OTHER)
            if self.MISSING not in keep:
                keep.append(self.MISSING)
            self.categories_[column] = sorted(set(keep))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        for column in config.NUMERIC_FEATURES:
            # HGB handles NaN natively; no imputation, no invented values.
            out[column] = pd.to_numeric(X[column], errors="coerce")

        for column in config.BOOLEAN_FEATURES:
            out[column] = pd.to_numeric(
                X[column].astype("object").map(
                    {True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}
                ),
                errors="coerce",
            )

        for column in config.CATEGORICAL_FEATURES:
            allowed = self.categories_[column]
            values = X[column].astype("string").fillna(self.MISSING)
            values = values.where(values.isin(allowed), self.OTHER)
            out[column] = pd.Categorical(values, categories=allowed)

        return out


class GradientBoostingModel(BaseEstimator, RegressorMixin):
    """HistGradientBoostingRegressor on log1p(days), inverted for reporting.

    Set `quantile` to fit a quantile model instead of the squared-error mean.
    Quantiles survive a monotonic transform unchanged, so a 10th-percentile
    fit in log space inverts to the 10th percentile in days — the interval is
    not distorted by the transform.
    """

    def __init__(
        self,
        quantile: float | None = None,
        mode: str = "log1p",
        **params,
    ):
        if mode not in {"log1p", "raw_absolute"}:
            raise ValueError(f"Unknown mode {mode!r}")
        self.quantile = quantile
        self.mode = mode
        self.params = params

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GradientBoostingModel":
        self.encoder_ = _CategoryEncoder().fit(X)
        encoded = self.encoder_.transform(X)
        y = np.asarray(y, dtype=float)

        base = (
            config.GBM_ABSOLUTE_PARAMS
            if self.mode == "raw_absolute"
            else config.GBM_PARAMS
        )
        settings = dict(base)
        settings.update(self.params)
        settings["random_state"] = config.RANDOM_STATE
        settings["categorical_features"] = "from_dtype"

        if self.quantile is not None:
            settings["loss"] = "quantile"
            settings["quantile"] = self.quantile

        self.estimator_ = HistGradientBoostingRegressor(**settings)

        if self.mode == "raw_absolute":
            # No transform, so nothing to invert and no Jensen gap to correct.
            # absolute_error optimises the conditional median directly, which
            # is also the quantity MAE and median-error actually reward.
            self.estimator_.fit(encoded, y)
        else:
            self.estimator_.fit(encoded, np.log1p(y))

        self.feature_names_ = list(encoded.columns)

        # Uncorrected until fit_smearing is called. 1.0 is a no-op.
        self.smearing_factor_ = 1.0
        return self

    def _log_prediction(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(self.encoder_.transform(X))

    def fit_smearing(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> "GradientBoostingModel":
        """Estimate Duan's smearing factor S = mean(exp(residuals)).

        See config.SMEARING_SOURCE for the derivation and the citation. Call
        this with the VALIDATION period, not the training period: a boosted
        model's in-sample residuals are shrunk by its own fit and would bias S
        downward.

        Only meaningful for the mean model. Quantiles are invariant under a
        monotonic transform, so a quantile fit needs no correction and calling
        this on one is refused rather than silently distorting the interval.
        """
        if self.quantile is not None:
            raise ValueError(
                "Smearing corrects a conditional MEAN. Quantiles survive the "
                "log transform unchanged and must not be rescaled."
            )
        if self.mode != "log1p":
            raise ValueError(
                "Smearing corrects a log retransformation. A raw-days model "
                "has no transform to invert and needs no correction."
            )
        residuals = np.log1p(np.asarray(y, dtype=float)) - self._log_prediction(X)
        self.smearing_factor_ = float(np.mean(np.exp(residuals)))
        self.smearing_residual_std_ = float(np.std(residuals))
        self.smearing_n_ = int(len(residuals))
        return self

    def predict_uncorrected(self, X: pd.DataFrame) -> np.ndarray:
        """Naive expm1 back-transform — biased low, kept for the comparison."""
        if self.mode == "raw_absolute":
            return self.predict(X)
        return _as_days(np.expm1(self._log_prediction(X)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediction in days.

        For a raw-days model this is the estimator's own output. For a log1p
        model it is exp(f) * S - 1, which is strictly increasing in f for any
        S > 0, so smearing changes the level and never the ranking.
        """
        if self.mode == "raw_absolute":
            return _as_days(self._log_prediction(X))
        raw = np.exp(self._log_prediction(X)) * self.smearing_factor_ - 1.0
        return _as_days(raw)


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------


class LongstayModels:
    """The three regressors plus the quantile trio, fitted together."""

    def __init__(self) -> None:
        self.baseline_global = GlobalMedianBaseline()
        self.baseline_group = GroupMedianBaseline()
        # The log1p fit. Its two back-transforms are the first two variants.
        self.gbm_log = GradientBoostingModel(mode="log1p")
        # The raw-days fit with absolute_error. The third variant.
        self.gbm_raw = GradientBoostingModel(mode="raw_absolute")
        self.quantile_models: dict[float, GradientBoostingModel] = {}

    @property
    def gbm(self) -> GradientBoostingModel:
        """The model the service and the headline table use."""
        return (
            self.gbm_raw
            if config.REGRESSION_OBJECTIVE == "raw_absolute"
            else self.gbm_log
        )

    def fit(
        self, train: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> "LongstayModels":
        X = build_feature_matrix(train)  # asserts the leakage whitelist
        y = train[config.TARGET].to_numpy(dtype=float)

        self.baseline_global.fit(X, y)
        self.baseline_group.fit(X, y)
        self.gbm_log.fit(X, y)
        self.gbm_raw.fit(X, y)

        # Quantile models stay on log1p: quantiles are invariant under a
        # monotonic transform, so there is no retransformation bias to avoid
        # here and the fitted interval is unaffected by the choice.
        for q in config.QUANTILES:
            self.quantile_models[q] = GradientBoostingModel(quantile=q).fit(X, y)

        # Duan smearing, estimated out of sample on the validation period.
        # Only meaningful for the log1p variant; kept so the three-way
        # comparison can still be reported.
        if validation is not None:
            self.gbm_log.fit_smearing(
                build_feature_matrix(validation),
                validation[config.TARGET].to_numpy(dtype=float),
            )

        self.trained_rows_ = len(train)
        self.trained_through_ = train["intake_datetime"].max()
        return self

    def predict_variants(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """All three regression variants, for the head-to-head table."""
        X = build_feature_matrix(frame)
        return {
            "log1p": self.gbm_log.predict_uncorrected(X),
            "log1p_smearing": self.gbm_log.predict(X),
            "raw_absolute": self.gbm_raw.predict(X),
        }

    def predict_all(
        self, frame: pd.DataFrame, corrected: bool = True
    ) -> dict[str, np.ndarray]:
        X = build_feature_matrix(frame)
        return {
            "baseline_global_median": self.baseline_global.predict(X),
            "baseline_group_median": self.baseline_group.predict(X),
            "hist_gradient_boosting": self.gbm.predict(X),
        }

    def predict_quantiles(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        X = build_feature_matrix(frame)
        return {q: m.predict(X) for q, m in self.quantile_models.items()}


# --------------------------------------------------------------------------
# The long-stay classifier — the primary model
# --------------------------------------------------------------------------


def long_stay_target(frame: pd.DataFrame) -> np.ndarray:
    """stay_90_plus — will this animal still be here in three months?"""
    return (
        frame[config.TARGET].to_numpy(dtype=float) >= config.LONG_STAY_DAYS
    ).astype(int)


class PositiveRateBaseline(BaseEstimator, ClassifierMixin):
    """Predict the training positive rate for every animal.

    The classifier equivalent of the global median: it knows the base rate and
    nothing else. Everything else is compared against it, and for a rare
    positive class it is a much stronger opponent than it looks — it cannot be
    beaten on accuracy at all.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PositiveRateBaseline":
        self.positive_rate_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.positive_rate_, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LongStayClassifier(BaseEstimator, ClassifierMixin):
    """HistGradientBoostingClassifier on stay_90_plus.

    Same features, same encoder, same leakage gate as the regressor. Raw
    boosted-tree scores are not reliable probabilities, so `fit_calibration`
    fits isotonic regression on the validation period and `predict_proba`
    applies it. Isotonic is monotone, so — like smearing — it changes the
    numbers and never the ranking.
    """

    def __init__(self, **params):
        self.params = params

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LongStayClassifier":
        self.encoder_ = _CategoryEncoder().fit(X)
        encoded = self.encoder_.transform(X)

        settings = dict(config.CLASSIFIER_PARAMS)
        settings.update(self.params)
        settings["random_state"] = config.RANDOM_STATE
        settings["categorical_features"] = "from_dtype"

        self.estimator_ = HistGradientBoostingClassifier(**settings)
        self.estimator_.fit(encoded, np.asarray(y, dtype=int))
        self.feature_names_ = list(encoded.columns)
        self.calibrator_ = None
        self.apply_calibration = config.USE_ISOTONIC_CALIBRATION
        self.train_positive_rate_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict_proba_uncalibrated(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(self.encoder_.transform(X))[:, 1]

    def fit_calibration(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> "LongStayClassifier":
        """Isotonic regression on the VALIDATION period only."""
        raw = self.predict_proba_uncalibrated(X)
        self.calibrator_ = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self.calibrator_.fit(raw, np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of a 90+ day stay.

        Returns the raw score unless isotonic calibration is both fitted AND
        switched on. It is off by default because measuring it showed it made
        calibration worse here, not better.
        """
        raw = self.predict_proba_uncalibrated(X)
        if self.calibrator_ is None or not getattr(
            self, "apply_calibration", False
        ):
            return raw
        return np.clip(self.calibrator_.transform(raw), 0.0, 1.0)

    def predict_proba_isotonic(self, X: pd.DataFrame) -> np.ndarray:
        """The isotonic-calibrated score, regardless of the switch.

        Only for the before/after comparison in the report.
        """
        raw = self.predict_proba_uncalibrated(X)
        if self.calibrator_ is None:
            return raw
        return np.clip(self.calibrator_.transform(raw), 0.0, 1.0)

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        threshold = (
            config.OPERATING_THRESHOLD if threshold is None else threshold
        )
        return (self.predict_proba(X) >= threshold).astype(int)


class LongStayClassifierBundle:
    """The classifier plus its baseline, fitted and calibrated together."""

    def __init__(self) -> None:
        self.baseline = PositiveRateBaseline()
        self.classifier = LongStayClassifier()

    def fit(
        self, train: pd.DataFrame, validation: pd.DataFrame
    ) -> "LongStayClassifierBundle":
        X_train = build_feature_matrix(train)
        y_train = long_stay_target(train)

        self.baseline.fit(X_train, y_train)
        self.classifier.fit(X_train, y_train)

        # Isotonic is fitted either way so `evaluate` can show both curves,
        # but only APPLIED when config says so. It was measured and it made
        # calibration slightly worse — see config.USE_ISOTONIC_CALIBRATION.
        self.classifier.fit_calibration(
            build_feature_matrix(validation), long_stay_target(validation)
        )
        self.classifier.apply_calibration = config.USE_ISOTONIC_CALIBRATION
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(build_feature_matrix(frame))

    def predict_proba_uncalibrated(self, frame: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba_uncalibrated(
            build_feature_matrix(frame)
        )

    def predict_proba_isotonic(self, frame: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba_isotonic(
            build_feature_matrix(frame)
        )

    def baseline_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.baseline.predict_proba(build_feature_matrix(frame))[:, 1]
