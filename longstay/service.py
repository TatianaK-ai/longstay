"""Prediction service — what the API and the triage page consume.

Returns both numbers, because staff need both: the probability sets the
priority, the day estimate sets the order of magnitude. The risk band comes
from the PROBABILITY, never from the day estimate — the day estimate is the
weaker of the two and must not drive triage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import config
from .clean import engineer_features
from .features import build_feature_matrix


def risk_band(probability: float) -> str:
    """Band from probability. Thresholds live in config, not here.

    Four display bands over three meaningful cut points. Only the Elevated
    boundary is operational — it is the cost-derived operating threshold.
    The Fast boundary exists so "not flagged" is not one undifferentiated
    bucket holding both a 9% animal and a 0.4% one.
    """
    if probability >= config.RISK_BAND_HIGH:
        return "High"
    if probability >= config.RISK_BAND_ELEVATED:
        return "Elevated"
    if probability >= config.RISK_BAND_FAST:
        return "Standard"
    return "Fast"


def intake_to_frame(record: dict) -> pd.DataFrame:
    """Turn one intake-day record into a single-row feature frame.

    Accepts the raw shape a shelter system would send — the same field names
    as the intakes table — and runs it through the same feature engineering
    the training data went through. Deriving features here rather than asking
    the caller for `age_days` means the API cannot drift from the pipeline.
    """
    raw = pd.DataFrame(
        [
            {
                "intake_datetime": pd.to_datetime(
                    record.get("intake_datetime") or pd.Timestamp.now()
                ),
                "intake_name_raw": record.get("name"),
                "animal_type": record.get("animal_type"),
                "sex_upon_intake": record.get("sex_upon_intake"),
                "age_upon_intake": record.get("age_upon_intake"),
                "intake_type": record.get("intake_type"),
                "intake_condition": record.get("intake_condition"),
                "breed": record.get("breed"),
                "color": record.get("color"),
            }
        ]
    )
    return engineer_features(raw)


@dataclass
class TriageService:
    """Loads the fitted models and scores intakes."""

    models: object
    classifier: object
    _reference: dict = field(default_factory=dict)
    _reliability: dict = field(default_factory=dict)
    _metrics: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "TriageService":
        reference_path = config.REFERENCE_ROW_PATH
        for path in (
            config.MODEL_PATH, config.CLASSIFIER_MODEL_PATH, reference_path
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run `python main.py train` first."
                )

        # Measured per-species reliability, written by `evaluate`. Absent
        # until an evaluation has been run, in which case we say we do not
        # know rather than implying the model is fine.
        reliability = {}
        if config.SPECIES_RELIABILITY_PATH.exists():
            reliability = json.loads(
                config.SPECIES_RELIABILITY_PATH.read_text(encoding="utf-8")
            )

        metrics = {}
        if config.METRICS_PATH.exists():
            metrics = json.loads(
                config.METRICS_PATH.read_text(encoding="utf-8")
            )

        return cls(
            models=joblib.load(config.MODEL_PATH),
            classifier=joblib.load(config.CLASSIFIER_MODEL_PATH),
            _reference=joblib.load(reference_path),
            _reliability=reliability,
            _metrics=metrics,
        )

    def day_caveat(self) -> dict:
        """The day-estimate caveat, read from metrics.json — never hardcoded.

        If no evaluation has been run we say the estimate is unverified rather
        than shipping a number with no warning attached.
        """
        caveat = self._metrics.get("day_estimate_caveat")
        if caveat:
            return caveat
        return {
            "text": "Indicative only. Not yet verified against a test set.",
            "r2": None,
        }

    def reliability_for(self, animal_type: object) -> dict:
        """Honest statement of how well the model works for this species.

        A model that silently emits a useless number is worse than one that
        says "I do not know here". Every response carries this; the UI makes
        it prominent when `reliable` is false.
        """
        species = str(animal_type) if animal_type is not None else "unknown"
        block = self._reliability.get(species)

        if block is None:
            return {
                "species": species,
                "reliable": None,
                "message": (
                    f"Predictive performance for {species} has not been "
                    "measured separately. Treat this score as unverified."
                ),
            }

        if block["reliable"]:
            return {
                "species": species,
                "reliable": True,
                "pr_auc": block["pr_auc"],
                "base_rate": block["base_rate"],
                "lift": block["lift"],
                "message": (
                    f"For {species.lower()}s the model finds long stays "
                    f"{block['lift']:.1f}x better than chance "
                    f"(PR-AUC {block['pr_auc']:.3f} against a base rate of "
                    f"{block['base_rate']:.1%})."
                ),
            }

        return {
            "species": species,
            "reliable": False,
            "pr_auc": block["pr_auc"],
            "base_rate": block["base_rate"],
            "lift": block["lift"],
            "message": (
                f"LOW CONFIDENCE FOR {species.upper()}S. On the held-out test "
                f"set this model scored PR-AUC {block['pr_auc']:.3f} for "
                f"{species.lower()}s against a base rate of "
                f"{block['base_rate']:.1%} — only {block['lift']:.2f}x better "
                f"than flagging at random, below the {block['min_lift']:.1f}x "
                "we treat as usable. The score below is close to a guess. Use "
                "your own judgement about this animal."
            ),
        }

    # ------------------------------------------------------------------ score

    def score(self, record: dict) -> dict:
        frame = intake_to_frame(record)
        return self.score_frame(frame)[0]

    def score_frame(self, frame: pd.DataFrame) -> list[dict]:
        """Score a frame of already-engineered intakes."""
        X = build_feature_matrix(frame)

        probability = self.classifier.classifier.predict_proba(X)
        days_median = self.models.quantile_models[0.5].predict(X)
        days_low = self.models.quantile_models[0.1].predict(X)
        days_high = self.models.quantile_models[0.9].predict(X)
        days_mean = self.models.gbm.predict(X)

        results = []
        for i in range(len(frame)):
            results.append(
                {
                    # PRIMARY — what triage is ordered by, and the only
                    # number on this response that earned a headline.
                    "risk_probability": float(probability[i]),
                    "risk_band": risk_band(float(probability[i])),
                    "primary_metric": "risk_probability",
                    # SECONDARY — order of magnitude, not a promise. The
                    # caveat travels with the number so no client can show
                    # one without the other.
                    "predicted_days": {
                        "median": float(days_median[i]),
                        "p10": float(days_low[i]),
                        "p90": float(days_high[i]),
                        "point_estimate": float(days_mean[i]),
                        "caveat": self.day_caveat(),
                        "is_primary": False,
                    },
                    "drivers": self.drivers(frame.iloc[[i]]),
                    # Prominent, not a footnote. See reliability_for().
                    "model_reliability": self.reliability_for(
                        frame.iloc[i]["animal_type"]
                    ),
                    "caveat": (
                        "Computed from paperwork recorded at intake. The model "
                        "cannot see: " + "; ".join(config.UNOBSERVED_FACTORS)
                    ),
                }
            )
        return results

    # ---------------------------------------------------------------- drivers

    def reference_row(self) -> pd.DataFrame:
        """The 'typical animal' each driver is measured against.

        Modal category and median age from the TRAINING period. Held on the
        service so it is computed once, and taken from training data so the
        explanation does not shift as new intakes arrive.
        """
        if not self._reference:
            raise RuntimeError(
                "Reference row not set. Build the service with "
                "TriageService.load() after `train` has stored it."
            )
        return pd.DataFrame([self._reference])

    def set_reference(self, train_features: pd.DataFrame) -> "TriageService":
        reference = {}
        for column in config.ALLOWED_FEATURES:
            values = train_features[column]
            if column in config.NUMERIC_FEATURES:
                reference[column] = float(values.median())
            else:
                mode = values.mode(dropna=True)
                reference[column] = mode.iloc[0] if len(mode) else None
        self._reference = reference
        return self

    def drivers(self, row: pd.DataFrame) -> list[dict]:
        """Per-feature contribution, in PERCENTAGE POINTS of risk probability.

        For each feature we hold everything else fixed and swap that one
        feature to its training-population reference value, then measure how
        far the probability moves. Positive means this animal's actual value
        pushes risk UP relative to a typical intake.

        This is a one-at-a-time counterfactual, not a Shapley value: it does
        not split credit for interactions, and the parts do not have to sum to
        the total. Stated plainly here so nobody reads it as something stronger.
        """
        if not self._reference:
            return []

        X = build_feature_matrix(row)
        actual = float(self.classifier.classifier.predict_proba(X)[0])

        contributions = []
        for column in config.ALLOWED_FEATURES:
            counterfactual = X.copy()
            counterfactual[column] = self._reference[column]
            swapped = float(
                self.classifier.classifier.predict_proba(counterfactual)[0]
            )
            delta = (actual - swapped) * 100.0
            if abs(delta) < 0.05:  # below a tenth of a point, not worth showing
                continue
            contributions.append(
                {
                    "feature": column,
                    "value": _display(row.iloc[0][column]),
                    "compared_to": _display(self._reference[column]),
                    "effect_percentage_points": round(delta, 2),
                }
            )

        contributions.sort(
            key=lambda item: abs(item["effect_percentage_points"]), reverse=True
        )
        return contributions[:6]


def _display(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):g}"
    return str(value)
