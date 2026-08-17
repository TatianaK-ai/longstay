"""The prediction service and the API contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longstay import config
from longstay.model import LongStayClassifierBundle, LongstayModels
from longstay.service import TriageService, intake_to_frame, risk_band
from tests.test_classifier import labelled_frame


# ---------------------------------------------------------------- risk band


def test_risk_band_comes_from_probability_not_days():
    assert risk_band(0.9) == "High"
    assert risk_band(config.RISK_BAND_HIGH) == "High"
    assert risk_band(config.OPERATING_THRESHOLD) == "Elevated"
    assert risk_band(config.OPERATING_THRESHOLD - 0.001) == "Standard"
    assert risk_band(config.RISK_BAND_FAST) == "Standard"
    assert risk_band(config.RISK_BAND_FAST - 0.001) == "Fast"
    assert risk_band(0.0) == "Fast"


def test_risk_bands_are_ordered():
    assert (
        config.RISK_BAND_FAST
        < config.RISK_BAND_ELEVATED
        < config.RISK_BAND_HIGH
    )


def test_only_the_elevated_boundary_is_operational():
    """The Fast cut is a display split; it must not move the flag threshold."""
    assert config.RISK_BAND_ELEVATED == config.OPERATING_THRESHOLD
    assert config.RISK_BAND_FAST < config.OPERATING_THRESHOLD


# ------------------------------------------------------------ intake parsing


def test_intake_record_becomes_engineered_features():
    frame = intake_to_frame(
        {
            "animal_type": "Dog",
            "breed": "Pit Bull Mix",
            "color": "Black/White",
            "age_upon_intake": "2 years",
            "sex_upon_intake": "Intact Male",
            "intake_type": "Stray",
            "intake_condition": "Normal",
            "name": "Rex",
            "intake_datetime": "2025-07-04T13:00:00",
        }
    )
    assert frame["age_days"].iloc[0] == pytest.approx(730.5)
    assert frame["primary_breed"].iloc[0] == "Pit Bull"
    assert bool(frame["is_mix"].iloc[0]) is True
    assert frame["primary_color"].iloc[0] == "Black"
    assert bool(frame["is_black"].iloc[0]) is True
    assert frame["sex"].iloc[0] == "Male"
    assert frame["sterilization_status"].iloc[0] == "Intact"
    assert bool(frame["has_name"].iloc[0]) is True
    assert frame["intake_season"].iloc[0] == "Summer"


def test_missing_fields_do_not_crash():
    frame = intake_to_frame({"animal_type": "Cat"})
    assert len(frame) == 1
    assert bool(frame["has_name"].iloc[0]) is False


def test_api_uses_the_same_feature_engineering_as_training():
    """If these drift, the model is scored on different features than it saw."""
    frame = intake_to_frame({"animal_type": "Dog", "age_upon_intake": "1 year"})
    missing = [c for c in config.ALLOWED_FEATURES if c not in frame.columns]
    assert not missing


# ------------------------------------------------------------------ service


@pytest.fixture(scope="module")
def service():
    frame = labelled_frame(1200)
    train, validation = frame.iloc[:900], frame.iloc[900:]

    models = LongstayModels().fit(train, validation)
    classifier = LongStayClassifierBundle().fit(train, validation)

    from longstay.features import build_feature_matrix

    built = TriageService(models=models, classifier=classifier)
    built.set_reference(build_feature_matrix(train))
    return built


def test_score_returns_both_numbers(service):
    result = service.score(
        {
            "animal_type": "Cat",
            "breed": "Domestic Shorthair",
            "color": "Black",
            "age_upon_intake": "5 years",
            "sex_upon_intake": "Intact Female",
            "intake_type": "Stray",
            "intake_condition": "Normal",
        }
    )

    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_band"] in {"Fast", "Standard", "Elevated", "High"}
    assert result["predicted_days"]["median"] >= 0
    assert "p10" in result["predicted_days"]
    assert "p90" in result["predicted_days"]


def test_probability_is_declared_the_primary_metric(service):
    """The day estimate must not be mistakable for the headline."""
    result = service.score({"animal_type": "Dog", "age_upon_intake": "2 years"})
    assert result["primary_metric"] == "risk_probability"
    assert result["predicted_days"]["is_primary"] is False


def test_day_estimate_always_carries_its_caveat(service):
    """A day number without the R² warning must never leave the service."""
    result = service.score({"animal_type": "Dog", "age_upon_intake": "2 years"})
    caveat = result["predicted_days"]["caveat"]
    assert set(caveat) == {"text", "r2"}
    assert caveat["text"]


def test_caveat_text_comes_from_metrics_not_a_literal():
    """The wording and the number both travel from metrics.json."""
    built = TriageService(
        models=None, classifier=None,
        _metrics={"day_estimate_caveat": {"text": "measured wording", "r2": -0.5}},
    )
    assert built.day_caveat()["text"] == "measured wording"
    assert built.day_caveat()["r2"] == -0.5


def test_missing_metrics_still_warns_rather_than_going_silent():
    built = TriageService(models=None, classifier=None, _metrics={})
    caveat = built.day_caveat()
    assert caveat["r2"] is None
    assert "Indicative only" in caveat["text"]


def test_risk_band_matches_the_probability(service):
    result = service.score({"animal_type": "Dog", "age_upon_intake": "3 years"})
    assert result["risk_band"] == risk_band(result["risk_probability"])


def test_interval_brackets_the_median(service):
    result = service.score({"animal_type": "Dog", "age_upon_intake": "3 years"})
    days = result["predicted_days"]
    assert days["p10"] <= days["median"] <= days["p90"]


def test_drivers_are_in_percentage_points(service):
    result = service.score(
        {
            "animal_type": "Cat",
            "age_upon_intake": "6 years",
            "breed": "Domestic Shorthair",
            "intake_type": "Stray",
        }
    )
    for driver in result["drivers"]:
        assert "effect_percentage_points" in driver
        assert -100.0 <= driver["effect_percentage_points"] <= 100.0
        assert driver["feature"] in config.ALLOWED_FEATURES


def test_drivers_are_sorted_by_magnitude(service):
    result = service.score({"animal_type": "Cat", "age_upon_intake": "6 years"})
    effects = [abs(d["effect_percentage_points"]) for d in result["drivers"]]
    assert effects == sorted(effects, reverse=True)


def test_response_always_carries_the_caveat(service):
    result = service.score({"animal_type": "Dog"})
    assert "cannot see" in result["caveat"]
    assert "temperament" in result["caveat"]


def test_every_response_carries_a_reliability_block(service):
    """Not optional, and not conditional on the species being weak."""
    for species in ("Dog", "Cat"):
        result = service.score({"animal_type": species})
        assert "model_reliability" in result
        assert result["model_reliability"]["species"] == species
        assert result["model_reliability"]["message"]


def test_unmeasured_species_is_declared_unverified(service):
    """The fixture has no reliability file, so it must not claim confidence."""
    result = service.score({"animal_type": "Cat"})
    assert result["model_reliability"]["reliable"] is None


def test_service_never_sees_a_forbidden_column(service):
    """The API path goes through the same leakage gate as training."""
    frame = intake_to_frame({"animal_type": "Dog", "age_upon_intake": "1 year"})
    frame["outcome_type"] = "Adoption"
    results = service.score_frame(frame)
    assert len(results) == 1
