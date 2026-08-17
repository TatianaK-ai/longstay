"""The leakage boundary.

CLAUDE.md principle 3: only features knowable AT INTAKE may reach the model.
These tests fail if a forbidden column gets through.
"""

from __future__ import annotations

import pandas as pd
import pytest

from longstay import config
from longstay.clean import assert_no_leakage
from longstay.features import assert_only_allowed_features, build_feature_matrix


def minimal_frame(**extra) -> pd.DataFrame:
    """A one-row frame holding exactly the allowed features, plus overrides."""
    row = {
        "age_days": 365.0,
        "animal_type": "Dog",
        "sex": "Male",
        "sterilization_status": "Fixed",
        "intake_type": "Stray",
        "intake_condition": "Normal",
        "primary_breed": "Labrador Retriever",
        "primary_color": "Black",
        "intake_month": 6,
        "intake_day_of_week": 2,
        "intake_season": "Summer",
        "is_mix": True,
        "is_black": True,
    }
    row.update(extra)
    return pd.DataFrame([row])


def test_clean_frame_passes():
    assert_only_allowed_features(minimal_frame())


@pytest.mark.parametrize("forbidden", config.FORBIDDEN_COLUMNS)
def test_every_forbidden_column_is_rejected(forbidden):
    """This is the test that must fail if someone adds leakage upstream."""
    frame = minimal_frame(**{forbidden: "Adoption"})
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_only_allowed_features(frame)


def test_outcome_type_specifically_is_rejected():
    """The single most tempting leak: the label everyone else predicts."""
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_only_allowed_features(minimal_frame(outcome_type="Adoption"))


def test_outcome_datetime_is_not_a_feature():
    """It computes the target. It is never an input."""
    assert "outcome_datetime" not in config.ALLOWED_FEATURES
    frame = minimal_frame(outcome_datetime=pd.Timestamp("2020-01-01"))
    with pytest.raises(AssertionError):
        assert_only_allowed_features(frame)


def test_target_itself_cannot_be_a_feature():
    assert config.TARGET not in config.ALLOWED_FEATURES
    with pytest.raises(AssertionError):
        assert_only_allowed_features(minimal_frame(**{config.TARGET: 12.0}))


def test_unknown_column_is_rejected_even_if_harmless():
    """Whitelist, not blacklist: unrecognised columns do not get a pass."""
    with pytest.raises(AssertionError, match="ALLOWED_FEATURES"):
        assert_only_allowed_features(minimal_frame(kennel_number="B12"))


def test_missing_expected_feature_is_rejected():
    frame = minimal_frame().drop(columns=["is_mix"])
    with pytest.raises(AssertionError, match="missing"):
        assert_only_allowed_features(frame)


# ------------------------------------------------------- timing leakage


def test_has_name_is_not_a_model_feature():
    """It passed the provenance check and leaked anyway. See
    config.TEMPORAL_LEAKAGE_NOTES and tools/audit_timing.py."""
    assert "has_name" not in config.ALLOWED_FEATURES
    assert "has_name" in config.EXCLUDED_FOR_TIMING


def test_has_name_reaching_the_model_is_rejected():
    """Re-adding it must fail loudly rather than quietly improving PR-AUC."""
    with pytest.raises(AssertionError, match="ALLOWED_FEATURES"):
        assert_only_allowed_features(minimal_frame(has_name=True))


def test_excluded_features_are_still_kept_in_the_processed_frame():
    """Removed from the model, retained for auditing."""
    for column in config.EXCLUDED_FOR_TIMING:
        assert column in config.NON_FEATURE_COLUMNS


def test_every_excluded_feature_has_a_documented_reason():
    """No silent deletions: if it was dropped, the reasoning is on record."""
    documented = " ".join(config.TEMPORAL_LEAKAGE_NOTES.keys())
    for column in config.EXCLUDED_FOR_TIMING:
        assert column in documented, f"{column} dropped without a note"


def test_timing_notes_cover_every_modelled_feature():
    """Every feature carries an explicit timing verdict, including the clean
    ones — 'we checked and it is fine' is a result worth recording."""
    documented = " ".join(config.TEMPORAL_LEAKAGE_NOTES.keys())
    missing = [f for f in config.ALLOWED_FEATURES if f not in documented]
    assert not missing, f"no timing verdict recorded for: {missing}"


def test_timing_notes_state_evidence_not_just_a_verdict():
    for name, note in config.TEMPORAL_LEAKAGE_NOTES.items():
        assert "status" in note, name
        assert "evidence" in note, name
        assert len(note["evidence"]) > 60, f"{name}: evidence is too thin"


def test_build_feature_matrix_strips_non_features():
    """The gate selects the whitelist rather than trusting its input."""
    frame = minimal_frame()
    frame["outcome_type"] = "Adoption"
    frame["animal_id"] = "A1"
    frame[config.TARGET] = 12.0

    matrix = build_feature_matrix(frame)

    assert list(matrix.columns) == config.ALLOWED_FEATURES
    assert "outcome_type" not in matrix.columns


def test_forbidden_and_allowed_lists_are_disjoint():
    overlap = set(config.FORBIDDEN_COLUMNS) & set(config.ALLOWED_FEATURES)
    assert not overlap, f"config contradicts itself: {overlap}"


def test_outcomes_table_contributes_only_two_columns():
    """The structural guarantee behind the whole leakage story.

    _prepare_outcomes is the only route by which outcome data enters the
    pipeline. If it yields exactly animal_id and outcome_datetime, then no
    other outcome column can reach the model regardless of what happens
    downstream. animal_type, breed and color exist in BOTH tables — this is
    what stops the outcome-side copies from being picked up.
    """
    from longstay.clean import DropLog, _prepare_outcomes

    raw = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "datetime": ["2020-01-05T12:00:00.000"],
            "outcome_type": ["Adoption"],
            "outcome_subtype": ["Foster"],
            "sex_upon_outcome": ["Neutered Male"],
            "age_upon_outcome": ["2 years"],
            "date_of_birth": ["2018-01-01"],
            "animal_type": ["Dog"],
            "breed": ["Beagle"],
            "color": ["Black"],
            "name": ["Rex"],
            "monthyear": ["01-2020"],
        }
    )

    out = _prepare_outcomes(raw, DropLog())

    assert sorted(out.columns) == ["animal_id", "outcome_datetime"]


def test_processed_frame_check_rejects_outcome_columns():
    frame = minimal_frame()
    for column in config.NON_FEATURE_COLUMNS:
        frame[column] = "x"
    assert_no_leakage(frame)  # baseline: this shape is fine

    frame["sex_upon_outcome"] = "Neutered Male"
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(frame)
