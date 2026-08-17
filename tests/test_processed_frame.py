"""Checks against the real processed frame, if it has been built.

These skip when data/processed/joined.parquet is absent, so the suite still
runs on a clean checkout.
"""

from __future__ import annotations

import pandas as pd
import pytest

from longstay import config
from longstay.clean import assert_no_leakage
from longstay.features import build_feature_matrix, temporal_split

pytestmark = pytest.mark.skipif(
    not config.JOINED_PARQUET.exists(),
    reason="run `python main.py clean` first",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_parquet(config.JOINED_PARQUET)


def test_no_leakage_in_the_real_frame(frame):
    assert_no_leakage(frame)


def test_real_frame_builds_a_clean_feature_matrix(frame):
    matrix = build_feature_matrix(frame)
    assert list(matrix.columns) == config.ALLOWED_FEATURES


def test_target_is_within_the_configured_bounds(frame):
    los = frame[config.TARGET]
    assert los.min() >= config.MIN_LOS_DAYS
    assert los.max() <= config.MAX_LOS_DAYS
    assert los.notna().all()


def test_outcome_never_precedes_intake(frame):
    assert (frame["outcome_datetime"] >= frame["intake_datetime"]).all()


def test_one_row_per_intake(frame):
    """A repeat visitor gets one row per stay, never a cross product."""
    duplicated = frame.duplicated(subset=["animal_id", "intake_datetime"])
    assert not duplicated.any(), f"{duplicated.sum()} duplicate intakes"


def test_no_outcome_settles_two_stays(frame):
    duplicated = frame.duplicated(subset=["animal_id", "outcome_datetime"])
    assert not duplicated.any(), f"{duplicated.sum()} reused outcomes"


def test_repeat_visitors_actually_exist(frame):
    """If this ever hits zero the join has collapsed and the tests above are
    passing vacuously."""
    counts = frame["animal_id"].value_counts()
    assert (counts > 1).sum() > 1000


def test_each_intake_got_the_earliest_available_outcome(frame):
    """Re-derive the pairing rule directly on the real frame.

    Within an animal, sorted by intake time, the outcomes must be strictly
    increasing too — an intake pointing backwards past a nearer outcome would
    break this.
    """
    ordered = frame.sort_values(["animal_id", "intake_datetime"])
    grouped = ordered.groupby("animal_id", sort=False)["outcome_datetime"]
    non_monotonic = grouped.apply(lambda s: not s.is_monotonic_increasing)
    assert not non_monotonic.any(), (
        f"{non_monotonic.sum()} animals have out-of-order outcome pairings"
    )


def test_a_later_stay_never_starts_before_the_previous_one_ends(frame):
    """Stays for one animal must not overlap."""
    ordered = frame.sort_values(["animal_id", "intake_datetime"])
    previous_outcome = ordered.groupby("animal_id", sort=False)[
        "outcome_datetime"
    ].shift(1)
    overlap = ordered["intake_datetime"] < previous_outcome
    assert not overlap.any(), f"{overlap.sum()} overlapping stays"


def test_temporal_split_leaves_every_period_populated(frame):
    """Guards against config split dates drifting past the end of the data."""
    train, validation, test = temporal_split(frame)
    assert len(train) > 0, "training period is empty — check TRAIN_END_DATE"
    assert len(validation) > 0, "validation period is empty — check VAL dates"
    assert len(test) > 0, "test period is empty — check TEST_START_DATE"


def test_split_is_strictly_temporal(frame):
    train, validation, test = temporal_split(frame)
    assert train["intake_datetime"].max() < validation["intake_datetime"].min()
    assert validation["intake_datetime"].max() < test["intake_datetime"].min()
