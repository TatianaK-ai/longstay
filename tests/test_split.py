"""The temporal split. CLAUDE.md principle 2: never random."""

from __future__ import annotations

import pandas as pd
import pytest

from longstay import config
from longstay.features import temporal_split


def frame_spanning_the_boundaries() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2019-01-01",                 # train
            config.TRAIN_END_DATE,        # train, boundary day
            config.VAL_START_DATE,        # validation, boundary day
            config.VAL_END_DATE,          # validation, boundary day
            config.TEST_START_DATE,       # test, boundary day
            "2025-01-01",                 # test
        ]
    )
    return pd.DataFrame(
        {"intake_datetime": dates, config.TARGET: [1, 2, 3, 4, 5, 6]}
    )


def test_split_is_on_intake_date():
    train, validation, test = temporal_split(frame_spanning_the_boundaries())
    assert len(train) == 2
    assert len(validation) == 2
    assert len(test) == 2


def test_periods_are_in_strict_time_order():
    """The whole point: each period is strictly later than the last."""
    train, validation, test = temporal_split(frame_spanning_the_boundaries())
    assert train["intake_datetime"].max() < validation["intake_datetime"].min()
    assert validation["intake_datetime"].max() < test["intake_datetime"].min()


def test_boundary_days_are_inclusive():
    train, validation, test = temporal_split(frame_spanning_the_boundaries())
    assert pd.Timestamp(config.TRAIN_END_DATE) in set(train["intake_datetime"])
    assert pd.Timestamp(config.VAL_END_DATE) in set(
        validation["intake_datetime"]
    )
    assert pd.Timestamp(config.TEST_START_DATE) in set(test["intake_datetime"])


def test_an_intake_late_on_the_boundary_day_still_lands_in_that_period():
    """Boundary dates cover the whole day, not just midnight.

    This is the bug that made `stats` and `temporal_split` disagree by 24 rows.
    """
    frame = pd.DataFrame(
        {
            "intake_datetime": pd.to_datetime(
                [f"{config.TRAIN_END_DATE} 23:59:00",
                 f"{config.VAL_END_DATE} 23:59:00"]
            ),
            config.TARGET: [1, 2],
        }
    )
    train, validation, test = temporal_split(frame)
    assert len(train) == 1
    assert len(validation) == 1
    assert len(test) == 0


def test_the_three_periods_do_not_overlap():
    train, validation, test = temporal_split(frame_spanning_the_boundaries())
    assert not set(train.index) & set(validation.index)
    assert not set(validation.index) & set(test.index)
    assert not set(train.index) & set(test.index)


def test_split_is_exhaustive_for_contiguous_dates():
    frame = frame_spanning_the_boundaries()
    train, validation, test = temporal_split(frame)
    assert len(train) + len(validation) + len(test) == len(frame)


def test_config_dates_are_ordered():
    assert (
        pd.Timestamp(config.TRAIN_END_DATE)
        < pd.Timestamp(config.VAL_START_DATE)
        <= pd.Timestamp(config.VAL_END_DATE)
        < pd.Timestamp(config.TEST_START_DATE)
    )


def test_misordered_config_dates_are_rejected(monkeypatch):
    """A split that leaks the future must fail loudly, not silently."""
    monkeypatch.setattr(config, "VAL_START_DATE", "2000-01-01")
    with pytest.raises(ValueError, match="strictly ordered"):
        temporal_split(frame_spanning_the_boundaries())
