"""Socrata timestamp normalisation.

The feed mixes naive local wall-clock timestamps with a minority that carry an
explicit -05:00 offset. Both mean Austin local time. If we converted the
offset-bearing ones to UTC, only that subgroup would shift five hours and
their length of stay would silently disagree with everyone else's.
"""

from __future__ import annotations

import pandas as pd

from longstay.fetch import count_offset_timestamps, parse_socrata_datetime


def test_naive_timestamps_parse_unchanged():
    out = parse_socrata_datetime(pd.Series(["2013-10-01T07:51:00.000"]))
    assert out.iloc[0] == pd.Timestamp("2013-10-01 07:51:00")


def test_offset_is_stripped_not_converted():
    """The real shape found in outcomes.csv: midnight with a -05:00 offset."""
    out = parse_socrata_datetime(pd.Series(["2013-12-02T00:00:00-05:00"]))
    assert out.iloc[0] == pd.Timestamp("2013-12-02 00:00:00")
    assert out.dt.tz is None


def test_mixed_offsets_and_naive_in_one_column():
    """This exact mix crashed a plain to_datetime call."""
    values = pd.Series(
        [
            "2013-10-01T07:51:00.000",
            "2013-12-02T00:00:00-05:00",
            "2014-02-22T00:00:00-05:00",
        ]
    )
    out = parse_socrata_datetime(values)
    assert out.notna().all()
    assert out.dt.tz is None
    assert list(out) == [
        pd.Timestamp("2013-10-01 07:51:00"),
        pd.Timestamp("2013-12-02 00:00:00"),
        pd.Timestamp("2014-02-22 00:00:00"),
    ]


def test_stay_length_is_consistent_across_both_formats():
    """A ten-day stay is ten days regardless of how the feed serialised it."""
    naive = parse_socrata_datetime(
        pd.Series(["2014-01-01T00:00:00.000", "2014-01-11T00:00:00.000"])
    )
    offset = parse_socrata_datetime(
        pd.Series(["2014-01-01T00:00:00-05:00", "2014-01-11T00:00:00-05:00"])
    )
    assert (naive.iloc[1] - naive.iloc[0]) == (offset.iloc[1] - offset.iloc[0])


def test_z_suffix_is_handled():
    out = parse_socrata_datetime(pd.Series(["2020-05-01T12:00:00Z"]))
    assert out.iloc[0] == pd.Timestamp("2020-05-01 12:00:00")


def test_malformed_becomes_nat():
    out = parse_socrata_datetime(pd.Series(["not a date", None, ""]))
    assert out.isna().all()


def test_count_offset_timestamps():
    values = pd.Series(
        ["2013-10-01T07:51:00.000", "2013-12-02T00:00:00-05:00", None]
    )
    assert count_offset_timestamps(values) == 1
