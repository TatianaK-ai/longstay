"""The intake -> next-outcome pairing.

Getting this wrong silently corrupts every downstream number, which is exactly
why it gets the most tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from longstay.clean import DropLog, pair_intakes_to_outcomes


def make_intakes(rows):
    return pd.DataFrame(
        [{"animal_id": a, "intake_datetime": pd.Timestamp(t)} for a, t in rows]
    )


def make_outcomes(rows):
    return pd.DataFrame(
        [{"animal_id": a, "outcome_datetime": pd.Timestamp(t)} for a, t in rows]
    )


def los_days(paired):
    return (
        paired["outcome_datetime"] - paired["intake_datetime"]
    ).dt.total_seconds() / 86400.0


def test_single_stay():
    paired = pair_intakes_to_outcomes(
        make_intakes([("A1", "2020-01-01")]),
        make_outcomes([("A1", "2020-01-11")]),
    )
    assert len(paired) == 1
    assert los_days(paired).iloc[0] == 10.0


def test_repeat_visitor_pairs_each_intake_with_its_own_outcome():
    """The core case. A1 visits twice; each intake takes the NEXT outcome."""
    intakes = make_intakes([("A1", "2020-01-01"), ("A1", "2020-06-01")])
    outcomes = make_outcomes([("A1", "2020-01-05"), ("A1", "2020-06-03")])

    paired = pair_intakes_to_outcomes(intakes, outcomes).sort_values(
        "intake_datetime"
    )

    assert len(paired) == 2
    assert list(los_days(paired)) == [4.0, 2.0]


def test_does_not_pair_intake_with_an_earlier_outcome():
    """An outcome before the intake belongs to a previous stay, not this one."""
    intakes = make_intakes([("A1", "2020-06-01")])
    outcomes = make_outcomes([("A1", "2020-01-05"), ("A1", "2020-06-10")])

    paired = pair_intakes_to_outcomes(intakes, outcomes)

    assert len(paired) == 1
    assert los_days(paired).iloc[0] == 9.0


def test_no_cross_product_for_repeat_visitors():
    """Three intakes, three outcomes, one animal -> three rows, not nine."""
    intakes = make_intakes(
        [("A1", "2020-01-01"), ("A1", "2020-03-01"), ("A1", "2020-05-01")]
    )
    outcomes = make_outcomes(
        [("A1", "2020-01-02"), ("A1", "2020-03-05"), ("A1", "2020-05-10")]
    )

    paired = pair_intakes_to_outcomes(intakes, outcomes).sort_values(
        "intake_datetime"
    )

    assert len(paired) == 3
    assert list(los_days(paired)) == [1.0, 4.0, 9.0]


def test_animals_never_cross_contaminate():
    """A2's outcome must never settle A1's intake."""
    intakes = make_intakes([("A1", "2020-01-01"), ("A2", "2020-01-02")])
    outcomes = make_outcomes([("A2", "2020-01-03"), ("A1", "2020-01-20")])

    paired = pair_intakes_to_outcomes(intakes, outcomes).set_index("animal_id")

    assert los_days(paired.reset_index()).tolist()
    assert (
        paired.loc["A1", "outcome_datetime"] == pd.Timestamp("2020-01-20")
    )
    assert (
        paired.loc["A2", "outcome_datetime"] == pd.Timestamp("2020-01-03")
    )


def test_two_intakes_one_outcome_keeps_the_later_intake():
    """A missing outcome record must not invent a stay spanning both visits."""
    intakes = make_intakes([("A1", "2020-01-01"), ("A1", "2020-02-01")])
    outcomes = make_outcomes([("A1", "2020-02-10")])

    drops = DropLog()
    paired = pair_intakes_to_outcomes(intakes, outcomes, drops)

    assert len(paired) == 1
    assert paired["intake_datetime"].iloc[0] == pd.Timestamp("2020-02-01")
    assert los_days(paired).iloc[0] == 9.0

    claimed = [s for s in drops.steps if s[0] == "outcome already claimed"]
    assert claimed and claimed[0][1] == 1


def test_intake_with_no_outcome_is_dropped_and_counted():
    """Still in the shelter today -> no target, so no row."""
    intakes = make_intakes([("A1", "2020-01-01"), ("A2", "2020-01-01")])
    outcomes = make_outcomes([("A1", "2020-01-05")])

    drops = DropLog()
    paired = pair_intakes_to_outcomes(intakes, outcomes, drops)

    assert list(paired["animal_id"]) == ["A1"]
    unmatched = [s for s in drops.steps if s[0] == "no matching outcome"]
    assert unmatched and unmatched[0][1] == 1


def test_same_day_outcome_is_a_valid_zero_day_stay():
    """Exact timestamp matches are real events, not errors."""
    paired = pair_intakes_to_outcomes(
        make_intakes([("A1", "2020-01-01 09:00")]),
        make_outcomes([("A1", "2020-01-01 09:00")]),
    )
    assert len(paired) == 1
    assert los_days(paired).iloc[0] == 0.0


def test_outcome_with_no_preceding_intake_is_ignored():
    """An outcome for an animal we never saw admitted adds no row."""
    intakes = make_intakes([("A1", "2020-01-01")])
    outcomes = make_outcomes([("A9", "2020-01-05"), ("A1", "2020-01-04")])

    paired = pair_intakes_to_outcomes(intakes, outcomes)

    assert list(paired["animal_id"]) == ["A1"]
    assert los_days(paired).iloc[0] == 3.0


def test_input_row_order_does_not_change_the_result():
    """Unsorted input must produce the same pairing as sorted input."""
    intakes = make_intakes(
        [("A1", "2020-05-01"), ("A2", "2020-01-01"), ("A1", "2020-01-01")]
    )
    outcomes = make_outcomes(
        [("A1", "2020-05-10"), ("A1", "2020-01-03"), ("A2", "2020-01-02")]
    )

    shuffled = pair_intakes_to_outcomes(
        intakes.iloc[::-1].reset_index(drop=True),
        outcomes.iloc[::-1].reset_index(drop=True),
    ).sort_values(["animal_id", "intake_datetime"]).reset_index(drop=True)

    ordered = pair_intakes_to_outcomes(intakes, outcomes).sort_values(
        ["animal_id", "intake_datetime"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(shuffled, ordered)


def test_extra_intake_columns_survive_the_join():
    intakes = make_intakes([("A1", "2020-01-01")])
    intakes["animal_type"] = "Dog"
    paired = pair_intakes_to_outcomes(intakes, make_outcomes([("A1", "2020-01-02")]))
    assert paired["animal_type"].iloc[0] == "Dog"


@pytest.mark.parametrize("bad", [None, pd.NaT])
def test_rows_with_missing_keys_are_excluded(bad):
    intakes = pd.DataFrame(
        [
            {"animal_id": "A1", "intake_datetime": pd.Timestamp("2020-01-01")},
            {"animal_id": "A2", "intake_datetime": bad},
        ]
    )
    outcomes = make_outcomes([("A1", "2020-01-02"), ("A2", "2020-01-02")])
    paired = pair_intakes_to_outcomes(intakes, outcomes)
    assert list(paired["animal_id"]) == ["A1"]
