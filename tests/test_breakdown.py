"""The generalised breakdown helper and the controlled-effect estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longstay import config
from longstay.evaluate import breakdown, controlled_effect, species_feature_contrast


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def synthetic(n_per_group: int = 200) -> pd.DataFrame:
    """Two groups with a deliberately different tail but the same median."""
    rows = []
    for group, long_share in (("A", 0.30), ("B", 0.05)):
        for i in range(n_per_group):
            long = i < int(n_per_group * long_share)
            rows.append({
                "grp": group,
                "animal_type": "Cat" if i % 2 else "Dog",
                config.TARGET: 200.0 if long else 5.0,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- breakdown


def test_breakdown_reports_both_targets():
    out = breakdown(synthetic(), "grp", min_n=1)
    assert {lvl["value"] for lvl in out["levels"]} == {"A", "B"}
    for level in out["levels"]:
        for key in ("n", "median_days", "mean_days", "long_stay_rate"):
            assert key in level


def test_breakdown_catches_a_tail_difference_the_median_hides():
    """The reason the helper reports both: same median, different tails."""
    out = breakdown(synthetic(), "grp", min_n=1)
    by = {lvl["value"]: lvl for lvl in out["levels"]}
    assert by["A"]["median_days"] == by["B"]["median_days"]  # identical
    assert by["A"]["long_stay_rate"] > by["B"]["long_stay_rate"]  # not identical


def test_breakdown_filter_narrows_the_population():
    full = breakdown(synthetic(), "grp", min_n=1)
    cats = breakdown(synthetic(), "grp", where={"animal_type": "Cat"}, min_n=1)
    assert cats["n"] < full["n"]
    assert cats["where"] == {"animal_type": "Cat"}


def test_breakdown_drops_small_levels_and_says_how_many():
    data = synthetic(200)
    data = pd.concat([data, frame([
        {"grp": "tiny", "animal_type": "Cat", config.TARGET: 4.0}
    ])])
    out = breakdown(data, "grp", min_n=50)
    assert "tiny" not in {lvl["value"] for lvl in out["levels"]}
    assert out["levels_dropped_below_min_n"] == 1
    assert out["rows_dropped_below_min_n"] == 1


def test_breakdown_levels_account_for_every_retained_row():
    out = breakdown(synthetic(), "grp", min_n=1)
    assert sum(lvl["n"] for lvl in out["levels"]) == out["n"]


def test_breakdown_shares_sum_to_one():
    out = breakdown(synthetic(), "grp", min_n=1)
    assert sum(lvl["share"] for lvl in out["levels"]) == pytest.approx(1.0)


def test_breakdown_handles_an_empty_selection():
    out = breakdown(synthetic(), "grp", where={"animal_type": "Bird"})
    assert out["n"] == 0
    assert out["levels"] == []


def test_breakdown_sorts_by_the_requested_key():
    out = breakdown(synthetic(), "grp", min_n=1, sort_by="long_stay_rate")
    rates = [lvl["long_stay_rate"] for lvl in out["levels"]]
    assert rates == sorted(rates, reverse=True)


def test_species_feature_contrast_still_returns_its_old_shape():
    """The wrapper must not have changed the contract diagnose_species uses."""
    data = synthetic(400).reset_index(drop=True)
    data["intake_type"] = np.where(data.index % 3 == 0, "Stray", "Owner Surrender")
    for column in config.CATEGORICAL_FEATURES + config.BOOLEAN_FEATURES:
        if column not in data:
            data[column] = "x"
    for column in config.NUMERIC_FEATURES:  # the wrapper summarises these too
        data[column] = 365.0
    out = species_feature_contrast(data, "Cat")
    assert set(out) >= {
        "species", "n", "long_stay_rate", "n_long_stay",
        "categorical", "numeric", "most_separating_features",
    }
    for block in out["categorical"].values():
        assert {"top", "bottom", "spread"} == set(block)
        for row in block["top"]:
            assert set(row) == {
                "value", "n", "long_stay_rate", "lift_over_species_base"
            }


# --------------------------------------------------------- controlled_effect


def confounded() -> pd.DataFrame:
    """A flag that looks harmful only because it clusters in a risky stratum.

    Within each stratum the flag does nothing; the raw comparison says
    otherwise purely because of composition.
    """
    rows = []
    for stratum, base_long in (("risky", 0.40), ("safe", 0.02)):
        # The flag is common in the risky stratum, rare in the safe one.
        for flag, count in ((True, 400 if stratum == "risky" else 100),
                            (False, 100 if stratum == "risky" else 400)):
            for i in range(count):
                rows.append({
                    "flag": flag,
                    "stratum": stratum,
                    config.TARGET: 200.0 if i < int(count * base_long) else 3.0,
                })
    return pd.DataFrame(rows)


def test_controlled_effect_removes_a_purely_compositional_difference():
    out = controlled_effect(confounded(), "flag", ["stratum"], min_stratum=10)
    assert out["usable"]
    assert abs(out["raw_difference_pp"]) > 10     # large before control
    assert abs(out["controlled_difference_pp"]) < 1  # gone after control
    assert out["ci95_low_pp"] < 0 < out["ci95_high_pp"]
    assert out["significant"] is False


def test_controlled_effect_keeps_a_real_within_stratum_difference():
    rows = []
    for stratum in ("a", "b"):
        for flag, long_share in ((True, 0.30), (False, 0.10)):
            for i in range(300):
                rows.append({
                    "flag": flag, "stratum": stratum,
                    config.TARGET: 200.0 if i < int(300 * long_share) else 3.0,
                })
    out = controlled_effect(pd.DataFrame(rows), "flag", ["stratum"],
                            min_stratum=10)
    assert out["controlled_difference_pp"] == pytest.approx(20.0, abs=1.0)
    assert out["significant"] is True


def test_controlled_effect_reports_its_coverage():
    out = controlled_effect(confounded(), "flag", ["stratum"], min_stratum=10)
    assert 0.0 < out["coverage"] <= 1.0
    assert out["strata_used"] == 2


def test_controlled_effect_drops_strata_missing_a_group():
    rows = [{"flag": True, "stratum": "only_flagged", config.TARGET: 200.0}] * 60
    rows += [{"flag": True, "stratum": "mixed", config.TARGET: 3.0}] * 60
    rows += [{"flag": False, "stratum": "mixed", config.TARGET: 3.0}] * 60
    out = controlled_effect(pd.DataFrame(rows), "flag", ["stratum"],
                            min_stratum=10)
    assert out["strata_used"] == 1
    assert out["coverage"] < 1.0


def test_controlled_effect_says_when_it_cannot_run():
    rows = [{"flag": True, "stratum": "a", config.TARGET: 3.0}] * 50
    out = controlled_effect(pd.DataFrame(rows), "flag", ["stratum"])
    assert out["usable"] is False
    assert "reason" in out


def test_confidence_interval_brackets_the_estimate():
    out = controlled_effect(confounded(), "flag", ["stratum"], min_stratum=10)
    assert out["ci95_low_pp"] < out["controlled_difference_pp"] < out["ci95_high_pp"]
