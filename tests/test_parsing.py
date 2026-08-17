"""Field parsers: age text, breed, colour, sex/sterilization, season."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longstay import config
from longstay.clean import (
    engineer_features,
    parse_age_to_days,
    parse_breed,
    parse_color,
    parse_sex_upon_intake,
    season_of,
)


# ---------------------------------------------------------------- age


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1 day", 1.0),
        ("3 days", 3.0),
        ("1 week", 7.0),
        ("4 weeks", 28.0),
        ("1 month", 365.25 / 12),
        ("3 months", 3 * 365.25 / 12),
        ("1 year", 365.25),
        ("2 years", 730.5),
        ("0 years", 0.0),
        ("  2 years  ", 730.5),
        ("2 YEARS", 730.5),
    ],
)
def test_age_parses(text, expected):
    assert parse_age_to_days(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad",
    [None, np.nan, "", "unknown", "NULL", "2", "years", "2 fortnights", "abc"],
)
def test_age_malformed_becomes_null(bad):
    assert np.isnan(parse_age_to_days(bad))


@pytest.mark.parametrize("bad", ["-1 years", "-3 months"])
def test_negative_age_becomes_null_not_a_small_number(bad):
    """These exist in the source. They mean the record is wrong."""
    assert np.isnan(parse_age_to_days(bad))


# ---------------------------------------------------------------- breed


@pytest.mark.parametrize(
    "text,primary,is_mix",
    [
        ("Labrador Retriever Mix", "Labrador Retriever", True),
        ("Domestic Shorthair Mix", "Domestic Shorthair", True),
        ("Beagle", "Beagle", False),
        ("Pit Bull/Labrador Retriever", "Pit Bull", True),
        ("Chihuahua Shorthair/Dachshund", "Chihuahua Shorthair", True),
        ("German Shepherd Mix", "German Shepherd", True),
        ("Australian Cattle Dog/Border Collie Mix", "Australian Cattle Dog", True),
    ],
)
def test_breed_parses(text, primary, is_mix):
    assert parse_breed(text) == (primary, is_mix)


def test_breed_mix_is_a_word_not_a_substring():
    """'Mixed' as part of another word must not count, but 'Mix' alone does."""
    primary, is_mix = parse_breed("Rhodesian Ridgeback")
    assert is_mix is False
    assert primary == "Rhodesian Ridgeback"


@pytest.mark.parametrize("bad", [None, np.nan, "", "   "])
def test_breed_null_input(bad):
    primary, is_mix = parse_breed(bad)
    assert primary is np.nan or pd.isna(primary)
    assert pd.isna(is_mix)


# ---------------------------------------------------------------- colour


@pytest.mark.parametrize(
    "text,primary,is_black",
    [
        ("Black", "Black", True),
        ("Black/White", "Black", True),
        ("White/Black", "White", False),
        ("Brown Tabby", "Brown Tabby", False),
        ("Black Tabby", "Black Tabby", False),
        ("Blue", "Blue", False),
        ("Black Smoke/White", "Black Smoke", False),
    ],
)
def test_color_parses(text, primary, is_black):
    assert parse_color(text) == (primary, is_black)


def test_is_black_is_strict_on_purpose():
    """A black tabby is a patterned cat, not a solid black one.

    The black-animal hypothesis is about solid black animals, so is_black is
    an exact match on the primary component.
    """
    assert parse_color("Black Tabby")[1] is False
    assert parse_color("Black")[1] is True


@pytest.mark.parametrize("bad", [None, np.nan, "", "   "])
def test_color_null_input(bad):
    primary, is_black = parse_color(bad)
    assert pd.isna(primary)
    assert pd.isna(is_black)


# ---------------------------------------------------------------- sex


@pytest.mark.parametrize(
    "text,sex,status",
    [
        ("Neutered Male", "Male", "Fixed"),
        ("Spayed Female", "Female", "Fixed"),
        ("Intact Male", "Male", "Intact"),
        ("Intact Female", "Female", "Intact"),
        ("Unknown", "Unknown", "Unknown"),
        (None, "Unknown", "Unknown"),
        (np.nan, "Unknown", "Unknown"),
        ("", "Unknown", "Unknown"),
    ],
)
def test_sex_splits_into_two_facts(text, sex, status):
    assert parse_sex_upon_intake(text) == (sex, status)


def test_female_is_not_misread_as_male():
    """'female' contains 'male' — the check must not be naive."""
    assert parse_sex_upon_intake("Spayed Female")[0] == "Female"
    assert parse_sex_upon_intake("Intact Female")[0] == "Female"


# ---------------------------------------------------------------- season


@pytest.mark.parametrize(
    "month,season",
    [
        (12, "Winter"), (1, "Winter"), (2, "Winter"),
        (3, "Spring"), (4, "Spring"), (5, "Spring"),
        (6, "Summer"), (7, "Summer"), (8, "Summer"),
        (9, "Fall"), (10, "Fall"), (11, "Fall"),
    ],
)
def test_season_of(month, season):
    assert season_of(month) == season


def test_every_month_has_a_season():
    assert set(config.SEASONS) == set(range(1, 13))


# ---------------------------------------------------------- engineer_features


def sample_frame():
    return pd.DataFrame(
        [
            {
                "intake_datetime": pd.Timestamp("2020-07-04 13:00"),
                "intake_name_raw": "Rex",
                "sex_upon_intake": "Neutered Male",
                "age_upon_intake": "2 years",
                "breed": "Labrador Retriever Mix",
                "color": "Black/White",
                # carried through unchanged, not derived
                "animal_type": "Dog",
                "intake_type": "Stray",
                "intake_condition": "Normal",
            },
            {
                "intake_datetime": pd.Timestamp("2020-01-15 08:00"),
                "intake_name_raw": None,
                "sex_upon_intake": "Intact Female",
                "age_upon_intake": "3 months",
                "breed": "Domestic Shorthair",
                "color": "Brown Tabby",
                "animal_type": "Cat",
                "intake_type": "Owner Surrender",
                "intake_condition": "Normal",
            },
        ]
    )


def test_engineer_features_end_to_end():
    out = engineer_features(sample_frame())

    assert out["age_days"].iloc[0] == pytest.approx(730.5)
    assert out["primary_breed"].iloc[0] == "Labrador Retriever"
    assert bool(out["is_mix"].iloc[0]) is True
    assert out["primary_color"].iloc[0] == "Black"
    assert bool(out["is_black"].iloc[0]) is True
    assert out["sex"].iloc[0] == "Male"
    assert out["sterilization_status"].iloc[0] == "Fixed"
    assert bool(out["has_name"].iloc[0]) is True

    assert out["intake_month"].iloc[0] == 7
    assert out["intake_season"].iloc[0] == "Summer"
    assert out["intake_day_of_week"].iloc[0] == 5  # 2020-07-04 was a Saturday

    assert bool(out["has_name"].iloc[1]) is False
    assert out["intake_season"].iloc[1] == "Winter"
    assert bool(out["is_mix"].iloc[1]) is False


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_has_name_is_false_for_blank_names(blank):
    frame = sample_frame()
    frame.loc[0, "intake_name_raw"] = blank
    assert bool(engineer_features(frame)["has_name"].iloc[0]) is False


def test_has_name_true_for_an_animal_literally_named_na():
    """Raw CSVs are read with keep_default_na=False for exactly this reason."""
    frame = sample_frame()
    frame.loc[0, "intake_name_raw"] = "NA"
    assert bool(engineer_features(frame)["has_name"].iloc[0]) is True


def test_engineer_features_produces_every_allowed_feature():
    out = engineer_features(sample_frame())
    missing = [c for c in config.ALLOWED_FEATURES if c not in out.columns]
    assert not missing, f"engineer_features does not produce: {missing}"
