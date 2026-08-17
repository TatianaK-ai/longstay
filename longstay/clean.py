"""Join intakes to outcomes, compute the target, and engineer intake features.

This is where the real work is. Two things here are easy to get wrong and
expensive when wrong:

1. The intake -> outcome pairing. `animal_id` is not unique; animals are
   admitted repeatedly. Pairing an intake with the wrong outcome silently
   corrupts every number downstream. Covered by tests/test_join.py.

2. The leakage boundary. Every column recorded at or after the outcome is
   dropped here, so that nothing downstream has to remember to exclude it.
   Covered by tests/test_leakage.py.

Every step that removes rows records how many and why, in a DropLog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .fetch import count_offset_timestamps, parse_socrata_datetime, read_raw

# --------------------------------------------------------------------------
# Drop accounting
# --------------------------------------------------------------------------


@dataclass
class DropLog:
    """Running record of rows removed, so nothing disappears silently."""

    steps: list[tuple[str, int, str]] = field(default_factory=list)

    def record(self, step: str, dropped: int, reason: str) -> None:
        self.steps.append((step, int(dropped), reason))

    def render(self) -> str:
        if not self.steps:
            return "  (no rows dropped)"
        width = max(len(step) for step, _, _ in self.steps)
        lines = []
        for step, dropped, reason in self.steps:
            lines.append(f"  {step:<{width}}  {dropped:>7,}  {reason}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------

_AGE_PATTERN = re.compile(
    r"^\s*(-?\d+)\s*(day|week|month|year)s?\s*$", re.IGNORECASE
)


def parse_age_to_days(text: object) -> float:
    """Convert age text like "2 years" / "3 months" / "4 weeks" to days.

    Returns NaN for nulls, malformed text, and negative ages (which do occur
    in this dataset and are not recoverable — an age of "-1 years" tells us
    the record is wrong, not that the animal is young).
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    match = _AGE_PATTERN.match(str(text))
    if not match:
        return np.nan
    value = int(match.group(1))
    if value < 0:
        return np.nan
    return value * config.DAYS_PER_UNIT[match.group(2).lower()]


def parse_breed(breed: object) -> tuple[object, object]:
    """Return (primary_breed, is_mix).

    is_mix is true when the text contains "Mix" or a "/" separator, both of
    which the shelter uses to mean the same thing. primary_breed is the first
    component with any trailing " Mix" stripped.
    """
    if breed is None or (isinstance(breed, float) and np.isnan(breed)):
        return np.nan, np.nan
    text = str(breed).strip()
    if not text:
        return np.nan, np.nan

    is_mix = ("/" in text) or bool(re.search(r"\bmix\b", text, re.IGNORECASE))
    primary = text.split("/")[0]
    primary = re.sub(r"\s*\bmix\b\s*$", "", primary, flags=re.IGNORECASE).strip()
    return (primary or np.nan), is_mix


def parse_color(color: object) -> tuple[object, object]:
    """Return (primary_color, is_black).

    is_black is an EXACT match on the primary component being "Black", not a
    prefix match. "Black/White" -> primary "Black" -> True. "Black Tabby" ->
    primary "Black Tabby" -> False, because a black tabby is a patterned cat,
    not a solid black one. The black-animal hypothesis is about solid black
    animals, so the strict reading is the right one. `stats` reports both
    counts so the choice stays visible rather than buried here.
    """
    if color is None or (isinstance(color, float) and np.isnan(color)):
        return np.nan, np.nan
    text = str(color).strip()
    if not text:
        return np.nan, np.nan

    primary = text.split("/")[0].strip()
    return (primary or np.nan), (primary.casefold() == "black")


def parse_sex_upon_intake(value: object) -> tuple[object, object]:
    """Split "Neutered Male" into (sex, sterilization_status).

    The source column encodes two independent facts in one string. Keeping
    them fused would make "Intact Male" and "Neutered Male" unrelated
    categories to the model, which they are not.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown", "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown", "Unknown"

    lowered = text.casefold()

    if "male" in lowered:
        sex = "Female" if "female" in lowered else "Male"
    else:
        sex = "Unknown"

    if "neutered" in lowered or "spayed" in lowered:
        status = "Fixed"
    elif "intact" in lowered:
        status = "Intact"
    else:
        status = "Unknown"

    return sex, status


def season_of(month: object) -> object:
    """Meteorological season for a month number."""
    if month is None or (isinstance(month, float) and np.isnan(month)):
        return np.nan
    return config.SEASONS.get(int(month), np.nan)


# --------------------------------------------------------------------------
# The join
# --------------------------------------------------------------------------


def pair_intakes_to_outcomes(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
    drops: DropLog | None = None,
) -> pd.DataFrame:
    """Pair each intake with the NEXT outcome for that animal, chronologically.

    `animal_id` is not unique, so a plain merge would produce a cross product
    of every stay against every other stay for repeat visitors.

    The rule: within one animal, sort both sides by datetime and give each
    intake the first outcome at or after it. An outcome can only settle one
    intake, so when two intakes precede the same outcome — which means the
    first stay's outcome record is missing from the data — the LATER intake
    wins and the earlier one is dropped as unresolved. Taking the earlier
    intake instead would invent a stay spanning a visit that clearly ended.

    Exact timestamp matches are kept: same-day intake and outcome is a real,
    common event (length of stay 0), not an error.

    Expects `intake_datetime` and `outcome_datetime` columns already parsed.
    """
    intakes = intakes.dropna(subset=["animal_id", "intake_datetime"])
    outcomes = outcomes.dropna(subset=["animal_id", "outcome_datetime"])

    # merge_asof requires both sides globally sorted on the join key.
    left = intakes.sort_values("intake_datetime", kind="mergesort")
    right = (
        outcomes[["animal_id", "outcome_datetime"]]
        .sort_values("outcome_datetime", kind="mergesort")
    )

    paired = pd.merge_asof(
        left,
        right,
        left_on="intake_datetime",
        right_on="outcome_datetime",
        by="animal_id",
        direction="forward",
        allow_exact_matches=True,
    )

    before = len(paired)
    paired = paired.dropna(subset=["outcome_datetime"])
    if drops is not None:
        drops.record(
            "no matching outcome",
            before - len(paired),
            "intake has no outcome recorded at or after it (still in shelter, "
            "or outcome row missing)",
        )

    # One outcome cannot settle two intakes. Keep the latest intake before it.
    before = len(paired)
    paired = (
        paired.sort_values(
            ["animal_id", "outcome_datetime", "intake_datetime"],
            kind="mergesort",
        )
        .drop_duplicates(subset=["animal_id", "outcome_datetime"], keep="last")
    )
    if drops is not None:
        drops.record(
            "outcome already claimed",
            before - len(paired),
            "two intakes mapped to one outcome; kept the later intake, the "
            "earlier stay has no outcome record of its own",
        )

    return paired.reset_index(drop=True)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def _prepare_intakes(raw: pd.DataFrame, drops: DropLog) -> pd.DataFrame:
    """Select and rename the intake columns we are allowed to use."""
    frame = pd.DataFrame(
        {
            "animal_id": raw["animal_id"].astype("string").str.strip(),
            "intake_datetime": parse_socrata_datetime(raw["datetime"]),
            "intake_name_raw": raw.get("name"),
            "animal_type": raw["animal_type"],
            "sex_upon_intake": raw.get("sex_upon_intake"),
            "age_upon_intake": raw.get("age_upon_intake"),
            "intake_type": raw.get("intake_type"),
            "intake_condition": raw.get("intake_condition"),
            "breed": raw.get("breed"),
            "color": raw.get("color"),
        }
    )

    before = len(frame)
    frame = frame.dropna(subset=["animal_id", "intake_datetime"])
    drops.record(
        "intake unusable",
        before - len(frame),
        "missing animal_id or unparseable intake datetime",
    )
    return frame


def _prepare_outcomes(raw: pd.DataFrame, drops: DropLog) -> pd.DataFrame:
    """Take the outcome datetime and NOTHING else.

    Every other outcome column is leakage (CLAUDE.md principle 3). Dropping
    them here rather than downstream means no later step has to remember to.
    """
    frame = pd.DataFrame(
        {
            "animal_id": raw["animal_id"].astype("string").str.strip(),
            "outcome_datetime": parse_socrata_datetime(raw["datetime"]),
        }
    )

    before = len(frame)
    frame = frame.dropna(subset=["animal_id", "outcome_datetime"])
    drops.record(
        "outcome unusable",
        before - len(frame),
        "missing animal_id or unparseable outcome datetime",
    )
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the intake-day feature columns. No outcome information used."""
    frame = frame.copy()

    frame["age_days"] = frame["age_upon_intake"].map(parse_age_to_days)

    breed_parts = frame["breed"].map(parse_breed)
    frame["primary_breed"] = [p[0] for p in breed_parts]
    frame["is_mix"] = [p[1] for p in breed_parts]

    color_parts = frame["color"].map(parse_color)
    frame["primary_color"] = [p[0] for p in color_parts]
    frame["is_black"] = [p[1] for p in color_parts]

    sex_parts = frame["sex_upon_intake"].map(parse_sex_upon_intake)
    frame["sex"] = [p[0] for p in sex_parts]
    frame["sterilization_status"] = [p[1] for p in sex_parts]

    # A name present at intake usually means the animal arrived with a history
    # — an owner surrender or a return — rather than off the street.
    name = frame["intake_name_raw"].astype("string").str.strip()
    frame["has_name"] = name.notna() & (name.str.len() > 0)

    frame["intake_month"] = frame["intake_datetime"].dt.month
    frame["intake_day_of_week"] = frame["intake_datetime"].dt.dayofweek
    frame["intake_season"] = frame["intake_month"].map(season_of)

    return frame


def clean(verbose: bool = True) -> pd.DataFrame:
    """Run the full clean: load raw, join, compute target, engineer, save."""
    config.ensure_dirs()
    drops = DropLog()

    raw_intakes = read_raw(config.INTAKES_CSV)
    raw_outcomes = read_raw(config.OUTCOMES_CSV)

    if verbose:
        print(f"  raw intakes:  {len(raw_intakes):,} rows")
        print(f"  raw outcomes: {len(raw_outcomes):,} rows")
        for label, raw in (("intakes", raw_intakes), ("outcomes", raw_outcomes)):
            offsets = count_offset_timestamps(raw["datetime"])
            if offsets:
                print(
                    f"  {label}: {offsets:,} timestamps carried an explicit UTC "
                    "offset; offset stripped, local wall clock kept "
                    "(see fetch.parse_socrata_datetime)"
                )

    intakes = _prepare_intakes(raw_intakes, drops)
    outcomes = _prepare_outcomes(raw_outcomes, drops)

    frame = pair_intakes_to_outcomes(intakes, outcomes, drops)

    frame[config.TARGET] = (
        frame["outcome_datetime"] - frame["intake_datetime"]
    ).dt.total_seconds() / 86400.0

    before = len(frame)
    frame = frame[frame[config.TARGET] >= config.MIN_LOS_DAYS]
    drops.record(
        "negative stay",
        before - len(frame),
        f"length of stay < {config.MIN_LOS_DAYS:g} days (impossible; bad source data)",
    )

    before = len(frame)
    frame = frame[frame[config.TARGET] <= config.MAX_LOS_DAYS]
    drops.record(
        "stay over cap",
        before - len(frame),
        f"length of stay > {config.MAX_LOS_DAYS:g} days (real but out of scope)",
    )

    frame = engineer_features(frame)

    keep = config.NON_FEATURE_COLUMNS + config.ALLOWED_FEATURES
    frame = frame[keep].reset_index(drop=True)

    assert_no_leakage(frame)

    frame.to_parquet(config.JOINED_PARQUET, index=False)

    if verbose:
        _report_clean(frame, drops)

    return frame


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Fail loudly if any post-outcome column survived into the frame."""
    present = [c for c in config.FORBIDDEN_COLUMNS if c in frame.columns]
    if present:
        raise AssertionError(
            "LEAKAGE: post-outcome columns present in the processed frame: "
            f"{present}"
        )

    allowed = set(config.NON_FEATURE_COLUMNS) | set(config.ALLOWED_FEATURES)
    unexpected = [c for c in frame.columns if c not in allowed]
    if unexpected:
        raise AssertionError(
            "Processed frame has columns outside the whitelist "
            f"(add to config or drop): {unexpected}"
        )


def _report_clean(frame: pd.DataFrame, drops: DropLog) -> None:
    """Print the drop ledger and the sanity checks a reader should want."""
    print("\n  rows dropped, by step:")
    print(drops.render())
    print(f"\n  final row count: {len(frame):,}")

    nulls = frame["age_days"].isna().sum()
    print(
        f"  age_upon_intake unparseable/negative: {nulls:,} rows "
        f"({nulls / len(frame):.2%}) — left as null, not imputed"
    )

    horizon = frame["outcome_datetime"].max()
    print(
        f"  censoring horizon: outcomes recorded through {horizon:%Y-%m-%d}. "
        "Intakes near this date are\n"
        "    right-censored — unresolved long stays are absent, so recent "
        "weeks skew short."
    )

    print(f"\n  saved -> {config.JOINED_PARQUET}")


if __name__ == "__main__":  # pragma: no cover
    clean()
