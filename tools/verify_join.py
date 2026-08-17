"""Human-readable verification of the join and the leakage boundary.

    python tools/verify_join.py

Prints three things:
  1. Ten random rows of the processed frame, all columns.
  2. Five repeat visitors, with every intake and outcome on one timeline, so
     the intake -> NEXT outcome pairing can be checked by eye.
  3. Every column of the training matrix, with its leakage status.

This is a read-only inspection tool. It changes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longstay import config  # noqa: E402
from longstay.clean import _prepare_intakes, _prepare_outcomes, DropLog  # noqa: E402
from longstay.features import build_feature_matrix, temporal_split  # noqa: E402
from longstay.fetch import read_raw  # noqa: E402

SEED = 20260816

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 28)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    frame = pd.read_parquet(config.JOINED_PARQUET)

    # ---------------------------------------------------------------- 1
    rule("1. TEN RANDOM ROWS OF THE PROCESSED FRAME (all columns)")
    sample = frame.sample(10, random_state=SEED).sort_values("intake_datetime")
    print(f"source: {config.JOINED_PARQUET}")
    print(f"rows: {len(frame):,}   columns: {len(frame.columns)}   seed: {SEED}\n")
    print(sample.to_string(index=False))

    # ---------------------------------------------------------------- 2
    rule("2. FIVE REPEAT VISITORS — EVERY INTAKE AND OUTCOME, IN TIME ORDER")

    drops = DropLog()
    intakes = _prepare_intakes(read_raw(config.INTAKES_CSV), drops)
    outcomes = _prepare_outcomes(read_raw(config.OUTCOMES_CSV), drops)

    # Pick animals with several visits, deterministically.
    counts = intakes["animal_id"].value_counts()
    repeat_ids = counts[counts >= 3].index.to_series()
    chosen = repeat_ids.sample(5, random_state=SEED).tolist()

    for animal_id in chosen:
        animal_intakes = intakes[intakes["animal_id"] == animal_id]
        animal_outcomes = outcomes[outcomes["animal_id"] == animal_id]
        paired = frame[frame["animal_id"] == animal_id].sort_values(
            "intake_datetime"
        )

        print(f"\n--- {animal_id} "
              f"({len(animal_intakes)} intakes, {len(animal_outcomes)} outcomes, "
              f"{len(paired)} rows kept) ---")

        timeline = pd.concat(
            [
                pd.DataFrame(
                    {
                        "when": animal_intakes["intake_datetime"],
                        "event": "INTAKE",
                        "detail": animal_intakes["intake_type"].astype(str)
                        + " / "
                        + animal_intakes["intake_condition"].astype(str),
                    }
                ),
                pd.DataFrame(
                    {
                        "when": animal_outcomes["outcome_datetime"],
                        "event": "outcome",
                        "detail": "",
                    }
                ),
            ]
        ).sort_values(["when", "event"])

        print("  raw timeline:")
        for _, row in timeline.iterrows():
            print(f"    {row['when']:%Y-%m-%d %H:%M}  {row['event']:<8} {row['detail']}")

        print("  pairing produced by the join:")
        if paired.empty:
            print("    (no rows survived cleaning for this animal)")
        for _, row in paired.iterrows():
            print(
                f"    intake {row['intake_datetime']:%Y-%m-%d %H:%M}"
                f"  ->  outcome {row['outcome_datetime']:%Y-%m-%d %H:%M}"
                f"   = {row[config.TARGET]:7.2f} days"
            )

        # Mechanical check: no intake may point at an outcome that is not the
        # earliest outcome at or after it.
        for _, row in paired.iterrows():
            later = animal_outcomes.loc[
                animal_outcomes["outcome_datetime"] >= row["intake_datetime"],
                "outcome_datetime",
            ]
            expected = later.min()
            flag = "OK" if row["outcome_datetime"] == expected else "MISMATCH"
            if flag != "OK":
                print(
                    f"    !! {flag}: intake {row['intake_datetime']} got "
                    f"{row['outcome_datetime']}, earliest available was {expected}"
                )
        print("    check: every kept intake points at the earliest outcome "
              "at or after it.")

    # ---------------------------------------------------------------- 3
    rule("3. TRAINING-MATRIX COLUMNS AND THEIR LEAKAGE STATUS")

    train, validation, test = temporal_split(frame)
    matrix = build_feature_matrix(train)

    print(
        f"temporal split: train <= {config.TRAIN_END_DATE} ({len(train):,}), "
        f"validation {config.VAL_START_DATE}..{config.VAL_END_DATE} "
        f"({len(validation):,}), test >= {config.TEST_START_DATE} "
        f"({len(test):,})"
    )
    print(f"training matrix: {matrix.shape[0]:,} rows x {matrix.shape[1]} columns\n")

    outcome_columns = set(
        read_raw(config.OUTCOMES_CSV).columns
    )

    header = f"  {'#':>2}  {'column':<22} {'kind':<12} {'origin':<34} status"
    print(header)
    print("  " + "-" * (len(header) - 2))

    kinds = {}
    for c in config.NUMERIC_FEATURES:
        kinds[c] = "numeric"
    for c in config.CATEGORICAL_FEATURES:
        kinds[c] = "categorical"
    for c in config.BOOLEAN_FEATURES:
        kinds[c] = "boolean"

    origins = {
        "age_days": "intakes.age_upon_intake",
        "animal_type": "intakes.animal_type",
        "sex": "intakes.sex_upon_intake",
        "sterilization_status": "intakes.sex_upon_intake",
        "intake_type": "intakes.intake_type",
        "intake_condition": "intakes.intake_condition",
        "primary_breed": "intakes.breed",
        "primary_color": "intakes.color",
        "intake_month": "intakes.datetime",
        "intake_day_of_week": "intakes.datetime",
        "intake_season": "intakes.datetime",
        "is_mix": "intakes.breed",
        "is_black": "intakes.color",
        "has_name": "intakes.name",
    }

    leaks = []
    for i, column in enumerate(matrix.columns, start=1):
        origin = origins.get(column, "UNKNOWN")
        from_outcomes = not origin.startswith("intakes.")
        if from_outcomes or column in config.FORBIDDEN_COLUMNS:
            status = "*** LEAK ***"
            leaks.append(column)
        else:
            status = "clean (known at intake)"
        print(f"  {i:>2}  {column:<22} {kinds[column]:<12} {origin:<34} {status}")

    print(f"\n  every feature derives from the intakes table: "
          f"{'NO — SEE ABOVE' if leaks else 'yes'}")

    # Proof rather than assertion: _prepare_outcomes is the ONLY path by which
    # outcome data enters the pipeline, and it selects exactly two columns.
    outcome_side = _prepare_outcomes(read_raw(config.OUTCOMES_CSV), DropLog())
    print(f"  columns the outcomes table contributes at all: "
          f"{sorted(outcome_side.columns)}")
    print("    (_prepare_outcomes selects only these two; every other outcome "
          "column is discarded\n     before the join, so it cannot reach the "
          "model by any route)")

    collisions = sorted(set(matrix.columns) & outcome_columns)
    if collisions:
        print(f"  name collisions with outcomes.csv: {collisions}")
        print("    these columns exist in BOTH tables; the frame takes them "
              "from intakes\n     (see clean._prepare_intakes), so the shared "
              "name is not leakage")

    print("\n  outcomes.csv columns and where each one went:")
    for column in sorted(outcome_columns):
        if column == "datetime":
            note = "-> outcome_datetime, used ONLY to compute the target"
        elif column == "animal_id":
            note = "-> join key only, not a feature"
        else:
            note = "dropped in clean.py (leakage)"
        print(f"    {column:<20} {note}")

    print("\n  forbidden columns present in training matrix: "
          f"{[c for c in config.FORBIDDEN_COLUMNS if c in matrix.columns] or 'none'}")
    print(f"  target ({config.TARGET}) present as a feature: "
          f"{config.TARGET in matrix.columns}")
    print(f"  outcome_datetime present as a feature: "
          f"{'outcome_datetime' in matrix.columns}")

    print("\n  build_feature_matrix() asserted the whitelist and did not raise.")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
