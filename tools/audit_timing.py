"""Audit every feature for TIMING leakage, not just column provenance.

    python tools/audit_timing.py

The mechanical whitelist in features.py checks which TABLE a column came from.
That check passed for `has_name` and still missed a leak, because the question
that matters is not "which table" but "WHEN was this value written". A field
can live in the intakes table and still be updated months later.

This tool answers that question wherever the data can, and says so plainly
where it cannot.

Read-only. Trains two classifiers for the ablation, writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longstay import config  # noqa: E402
from longstay.evaluate import classifier_metrics, operating_point  # noqa: E402
from longstay.features import temporal_split  # noqa: E402
from longstay.fetch import parse_socrata_datetime, read_raw  # noqa: E402
from longstay.model import LongStayClassifierBundle, long_stay_target  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def named(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.notna() & (s.str.len() > 0)


# --------------------------------------------------------------------------
# Pair intakes to outcomes keeping BOTH sides' descriptive fields
# --------------------------------------------------------------------------


def paired_raw() -> pd.DataFrame:
    """Same pairing rule as clean.py, but retaining outcome-side columns.

    Only for auditing: comparing the intake-side and outcome-side copy of a
    field is how we detect whether the field is per-visit or a single
    animal-level attribute written into both.
    """
    intakes = read_raw(config.INTAKES_CSV)
    outcomes = read_raw(config.OUTCOMES_CSV)

    left = pd.DataFrame({
        "animal_id": intakes["animal_id"].astype("string").str.strip(),
        "intake_datetime": parse_socrata_datetime(intakes["datetime"]),
        "i_name": intakes["name"],
        "i_sex": intakes["sex_upon_intake"],
        "i_age": intakes["age_upon_intake"],
        "i_breed": intakes["breed"],
        "i_color": intakes["color"],
        "intake_type": intakes["intake_type"],
        "intake_condition": intakes["intake_condition"],
    }).dropna(subset=["animal_id", "intake_datetime"])

    right = pd.DataFrame({
        "animal_id": outcomes["animal_id"].astype("string").str.strip(),
        "outcome_datetime": parse_socrata_datetime(outcomes["datetime"]),
        "o_name": outcomes["name"],
        "o_sex": outcomes["sex_upon_outcome"],
        "o_age": outcomes["age_upon_outcome"],
        "o_breed": outcomes["breed"],
        "o_color": outcomes["color"],
        "o_dob": outcomes.get("date_of_birth"),
    }).dropna(subset=["animal_id", "outcome_datetime"])

    paired = pd.merge_asof(
        left.sort_values("intake_datetime", kind="mergesort"),
        right.sort_values("outcome_datetime", kind="mergesort"),
        left_on="intake_datetime", right_on="outcome_datetime",
        by="animal_id", direction="forward", allow_exact_matches=True,
    ).dropna(subset=["outcome_datetime"])

    return paired.sort_values(
        ["animal_id", "outcome_datetime", "intake_datetime"], kind="mergesort"
    ).drop_duplicates(subset=["animal_id", "outcome_datetime"], keep="last")


def agreement(frame: pd.DataFrame, left: str, right: str) -> dict:
    """How often the intake-side and outcome-side copies of a field differ."""
    both = frame[[left, right]].dropna()
    same = (
        both[left].astype(str).str.strip().str.casefold()
        == both[right].astype(str).str.strip().str.casefold()
    )
    return {
        "compared": int(len(both)),
        "identical": int(same.sum()),
        "identical_rate": float(same.mean()) if len(both) else float("nan"),
        "differ": int((~same).sum()),
    }


# --------------------------------------------------------------------------
# 2. Does the data support the has_name leak being real?
# --------------------------------------------------------------------------


def hypothesis_has_name(frame: pd.DataFrame) -> None:
    rule("2. IS THE has_name LEAK REAL? — what the data says")

    named_flag = frame["has_name"].fillna(False).astype(bool)
    y = long_stay_target(frame)

    print("  has_name rate by intake_type")
    print("  (a stray arriving off the street should have no name;")
    print("   an owner surrender should nearly always have one)\n")
    print(f"    {'intake_type':<20} {'n':>8} {'has_name':>10} {'expected':>12}")
    print("    " + "-" * 54)

    expectation = {
        "Stray": "~0%",
        "Public Assist": "~0%",
        "Abandoned": "~0%",
        "Owner Surrender": "~100%",
        "Euthanasia Request": "~100%",
        "Wildlife": "~0%",
    }
    for value, group in frame.groupby(frame["intake_type"].astype(str)):
        rate = group["has_name"].fillna(False).astype(bool).mean()
        print(f"    {value:<20} {len(group):>8,} {rate:>9.1%} "
              f"{expectation.get(value, '?'):>12}")

    strays = frame[frame["intake_type"] == "Stray"]
    stray_named = strays["has_name"].fillna(False).astype(bool)
    print(f"\n  Strays with a name: {stray_named.sum():,} of {len(strays):,} "
          f"({stray_named.mean():.1%})")
    print("    Every one of these is an animal that arrived off the street")
    print("    already carrying a name in the record. Either the finder")
    print("    supplied one, or the shelter added it later.")

    print(f"\n  has_name among animals that stayed "
          f"{config.LONG_STAY_DAYS}+ days:")
    print(f"    long stays  : {named_flag[y == 1].mean():.1%} "
          f"(n={int((y == 1).sum()):,})")
    print(f"    everyone else: {named_flag[y == 0].mean():.1%} "
          f"(n={int((y == 0).sum()):,})")
    print(f"    baseline     : {named_flag.mean():.1%}")

    # The decisive one: among strays only, does having a name still predict?
    print("\n  Within STRAYS only — holding intake circumstance constant:")
    for flag in (False, True):
        part = strays[stray_named == flag]
        if len(part):
            print(f"    has_name={str(flag):<5} n={len(part):>7,}  "
                  f"90+ rate={long_stay_target(part).mean():.3%}  "
                  f"median stay={part[config.TARGET].median():.1f}d")
    print("    If names were recorded at intake, a named stray and an unnamed")
    print("    stray should wait about the same. A large gap means the name")
    print("    is telling us about what happened AFTER intake.")


# --------------------------------------------------------------------------
# 4. Timing audit of every other feature
# --------------------------------------------------------------------------


def audit_fields(paired: pd.DataFrame, frame: pd.DataFrame) -> None:
    rule("4. TIMING AUDIT — could this field be written after intake?")

    print("  Method: compare the intake-side and outcome-side copy of the same")
    print("  field for the same stay. If they NEVER differ, the field is one")
    print("  animal-level value written into both rows, so any later edit")
    print("  rewrites history. If they sometimes differ, the field is stored")
    print("  per event and the intake copy is safe.\n")

    print(f"  {'field':<26} {'compared':>9} {'identical':>10} {'differ':>8}")
    print("  " + "-" * 56)
    results = {}
    for label, left, right in [
        ("name", "i_name", "o_name"),
        ("sex / sterilisation", "i_sex", "o_sex"),
        ("breed", "i_breed", "o_breed"),
        ("colour", "i_color", "o_color"),
        ("age", "i_age", "o_age"),
    ]:
        stats = agreement(paired, left, right)
        results[label] = stats
        print(f"  {label:<26} {stats['compared']:>9,} "
              f"{stats['identical_rate']:>9.2%} {stats['differ']:>8,}")

    # Sterilisation is the sharpest probe: shelters neuter animals DURING the
    # stay, so an intake-time field must show Intact -> Fixed transitions.
    print("\n  Sterilisation transitions during the stay:")
    sex = paired[["i_sex", "o_sex"]].dropna()

    def status(series):
        low = series.astype(str).str.casefold()
        return np.where(
            low.str.contains("neutered|spayed"), "Fixed",
            np.where(low.str.contains("intact"), "Intact", "Unknown"))

    before, after = status(sex["i_sex"]), status(sex["o_sex"])
    transitions = pd.crosstab(
        pd.Series(before, name="at intake"),
        pd.Series(after, name="at outcome"),
    )
    print(transitions.to_string().replace("\n", "\n    ").rjust(4))
    intact_to_fixed = int(((before == "Intact") & (after == "Fixed")).sum())
    print(f"\n    Intact at intake -> Fixed at outcome: {intact_to_fixed:,}")
    if intact_to_fixed == 0:
        print("    ZERO transitions. The shelter certainly performs surgery")
        print("    during stays, so a zero means the sterilisation status is")
        print("    a single current value copied into both rows — the same")
        print("    defect as has_name.")
    else:
        print("    Non-zero, so this field is recorded per event and the")
        print("    intake copy reflects intake-time state.")

    # Does a repeat visitor's descriptive data ever change between visits?
    print("\n  Do these fields vary across repeat visits of the same animal?")
    print("  (age must increase; breed and colour normally should not change,")
    print("   so this mainly tells us whether the row is per-visit at all)\n")
    repeats = frame[frame.duplicated("animal_id", keep=False)]
    for column in ("age_days", "primary_breed", "primary_color", "has_name"):
        varies = repeats.groupby("animal_id")[column].nunique(dropna=False)
        share = float((varies > 1).mean()) if len(varies) else float("nan")
        print(f"    {column:<16} varies across visits for {share:>6.1%} "
              f"of repeat animals")

    print("\n  intake_condition and intake_type have no outcome-side copy, so")
    print("  this comparison CANNOT be run for them. See the notes below.")


# --------------------------------------------------------------------------
# 1. Ablation: what does removing has_name actually cost?
# --------------------------------------------------------------------------


def ablation(frame: pd.DataFrame) -> None:
    rule("1. ABLATION — what does removing has_name cost?")

    train, validation, test = temporal_split(frame)
    y_test = long_stay_target(test)

    original_allowed = list(config.ALLOWED_FEATURES)
    original_boolean = list(config.BOOLEAN_FEATURES)

    rows = {}
    for label, drop in [("with has_name", False), ("without has_name", True)]:
        if drop:
            config.BOOLEAN_FEATURES = [
                c for c in original_boolean if c != "has_name"
            ]
            config.ALLOWED_FEATURES = [
                c for c in original_allowed if c != "has_name"
            ]
        else:
            config.BOOLEAN_FEATURES = list(original_boolean)
            config.ALLOWED_FEATURES = list(original_allowed)

        bundle = LongStayClassifierBundle().fit(train, validation)
        scores = bundle.predict_proba(test)
        rows[label] = {
            **classifier_metrics(y_test, scores),
            "op": operating_point(y_test, scores, config.OPERATING_THRESHOLD),
        }

    config.ALLOWED_FEATURES = original_allowed
    config.BOOLEAN_FEATURES = original_boolean

    a, b = rows["with has_name"], rows["without has_name"]
    print(f"  {'metric':<28} {'with':>12} {'without':>12} {'change':>12}")
    print("  " + "-" * 66)
    for label, get, fmt in [
        ("PR-AUC", lambda r: r["pr_auc"], "{:.4f}"),
        ("PR-AUC lift over base", lambda r: r["pr_auc_lift"], "{:.2f}x"),
        ("ROC-AUC", lambda r: r["roc_auc"], "{:.4f}"),
        ("Brier score", lambda r: r["brier_score"], "{:.4f}"),
        ("Precision @ threshold", lambda r: r["op"]["precision"], "{:.1%}"),
        ("Recall @ threshold", lambda r: r["op"]["recall"], "{:.1%}"),
        ("F1 @ threshold", lambda r: r["op"]["f1"], "{:.3f}"),
        ("Flagged share", lambda r: r["op"]["flagged_rate"], "{:.1%}"),
    ]:
        va, vb = get(a), get(b)
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<28} {fmt.format(va):>12} {fmt.format(vb):>12} "
              f"{sign + fmt.format(delta):>12}")

    change = (b["pr_auc"] - a["pr_auc"]) / a["pr_auc"]
    print(f"\n  PR-AUC change: {change:+.1%}")
    print(f"  True positives found at the working threshold: "
          f"{a['op']['confusion_matrix']['true_positive']:,} -> "
          f"{b['op']['confusion_matrix']['true_positive']:,} "
          f"(of {int(y_test.sum()):,} long stays)")


def main() -> int:
    frame = pd.read_parquet(config.JOINED_PARQUET)
    paired = paired_raw()

    ablation(frame)
    hypothesis_has_name(frame)
    audit_fields(paired, frame)

    rule("VERDICT")
    print("  See config.TEMPORAL_LEAKAGE_NOTES for the standing record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
