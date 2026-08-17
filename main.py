"""Longstay CLI — fetch, clean, stats.

    python main.py fetch [--force]
    python main.py clean
    python main.py stats
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import pandas as pd

from longstay import config
from longstay.clean import clean
from longstay.evaluate import (
    age_bucket_plot,
    black_effect_plot,
    build_report,
    calibration_plot,
    condition_plot,
    month_plot,
    importance_plot,
    precision_recall_plot,
    reliability_plot,
    shelter_load_plot,
    smearing_before_after_plot,
    species_reliability,
    worst_predictions,
)
from longstay.features import build_feature_matrix, temporal_split
from longstay.service import TriageService
from longstay.fetch import fetch
from longstay.model import (
    LongStayClassifierBundle,
    LongstayModels,
    long_stay_target,
)


def _banner(text: str) -> None:
    print(f"\n{text}\n{'=' * len(text)}")


def cmd_fetch(args: argparse.Namespace) -> int:
    _banner("FETCH — Austin Animal Center, Socrata open data")
    fetch(force=args.force)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    _banner("CLEAN — join, target, intake features")
    clean(verbose=True)
    return 0


def _load_processed() -> pd.DataFrame:
    if not config.JOINED_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.JOINED_PARQUET} not found. Run `python main.py clean` first."
        )
    return pd.read_parquet(config.JOINED_PARQUET)


def cmd_stats(args: argparse.Namespace) -> int:
    _banner("STATS — processed frame")
    frame = _load_processed()
    los = frame[config.TARGET]

    print(f"rows: {len(frame):,}")
    print(
        f"intake dates: {frame['intake_datetime'].min():%Y-%m-%d} "
        f"to {frame['intake_datetime'].max():%Y-%m-%d}"
    )

    print("\nLength of stay (days) — the target. We regress this, we do not")
    print("classify outcome type. See CLAUDE.md.")
    print(f"  mean    {los.mean():8.2f}")
    print(f"  median  {los.median():8.2f}")
    print(f"  p90     {los.quantile(0.90):8.2f}")
    print(f"  p99     {los.quantile(0.99):8.2f}")
    print(f"  min     {los.min():8.2f}")
    print(f"  max     {los.max():8.2f}")
    print(f"  std     {los.std():8.2f}")

    print("\nRows by animal_type:")
    counts = frame["animal_type"].value_counts(dropna=False)
    for value, count in counts.items():
        median = los[frame["animal_type"] == value].median()
        print(
            f"  {str(value):<12} {count:>8,}  ({count / len(frame):6.2%})"
            f"   median stay {median:6.1f}d"
        )

    print(f"\nTop {config.TOP_BREEDS_N} primary breeds by frequency:")
    breeds = frame["primary_breed"].value_counts(dropna=False).head(
        config.TOP_BREEDS_N
    )
    for rank, (value, count) in enumerate(breeds.items(), start=1):
        median = los[frame["primary_breed"] == value].median()
        print(
            f"  {rank:>2}. {str(value):<32} {count:>7,}"
            f"   median stay {median:6.1f}d"
        )

    print("\nBlack-animal columns (hypothesis tested later, not here):")
    black = frame["is_black"].fillna(False).astype(bool)
    starts_black = (
        frame["primary_color"].astype("string").str.casefold()
        .str.startswith("black").fillna(False)
    )
    print(
        f"  is_black (primary colour exactly 'Black'): {black.sum():,} rows "
        f"({black.mean():.2%}), median stay {los[black].median():.1f}d "
        f"vs {los[~black].median():.1f}d otherwise"
    )
    print(
        f"  primary colour merely STARTS with 'Black': {starts_black.sum():,} "
        f"rows ({starts_black.mean():.2%}) — the wider reading, reported so "
        "the\n    strict definition above is a visible choice, not a hidden one"
    )

    # Uses features.temporal_split rather than re-deriving the boundary here:
    # a second copy of the rule drifts from the first one, and did.
    print("\nTemporal split (train / validation / test, never shuffled):")
    train, validation, test = temporal_split(frame)
    print(f"  train      (<= {config.TRAIN_END_DATE}): {len(train):>8,} rows")
    print(
        f"  validation ({config.VAL_START_DATE} .. {config.VAL_END_DATE}): "
        f"{len(validation):>8,} rows"
    )
    print(f"  test       (>= {config.TEST_START_DATE}): {len(test):>8,} rows")
    print(
        f"  accounted for: {len(train) + len(validation) + len(test):,} "
        f"of {len(frame):,}"
    )

    print("\nMissingness in allowed features:")
    for column in config.ALLOWED_FEATURES:
        missing = frame[column].isna().sum()
        if missing:
            print(f"  {column:<22} {missing:>8,}  ({missing / len(frame):6.2%})")

    print("\nWhat this data cannot see, and never will:")
    for factor in config.UNOBSERVED_FACTORS:
        print(f"  - {factor}")
    print(
        "  A predicted wait is a prior from paperwork, not a judgement about\n"
        "  an animal. Every number above is computed from the frame; none is\n"
        "  estimated."
    )
    return 0


def _split_frames():
    frame = _load_processed()
    train, validation, test = temporal_split(frame)
    print("Temporal split on intake date — never shuffled:")
    for name, part in (
        ("train", train), ("validation", validation), ("test", test)
    ):
        print(
            f"  {name:<11} {len(part):>8,} rows   "
            f"{part['intake_datetime'].min():%Y-%m-%d} .. "
            f"{part['intake_datetime'].max():%Y-%m-%d}"
        )
    total = len(train) + len(validation) + len(test)
    print(f"  {'total':<11} {total:>8,} rows of {len(frame):,} in the frame")
    return train, validation, test


def cmd_train(args: argparse.Namespace) -> int:
    _banner("TRAIN — two baselines and a gradient boosting regressor")
    config.ensure_dirs()

    train, validation, test = _split_frames()

    print("\nRegression — how many days? (secondary)")
    models = LongstayModels().fit(train, validation)
    print("  1. baseline_global_median  — training median for every animal")
    print(f"     training median = {models.baseline_global.median_:.2f} days")
    print("  2. baseline_group_median   — median by animal_type x age_bucket")
    print(f"     lookup cells = {len(models.baseline_group.table_)}")
    print("  3. hist_gradient_boosting  — native categorical, no one-hot breed")
    print(f"     {len(models.gbm.feature_names_)} features; three objectives "
          "fitted for the comparison:")
    print("       log1p           squared error on log1p(days), naive expm1")
    print(
        f"       log1p_smearing  the same fit, Duan S = "
        f"{models.gbm_log.smearing_factor_:.4f} "
        f"(on {config.SMEARING_SOURCE}, n={models.gbm_log.smearing_n_:,})"
    )
    print("       raw_absolute    absolute_error on raw days, no transform")
    print(f"     IN USE: {config.REGRESSION_OBJECTIVE}")
    print(f"  + quantile models at {config.QUANTILES} for the interval")

    print(f"\nClassification — will they still be here in "
          f"{config.LONG_STAY_DAYS} days? (PRIMARY)")
    classifier = LongStayClassifierBundle().fit(train, validation)
    print(
        f"  baseline positive rate in training: "
        f"{classifier.classifier.train_positive_rate_:.2%}"
    )
    print("  hist_gradient_boosting_classifier + isotonic calibration")
    print(f"  isotonic fitted on the validation period ({len(validation):,} rows)")
    print(
        f"  operating threshold p >= {config.OPERATING_THRESHOLD:.3f} "
        f"(cost ratio {config.COST_RATIO_MISS_TO_FALSE_ALARM:g}:1, miss:false alarm)"
    )

    # The "typical animal" the API's drivers are measured against, taken from
    # the training period so explanations do not drift as new intakes arrive.
    service = TriageService(models=models, classifier=classifier)
    service.set_reference(build_feature_matrix(train))

    joblib.dump(models, config.MODEL_PATH)
    joblib.dump(classifier, config.CLASSIFIER_MODEL_PATH)
    joblib.dump(service._reference, config.REFERENCE_ROW_PATH)
    print(f"\n  saved -> {config.MODEL_PATH}")
    print(f"  saved -> {config.CLASSIFIER_MODEL_PATH}")
    print(f"  saved -> {config.REFERENCE_ROW_PATH}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    _banner("EVALUATE — held-out test set")
    config.ensure_dirs()

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{config.MODEL_PATH} not found. Run `python main.py train` first."
        )

    models = joblib.load(config.MODEL_PATH)
    classifier = joblib.load(config.CLASSIFIER_MODEL_PATH)
    train, validation, test = _split_frames()

    print("\nComputing metrics, calibration, permutation importance...")
    report = build_report(models, classifier, train, validation, test)
    report.save()

    payload = report.payload
    subtitle = (
        f"{payload['split']['test']['rows']:,} animals admitted "
        f"{config.TEST_START_DATE} onward, unseen during training"
    )
    calibration_plot(
        payload["calibration"]["table"], config.CALIBRATION_PLOT_PATH, subtitle
    )
    importance_plot(
        payload["permutation_importance"], config.IMPORTANCE_PLOT_PATH, subtitle
    )
    smearing_before_after_plot(
        payload["duan_smearing"]["calibration_table_log1p"],
        payload["duan_smearing"]["calibration_table_log1p_smearing"],
        config.CALIBRATION_BEFORE_AFTER_PATH,
        subtitle,
    )

    y_long = long_stay_target(test)
    scores = classifier.predict_proba(test)
    precision_recall_plot(
        y_long, scores, payload["classifier"]["at_operating_threshold"],
        config.PR_CURVE_PLOT_PATH, subtitle,
    )
    reliability_plot(
        payload["classifier"]["calibration"]["reliability_before"],
        payload["classifier"]["calibration"]["reliability_after"],
        config.RELIABILITY_PLOT_PATH,
        subtitle,
    )
    shelter_load_plot(payload["shelter_load"], config.SHELTER_LOAD_PLOT_PATH)

    b = payload["breakdowns"]
    age_bucket_plot(b["by_age"]["Cat"], b["by_age"]["Dog"],
                    config.AGE_BUCKET_PLOT_PATH)
    month_plot(b["by_month"]["Cat"], b["by_month"]["Dog"],
               config.MONTH_PLOT_PATH)
    condition_plot(b["by_condition"], config.CONDITION_PLOT_PATH)
    black_effect_plot(b["black_controlled"], config.BLACK_EFFECT_PLOT_PATH)

    gbm_predictions = models.predict_all(test)["hist_gradient_boosting"]
    worst = worst_predictions(test, gbm_predictions)
    worst.to_csv(config.WORST_PREDICTIONS_CSV, index=False)

    # Measured per-species reliability, consumed by the API and the result
    # card. Written here rather than hardcoded so the disclosure and the
    # evaluation can never disagree.
    reliability = species_reliability(payload["classifier"])
    config.SPECIES_RELIABILITY_PATH.write_text(
        json.dumps(reliability, indent=2), encoding="utf-8"
    )

    print_report(payload)

    print(f"\n  metrics    -> {config.METRICS_PATH}")
    print(f"  calibration-> {config.CALIBRATION_PLOT_PATH}")
    print(f"  importance -> {config.IMPORTANCE_PLOT_PATH}")
    print(f"  worst 50   -> {config.WORST_PREDICTIONS_CSV}")
    return 0


def print_report(payload: dict) -> None:
    """Print every headline number, baselines first."""
    models = payload["models"]

    # ------------------------------------------------------------- finding
    load = payload["shelter_load"]
    print("\n" + "=" * 78)
    print("THE FINDING")
    print("=" * 78)
    print(
        f"  {load['quick_exit_share_of_animals']:.1%} of animals leave within "
        f"{load['quick_exit_days_threshold']} days, consuming only "
        f"{load['quick_exit_share_of_animal_days']:.1%} of all shelter-days."
    )
    print(
        f"  {load['long_stay_share_of_animals']:.1%} of animals stay "
        f"{load['long_stay_days_threshold']}+ days, consuming "
        f"{load['long_stay_share_of_animal_days']:.1%} of all shelter-days."
    )
    print(
        f"  ({load['n_animals']:,} animals, "
        f"{load['total_animal_days']:,.0f} animal-days total; "
        f"long stays average {load['long_stay_mean_days']:.0f} days each)"
    )

    # -------------------------------------------------------- classifier
    clf = payload["classifier"]
    print("\n" + "=" * 78)
    print(f"PRIMARY MODEL — will this animal still be here in "
          f"{config.LONG_STAY_DAYS} days?")
    print("=" * 78)
    print(f"  {clf['human_summary']}")

    test = clf["test"]
    baseline = clf["baseline_positive_rate"]
    print(f"\n  {'metric':<22} {'classifier':>12} {'baseline':>12} {'lift':>10}")
    print("  " + "-" * 58)
    print(
        f"  {'PR-AUC (primary)':<22} {test['pr_auc']:>12.4f} "
        f"{baseline['pr_auc']:>12.4f} {test['pr_auc_lift']:>9.2f}x"
    )
    print(
        f"  {'ROC-AUC':<22} {test['roc_auc']:>12.4f} "
        f"{baseline['roc_auc']:>12.4f} {'—':>10}"
    )
    print(
        f"  {'Brier score':<22} {test['brier_score']:>12.4f} "
        f"{baseline['brier_score']:>12.4f} {'lower is better':>10}"
    )
    print(f"  positive rate in test: {test['positive_rate']:.2%}"
          f"   in training: {clf['train_positive_rate']:.2%}")

    print("\n  Operating points:")
    print(f"  {'point':<26} {'threshold':>10} {'precision':>10} "
          f"{'recall':>9} {'F1':>8} {'flagged':>9}")
    print("  " + "-" * 76)
    for label, key in (
        (f"chosen (cost ratio {config.COST_RATIO_MISS_TO_FALSE_ALARM:g}:1)",
         "at_operating_threshold"),
        ("at recall = 0.50", "at_recall_50"),
        ("at precision = 0.50", "at_precision_50"),
    ):
        point = clf[key]
        if point.get("unattainable"):
            print(f"  {label:<26} {'—':>10}  NOT ATTAINABLE — {point['note']}")
            continue
        print(
            f"  {label:<26} {point['threshold']:>10.4f} "
            f"{point['precision']:>10.1%} {point['recall']:>9.1%} "
            f"{point['f1']:>8.3f} {point['flagged_rate']:>8.1%}"
        )

    matrix = clf["at_operating_threshold"]["confusion_matrix"]
    print(f"\n  Confusion matrix at the chosen threshold "
          f"(p >= {config.OPERATING_THRESHOLD:.3f}):")
    print(f"    {'':<18}{'predicted quick':>18}{'predicted at-risk':>20}")
    print(f"    {'actually quick':<18}{matrix['true_negative']:>18,}"
          f"{matrix['false_positive']:>20,}")
    print(f"    {'actually 90+ days':<18}{matrix['false_negative']:>18,}"
          f"{matrix['true_positive']:>20,}")

    cal = clf["calibration"]
    print(f"\n  Probability calibration — isotonic is "
          f"{'ON' if cal['isotonic_applied'] else 'OFF'}:")
    print(f"    {'':<26}{'raw score':>12}{'isotonic':>12}")
    print(f"    {'Brier score':<26}{cal['brier_raw']:>12.4f}"
          f"{cal['brier_isotonic']:>12.4f}")
    print(f"    {'calibration error (ECE)':<26}"
          f"{cal['expected_calibration_error_raw']:>12.4f}"
          f"{cal['expected_calibration_error_isotonic']:>12.4f}")
    print(f"    in use: {cal['method']}  "
          f"(Brier {cal['brier_in_use']:.4f})")
    print(f"    {cal['decision']}")
    print(f"    isotonic would preserve ranking: {cal['ranking_preserved']}")

    print("\n  Reliability — predicted probability vs what actually happened:")
    print(f"    {'bin':>4} {'n':>7} {'mean predicted':>15} {'actual rate':>13}")
    for row in cal["reliability_after"]:
        print(
            f"    {row['bin']:>4} {row['n']:>7,} "
            f"{row['mean_predicted_probability']:>15.3f} "
            f"{row['actual_positive_rate']:>13.3f}"
        )

    if clf["by_animal_type"]:
        print("\n  By species:")
        print(f"    {'species':<10} {'n':>8} {'PR-AUC':>9} {'base':>8} "
              f"{'lift':>7} {'precision':>11} {'recall':>9}")
        for species, block in clf["by_animal_type"].items():
            point = block["at_operating_threshold"]
            print(
                f"    {species:<10} {block['n']:>8,} {block['pr_auc']:>9.4f} "
                f"{block['positive_rate']:>8.2%} {block['pr_auc_lift']:>6.2f}x "
                f"{point['precision']:>11.1%} {point['recall']:>9.1%}"
            )

    # ------------------------------------------------ species diagnosis
    print("\n" + "=" * 78)
    print("SPECIES DIAGNOSIS — is the weakness imbalance, or missing signal?")
    print("=" * 78)
    for species, block in payload["species_diagnosis"].items():
        general = block["general_model"]
        special = block["specialised_model"]
        contrast = block["feature_contrast"]

        print(f"\n  {species.upper()}  (n={general['n']:,} in test, "
              f"{contrast.get('n_long_stay', 0):,} stayed 90+ days)")
        print(f"    base rate for this species:  {general['positive_rate']:.2%}")
        print(f"    general model PR-AUC:        {general['pr_auc']:.4f}   "
              f"lift over its OWN base rate {general['pr_auc_lift']:.2f}x")
        if special:
            print(f"    species-only model PR-AUC:   {special['pr_auc']:.4f}   "
                  f"lift {special['pr_auc_lift']:.2f}x")
            print(f"    improvement from specialising: "
                  f"{block['specialised_improvement']:+.1%}  "
                  f"(threshold {block['verdict_threshold']:.0%})")
            print(f"    VERDICT: a species-only model "
                  + ("HELPS." if block["specialised_helps"]
                     else "does NOT help — the signal is not in these features."))
        else:
            print("    species-only model: not enough positives to fit one")

        importance = block["permutation_importance_within_species"]
        if importance:
            print("    permutation importance within this species "
                  "(PR-AUC lost when shuffled):")
            for row in importance[:6]:
                print(f"      {row['feature']:<24} {row['pr_auc_drop']:>8.4f}"
                      f"  +/- {row['std']:.4f}")

        if contrast.get("most_separating_features"):
            print("    features that separate long from short stays "
                  "(spread in 90+ rate):")
            for item in contrast["most_separating_features"][:4]:
                print(f"      {item['feature']:<24} {item['spread']:>8.3f}")
            for name in [i["feature"] for i in
                         contrast["most_separating_features"][:2]]:
                top = contrast["categorical"][name]["top"][0]
                bottom = contrast["categorical"][name]["bottom"][-1]
                print(f"      {name}: highest '{top['value']}' "
                      f"{top['long_stay_rate']:.1%} (n={top['n']:,})  vs "
                      f"lowest '{bottom['value']}' "
                      f"{bottom['long_stay_rate']:.1%} (n={bottom['n']:,})")

        numeric = contrast.get("numeric", {}).get("age_days")
        if numeric:
            print(f"      age_days: long stays median "
                  f"{numeric['median_long_stay']:.0f}d vs short stays "
                  f"{numeric['median_short_stay']:.0f}d")

        print(f"    RELIABLE FOR THE UI: {block['reliable']} "
              f"(needs lift >= {config.RELIABILITY_MIN_LIFT:.1f}x)")

    # -------------------------------------------- regression variants
    variants = payload["regression_variants"]
    print("\n" + "=" * 78)
    print("REGRESSION OBJECTIVE — three variants, held-out test set")
    print("=" * 78)
    print(f"  {'variant':<18} {'MAE':>8} {'MedAE':>8} {'R2':>8} "
          f"{'bias':>9} {'max|decile bias|':>18} {'<=7d':>8}")
    print("  " + "-" * 82)
    for name in config.REGRESSION_VARIANTS:
        block = variants["comparison"][name]
        marker = " *" if name == variants["chosen"] else "  "
        print(
            f"{marker}{name:<18} {block['mae_days']:>8.2f} "
            f"{block['median_absolute_error_days']:>8.2f} {block['r2']:>8.3f} "
            f"{block['mean_bias_days']:>+9.2f} "
            f"{block['max_abs_decile_bias_days']:>18.2f} "
            f"{block['within_7_days_rate']:>7.1%}"
        )
    print(f"\n  * chosen: {variants['chosen']}")

    print("\n  Mean bias by predicted-duration decile (days, + = over-predicts):")
    header = f"  {'decile':>7}" + "".join(
        f"{name[:14]:>16}" for name in config.REGRESSION_VARIANTS
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    decile_rows = {
        name: variants["comparison"][name]["bias_by_decile"]
        for name in config.REGRESSION_VARIANTS
    }
    for index in range(len(decile_rows[config.REGRESSION_VARIANTS[0]])):
        line = f"  {decile_rows[config.REGRESSION_VARIANTS[0]][index]['decile']:>7}"
        for name in config.REGRESSION_VARIANTS:
            line += f"{decile_rows[name][index]['mean_bias_days']:>+16.2f}"
        print(line)

    # Duan smearing is no longer applied — raw_absolute has no transform to
    # invert. Reported because it is why log1p was dropped.
    smearing = payload["duan_smearing"]
    comparison = smearing["comparison"]
    print(f"\n  Duan smearing — S = {smearing['factor_S']:.4f}, estimated on "
          f"{smearing['estimated_on']} (n={smearing['n_residuals']:,}):")
    print(f"    {smearing['citation']}")
    print(f"    STATUS: {smearing['status']}")
    print(
        f"    It did exactly what it promises: mean bias "
        f"{comparison['before']['mean_bias_days']:+.2f} -> "
        f"{comparison['after']['mean_bias_days']:+.2f} days, "
        f"no pair inverted ({comparison['no_inversions']}), decile "
        f"assignment identical "
        f"({comparison['decile_assignment_identical']})."
    )
    print(
        f"    It also cost "
        f"{comparison['after']['mae_days'] - comparison['before']['mae_days']:+.2f}"
        f" days of MAE and "
        f"{comparison['after']['median_absolute_error_days'] - comparison['before']['median_absolute_error_days']:+.2f}"
        " days of median error, because it targets the\n    conditional MEAN "
        "while MAE and median error are minimised by the MEDIAN. That "
        "trade-off is why\n    raw_absolute — which optimises the median "
        "directly and needs no back-transform — was chosen."
    )

    print("\n" + "=" * 78)
    print("SECONDARY MODEL — how many days? (held-out test set)")
    print("=" * 78)
    header = (
        f"  {'model':<26} {'MAE':>8} {'MedAE':>8} {'R2':>8} "
        f"{'<=7d':>8} {'<=14d':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in (
        "baseline_global_median",
        "baseline_group_median",
        "hist_gradient_boosting",
    ):
        m = models[name]["test"]
        print(
            f"  {name:<26} {m['mae_days']:>8.2f} "
            f"{m['median_absolute_error_days']:>8.2f} {m['r2']:>8.3f} "
            f"{m['within_7_days_rate']:>7.1%} {m['within_14_days_rate']:>8.1%}"
        )

    best_baseline = min(
        models["baseline_global_median"]["test"]["mae_days"],
        models["baseline_group_median"]["test"]["mae_days"],
    )
    gbm_mae = models["hist_gradient_boosting"]["test"]["mae_days"]
    delta = best_baseline - gbm_mae
    print(
        f"\n  gradient boosting vs best baseline: "
        f"{delta:+.2f} days MAE ({delta / best_baseline:+.1%})"
    )
    print(
        "  VERDICT: "
        + (
            "gradient boosting beats both baselines."
            if delta > 0
            else "gradient boosting DOES NOT beat the baselines."
        )
    )

    print("\n" + "-" * 78)
    print("MAE BY ANIMAL TYPE (gradient boosting)")
    print("-" * 78)
    print(f"  {'type':<12} {'n':>8} {'MAE':>8} {'MedAE':>8} "
          f"{'mean actual':>12} {'mean pred':>10}")
    for row in models["hist_gradient_boosting"]["test_by_animal_type"]:
        print(
            f"  {row['group']:<12} {row['n']:>8,} {row['mae_days']:>8.2f} "
            f"{row['median_absolute_error_days']:>8.2f} "
            f"{row['mean_actual_days']:>12.2f} {row['mean_predicted_days']:>10.2f}"
        )

    print("\n" + "-" * 78)
    print("MAE BY PREDICTED-DURATION DECILE (gradient boosting)")
    print("-" * 78)
    print(f"  {'decile':>6} {'n':>8} {'mean pred':>10} {'mean actual':>12} "
          f"{'median actual':>14} {'MAE':>8}")
    for row in models["hist_gradient_boosting"]["test_by_predicted_decile"]:
        print(
            f"  {row['decile']:>6} {row['n']:>8,} "
            f"{row['mean_predicted_days']:>10.2f} "
            f"{row['mean_actual_days']:>12.2f} "
            f"{row['median_actual_days']:>14.2f} {row['mae_days']:>8.2f}"
        )

    print("\n" + "-" * 78)
    print("CALIBRATION — do the intervals mean what they claim?")
    print("-" * 78)
    interval = payload["calibration"]["interval"]
    print(f"  nominal coverage of the 10-90 band: "
          f"{interval['nominal_coverage']:.0%}")
    print(f"  EMPIRICAL coverage on the test set:  "
          f"{interval['empirical_coverage']:.1%}")
    print(f"  fell below the band: {interval['below_interval_rate']:.1%}"
          f"   above the band: {interval['above_interval_rate']:.1%}")
    print(f"  median band width: "
          f"{interval['median_interval_width_days']:.1f} days"
          f"   (mean {interval['mean_interval_width_days']:.1f})")
    print(f"  quantile crossing (p10 > p90): "
          f"{interval['quantile_crossing_rate']:.2%}")

    print("\n" + "-" * 78)
    print(f"PERMUTATION IMPORTANCE — test set, top "
          f"{config.N_TOP_IMPORTANCES}")
    print("-" * 78)
    print(f"  {'feature':<24} {'MAE increase (days)':>22} {'+/- std':>10}")
    for row in payload["permutation_importance"][: config.N_TOP_IMPORTANCES]:
        print(
            f"  {row['feature']:<24} {row['mae_increase_days']:>22.3f} "
            f"{row['std']:>10.3f}"
        )

    print("\n" + "-" * 78)
    print("FAILURE ANALYSIS — the 50 worst predictions")
    print("-" * 78)
    worst = payload["failure_analysis"]["worst_predictions"]
    print(f"  mean absolute error in this group: "
          f"{worst['mean_absolute_error_days']:.1f} days")
    print(f"  mean actual {worst['mean_actual_days']:.1f} days vs "
          f"mean predicted {worst['mean_predicted_days']:.1f} days")
    print(f"  under-predicted: {worst['under_predicted']}/{worst['n']}   "
          f"over-predicted: {worst['over_predicted']}/{worst['n']}")
    print(f"  actual range: {worst['min_actual_days']:.1f} to "
          f"{worst['max_actual_days']:.1f} days")
    print(f"  by animal type: {worst['by_animal_type']}")
    print(f"  by intake type: {worst['by_intake_type']}")
    print(f"  by condition:   {worst['by_intake_condition']}")
    print(f"  by age bucket:  {worst['by_age_bucket']}")
    print(f"  top breeds:     {worst['top_breeds']}")

    print("\n" + "-" * 78)
    print(f"THE TAIL — animals who actually stayed "
          f"{config.LONG_STAY_DAYS}+ days")
    print("-" * 78)
    tail = payload["failure_analysis"]["long_stay_tail"]
    print(f"  n = {tail['n']:,} ({tail['share_of_test']:.2%} of the test set)")
    print(f"  mean actual    {tail['mean_actual_days']:>8.1f} days   "
          f"median {tail['median_actual_days']:.1f}")
    print(f"  mean PREDICTED {tail['mean_predicted_days']:>8.1f} days   "
          f"median {tail['median_predicted_days']:.1f}")
    print(f"  mean bias      {tail['mean_bias_days']:>+8.1f} days")
    print(f"  under-predicted for {tail['under_predicted_rate']:.1%} of them")
    print(f"  predicted 90+ days for only "
          f"{tail['predicted_over_threshold_rate']:.1%} of them")
    print(f"  recall if staff worked the top decile of the ranked list: "
          f"{tail['recall_in_top_decile']:.1%}")
    print(f"  recall in the top two deciles: "
          f"{tail['recall_in_top_two_deciles']:.1%}")
    print(f"  of the top decile, {tail['base_rate_top_decile']:.1%} really "
          f"did stay {config.LONG_STAY_DAYS}+ days")

    print("\n" + "-" * 78)
    print("WHAT THIS MODEL CANNOT SEE")
    print("-" * 78)
    for factor in payload["unobserved_factors"]:
        print(f"  - {factor}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="longstay",
        description=(
            "Longstay — predict shelter length of stay from intake-day "
            "features. A triage tool, not a dashboard."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download raw data from Socrata")
    p_fetch.add_argument(
        "--force",
        action="store_true",
        help="re-download even if data/raw/*.csv already exist",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    p_clean = sub.add_parser("clean", help="join, clean, engineer, save parquet")
    p_clean.set_defaults(func=cmd_clean)

    p_stats = sub.add_parser("stats", help="describe the processed frame")
    p_stats.set_defaults(func=cmd_stats)

    p_train = sub.add_parser("train", help="fit baselines and the GBM")
    p_train.set_defaults(func=cmd_train)

    p_evaluate = sub.add_parser(
        "evaluate", help="score on the held-out test set and write evals/results"
    )
    p_evaluate.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
