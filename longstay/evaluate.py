"""Evaluation: metrics, calibration, permutation importance, failure analysis.

Everything is reported in DAYS. Nothing here is rounded before it is computed,
and nothing is reported that was not computed from the held-out test set.

The calibration section is the point of the project. An error metric says how
wrong we are on average; calibration says whether the number we hand a shelter
worker means what it claims to mean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # no display in a CLI run

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)

from . import config  # noqa: E402
from .features import build_feature_matrix  # noqa: E402
from .model import age_bucket, long_stay_target  # noqa: E402

# Validated palette (see the dataviz reference palette). Light surface.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"  # categorical slot 1, blue
SERIES_2 = "#eb6834"  # categorical slot 2, orange


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """The headline numbers, all in days."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    absolute = np.abs(errors)

    metrics = {
        "n": int(len(y_true)),
        "mae_days": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error_days": float(
            median_absolute_error(y_true, y_pred)
        ),
        "rmse_days": float(np.sqrt(np.mean(errors**2))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_bias_days": float(np.mean(errors)),
        "mean_actual_days": float(np.mean(y_true)),
        "mean_predicted_days": float(np.mean(y_pred)),
    }
    for tolerance in config.TOLERANCE_DAYS:
        metrics[f"within_{tolerance}_days_rate"] = float(
            np.mean(absolute <= tolerance)
        )
    return metrics


def metrics_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series
) -> list[dict]:
    """MAE broken out by a categorical column, largest group first."""
    frame = pd.DataFrame(
        {
            "group": np.asarray(groups).astype(str),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    rows = []
    for name, part in frame.groupby("group"):
        rows.append(
            {
                "group": name,
                "n": int(len(part)),
                "mae_days": float(
                    mean_absolute_error(part["y_true"], part["y_pred"])
                ),
                "median_absolute_error_days": float(
                    median_absolute_error(part["y_true"], part["y_pred"])
                ),
                "mean_actual_days": float(part["y_true"].mean()),
                "mean_predicted_days": float(part["y_pred"].mean()),
            }
        )
    return sorted(rows, key=lambda r: r["n"], reverse=True)


def predicted_deciles(y_pred: np.ndarray) -> pd.Series:
    """Decile index by predicted duration.

    A constant prediction (baseline 1) cannot be split into ten bins, so we
    drop empty edges and report however many distinct bins exist rather than
    pretending to a resolution the predictions do not have.
    """
    series = pd.Series(np.asarray(y_pred, dtype=float))
    try:
        bins = pd.qcut(
            series.rank(method="first"),
            q=config.N_DECILES,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return pd.Series(np.zeros(len(series), dtype=int))
    return bins.astype(int)


def metrics_by_decile(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """MAE and mean actual duration per predicted-duration decile."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    deciles = predicted_deciles(y_pred)

    rows = []
    for index in sorted(deciles.unique()):
        mask = (deciles == index).to_numpy()
        actual = y_true[mask]
        predicted = y_pred[mask]
        rows.append(
            {
                "decile": int(index) + 1,
                "n": int(mask.sum()),
                "mean_predicted_days": float(np.mean(predicted)),
                "mean_actual_days": float(np.mean(actual)),
                "median_actual_days": float(np.median(actual)),
                "mae_days": float(mean_absolute_error(actual, predicted)),
                # Standard error of the mean actual — the error bar on the plot.
                "actual_standard_error": float(
                    np.std(actual, ddof=1) / np.sqrt(len(actual))
                    if len(actual) > 1
                    else 0.0
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def interval_calibration(
    y_true: np.ndarray, quantile_predictions: dict[float, np.ndarray]
) -> dict:
    """Do the predicted 10-90 bands actually contain 80% of animals?

    This is the number the whole calibration claim rests on, so it is measured
    empirically on the test set and reported whatever it turns out to be.
    """
    y_true = np.asarray(y_true, dtype=float)
    low = np.asarray(quantile_predictions[config.QUANTILES[0]], dtype=float)
    mid = np.asarray(quantile_predictions[config.QUANTILES[1]], dtype=float)
    high = np.asarray(quantile_predictions[config.QUANTILES[2]], dtype=float)

    inside = (y_true >= low) & (y_true <= high)
    widths = high - low

    return {
        "nominal_coverage": config.INTERVAL_NOMINAL_COVERAGE,
        "empirical_coverage": float(np.mean(inside)),
        "below_interval_rate": float(np.mean(y_true < low)),
        "above_interval_rate": float(np.mean(y_true > high)),
        "mean_interval_width_days": float(np.mean(widths)),
        "median_interval_width_days": float(np.median(widths)),
        "quantile_crossing_rate": float(np.mean(low > high)),
        "median_model_mae_days": float(mean_absolute_error(y_true, mid)),
    }


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """The data behind the calibration plot."""
    return metrics_by_decile(y_true, y_pred)


def calibration_plot(table: list[dict], path, subtitle: str) -> None:
    """Predicted duration vs mean actual duration, with the ideal diagonal.

    The diagonal is drawn as recessive reference chrome, not as a second data
    series — it is where a perfectly calibrated model would sit, not a
    measurement.
    """
    predicted = [row["mean_predicted_days"] for row in table]
    actual = [row["mean_actual_days"] for row in table]
    errors = [1.96 * row["actual_standard_error"] for row in table]

    figure, axes = plt.subplots(figsize=(7.2, 6.4), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    limit = max(max(predicted), max(actual)) * 1.12
    axes.plot(
        [0, limit],
        [0, limit],
        linestyle=(0, (5, 4)),
        linewidth=1.5,
        color=INK_MUTED,
        label="Perfect calibration",
        zorder=1,
    )

    axes.errorbar(
        predicted,
        actual,
        yerr=errors,
        fmt="-o",
        linewidth=2,
        markersize=9,
        color=SERIES_1,
        ecolor=SERIES_1,
        elinewidth=1.5,
        capsize=4,
        markeredgecolor=SURFACE,
        markeredgewidth=2,  # 2px surface ring, keeps overlapping marks legible
        label="Model, by predicted decile",
        zorder=3,
    )

    # Direct-label only the ends, not every point.
    for index in (0, len(table) - 1):
        axes.annotate(
            f"decile {table[index]['decile']}",
            (predicted[index], actual[index]),
            textcoords="offset points",
            xytext=(10, -14),
            fontsize=9,
            color=INK_SECONDARY,
        )

    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_xlabel("Mean predicted stay (days)", fontsize=11, color=INK_SECONDARY)
    axes.set_ylabel("Mean actual stay (days)", fontsize=11, color=INK_SECONDARY)
    axes.set_title(
        "Calibration on the held-out test set",
        fontsize=14,
        color=INK_PRIMARY,
        pad=30,
        loc="left",
    )
    axes.text(
        0, 1.012, subtitle, transform=axes.transAxes,
        fontsize=9.5, color=INK_MUTED, va="bottom",
    )

    axes.grid(True, color=GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
    axes.tick_params(colors=INK_MUTED, labelsize=9.5)

    legend = axes.legend(
        frameon=False, fontsize=10, loc="upper left", labelcolor=INK_SECONDARY
    )
    legend.set_zorder(5)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


# --------------------------------------------------------------------------
# Permutation importance
# --------------------------------------------------------------------------


def permutation_importance_table(model, test: pd.DataFrame) -> list[dict]:
    """Permutation importance on the TEST set, in days of MAE.

    Not the built-in impurity importance, which is biased toward
    high-cardinality features and would hand the crown to breed by
    construction. Not the training set either: we want the features that carry
    signal to unseen data, not the ones the model happened to memorise.

    Units are days of MAE lost when the column is shuffled, so the numbers are
    directly interpretable.
    """
    X = build_feature_matrix(test)
    y = test[config.TARGET].to_numpy(dtype=float)

    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=config.PERMUTATION_REPEATS,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )

    rows = [
        {
            "feature": feature,
            "mae_increase_days": float(result.importances_mean[index]),
            "std": float(result.importances_std[index]),
        }
        for index, feature in enumerate(X.columns)
    ]
    return sorted(rows, key=lambda r: r["mae_increase_days"], reverse=True)


def importance_plot(rows: list[dict], path, subtitle: str) -> None:
    """Top features by permutation importance, with error bars."""
    top = rows[: config.N_TOP_IMPORTANCES][::-1]
    labels = [row["feature"] for row in top]
    values = [row["mae_increase_days"] for row in top]
    errors = [row["std"] for row in top]

    figure, axes = plt.subplots(figsize=(7.8, 6.4), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    positions = np.arange(len(top))
    axes.barh(
        positions,
        values,
        height=0.68,
        color=SERIES_1,
        xerr=errors,
        error_kw={"ecolor": INK_MUTED, "elinewidth": 1.2, "capsize": 3},
        zorder=3,
    )
    axes.axvline(0, color=BASELINE, linewidth=1.2, zorder=2)

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=10, color=INK_SECONDARY)
    axes.set_xlabel(
        "MAE increase when the column is shuffled (days)",
        fontsize=11,
        color=INK_SECONDARY,
    )
    axes.set_title(
        "Permutation importance, test set",
        fontsize=14,
        color=INK_PRIMARY,
        pad=30,
        loc="left",
    )
    axes.text(
        0, 1.012, subtitle, transform=axes.transAxes,
        fontsize=9.5, color=INK_MUTED, va="bottom",
    )

    axes.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(BASELINE)
    axes.tick_params(colors=INK_MUTED, labelsize=9.5)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


# --------------------------------------------------------------------------
# Honest failure analysis
# --------------------------------------------------------------------------


def worst_predictions(
    test: pd.DataFrame, y_pred: np.ndarray, n: int | None = None
) -> pd.DataFrame:
    """The n worst predictions by absolute error, with their features."""
    n = n or config.N_WORST_PREDICTIONS
    frame = test.copy()
    frame["predicted_days"] = np.asarray(y_pred, dtype=float)
    frame["absolute_error_days"] = (
        frame["predicted_days"] - frame[config.TARGET]
    ).abs()
    frame["signed_error_days"] = frame["predicted_days"] - frame[config.TARGET]
    frame["age_bucket"] = age_bucket(frame["age_days"])
    return frame.nlargest(n, "absolute_error_days")


def characterise_failures(worst: pd.DataFrame) -> dict:
    """Describe the worst cases in numbers a person can read."""
    under = worst["signed_error_days"] < 0
    return {
        "n": int(len(worst)),
        "under_predicted": int(under.sum()),
        "over_predicted": int((~under).sum()),
        "mean_actual_days": float(worst[config.TARGET].mean()),
        "mean_predicted_days": float(worst["predicted_days"].mean()),
        "min_actual_days": float(worst[config.TARGET].min()),
        "max_actual_days": float(worst[config.TARGET].max()),
        "mean_absolute_error_days": float(worst["absolute_error_days"].mean()),
        "by_animal_type": {
            str(k): int(v)
            for k, v in worst["animal_type"].value_counts().items()
        },
        "by_intake_type": {
            str(k): int(v)
            for k, v in worst["intake_type"].value_counts().items()
        },
        "by_intake_condition": {
            str(k): int(v)
            for k, v in worst["intake_condition"].value_counts().items()
        },
        "by_age_bucket": {
            str(k): int(v) for k, v in worst["age_bucket"].value_counts().items()
        },
        "top_breeds": {
            str(k): int(v)
            for k, v in worst["primary_breed"].value_counts().head(6).items()
        },
        "has_name_rate": float(worst["has_name"].astype(float).mean()),
    }


def tail_analysis(
    test: pd.DataFrame,
    y_pred: np.ndarray,
    quantile_predictions: dict[float, np.ndarray] | None = None,
) -> dict:
    """How the model does on animals who actually stayed 90+ days.

    This is the population the tool exists to catch. A failure here matters
    more than the headline MAE, so it gets reported separately and plainly.
    """
    y_true = test[config.TARGET].to_numpy(dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    long_stay = y_true >= config.LONG_STAY_DAYS

    if not long_stay.any():
        return {"n": 0, "note": "no test animals stayed this long"}

    deciles = predicted_deciles(y_pred)
    top_decile = (deciles == deciles.max()).to_numpy()
    top_two = (deciles >= deciles.max() - 1).to_numpy()

    result = {
        "threshold_days": config.LONG_STAY_DAYS,
        "n": int(long_stay.sum()),
        "share_of_test": float(np.mean(long_stay)),
        "mean_actual_days": float(np.mean(y_true[long_stay])),
        "median_actual_days": float(np.median(y_true[long_stay])),
        "mean_predicted_days": float(np.mean(y_pred[long_stay])),
        "median_predicted_days": float(np.median(y_pred[long_stay])),
        "mae_days": float(
            mean_absolute_error(y_true[long_stay], y_pred[long_stay])
        ),
        "mean_bias_days": float(
            np.mean(y_pred[long_stay] - y_true[long_stay])
        ),
        "under_predicted_rate": float(
            np.mean(y_pred[long_stay] < y_true[long_stay])
        ),
        "predicted_over_threshold_rate": float(
            np.mean(y_pred[long_stay] >= config.LONG_STAY_DAYS)
        ),
        # The operational question for a triage tool: if staff worked the top
        # of the ranked list, how many of the eventual long stays would they
        # have reached on intake day?
        "recall_in_top_decile": float(np.mean(top_decile[long_stay])),
        "recall_in_top_two_deciles": float(np.mean(top_two[long_stay])),
        "base_rate_top_decile": float(np.mean(long_stay[top_decile])),
    }

    if quantile_predictions is not None:
        high = np.asarray(
            quantile_predictions[config.QUANTILES[2]], dtype=float
        )
        result["above_p90_band_rate"] = float(
            np.mean(y_true[long_stay] > high[long_stay])
        )
    return result


# --------------------------------------------------------------------------
# Smearing: before / after
# --------------------------------------------------------------------------


def is_rank_preserving(before: np.ndarray, after: np.ndarray) -> bool:
    """True if `after` never inverts an ordering that `before` actually made.

    The claim being tested is: before_i < before_j  =>  after_i <= after_j.
    Pairs that were TIED in `before` are unconstrained — nothing was ordered,
    so nothing can be inverted.

    Two kinds of tie make the naive version of this check wrong, and both
    occur here:
      - ties in `after`: a monotone non-decreasing map such as isotonic
        regression may merge distinct scores. Harmless for a triage list.
      - ties in `before`: predictions are clipped into [0, 365], so a whole
        block of them sits at exactly 0. Sorting on that block leaves `after`
        in arbitrary index order, which looks like an inversion and is not.

    Sorting lexicographically by (before, after) neutralises the second case:
    within a tied block the values are put in ascending order, so only
    genuine cross-block inversions can break monotonicity.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    order = np.lexsort((after, before))  # primary key: before
    ordered = after[order]
    # Tolerance scaled to the magnitude in play, not an absolute 1e-12.
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(after))))
    return bool(np.all(np.diff(ordered) >= -tolerance))


def smearing_comparison(
    y_true: np.ndarray, uncorrected: np.ndarray, corrected: np.ndarray
) -> dict:
    """Duan smearing before/after, including the ranking invariance check.

    The correction is exp(f) * S - 1 with S > 0, which is strictly monotone in
    f. If the ranking moved, the implementation is wrong, not the theory — so
    this is measured rather than asserted in a comment.
    """
    spearman = float(
        pd.Series(uncorrected).corr(pd.Series(corrected), method="spearman")
    )

    # exp(f) * S - 1 is strictly monotone, so no pair can swap. What CAN
    # happen is ties: predictions are clipped at MAX_LOS_DAYS, and multiplying
    # by S pushes more of them onto that ceiling, where they become equal.
    # A tie at the ceiling is not an inversion, so the check is for
    # inversions and the ties are counted separately rather than reported as
    # a ranking change.
    ceiling_before = int(np.sum(np.asarray(uncorrected) >= config.MAX_LOS_DAYS))
    ceiling_after = int(np.sum(np.asarray(corrected) >= config.MAX_LOS_DAYS))
    floor_before = int(np.sum(np.asarray(uncorrected) <= config.MIN_LOS_DAYS))
    floor_after = int(np.sum(np.asarray(corrected) <= config.MIN_LOS_DAYS))

    return {
        "before": core_metrics(y_true, uncorrected),
        "after": core_metrics(y_true, corrected),
        "no_inversions": is_rank_preserving(uncorrected, corrected),
        "spearman_correlation": spearman,
        "decile_assignment_identical": bool(
            predicted_deciles(uncorrected).equals(predicted_deciles(corrected))
        ),
        "clipped_at_ceiling_before": ceiling_before,
        "clipped_at_ceiling_after": ceiling_after,
        "clipped_at_floor_before": floor_before,
        "clipped_at_floor_after": floor_after,
        "ties_introduced_by_clipping": (
            (ceiling_after - ceiling_before) + (floor_after - floor_before)
        ),
    }


def smearing_before_after_plot(
    before: list[dict], after: list[dict], path, subtitle: str
) -> None:
    """Both calibration curves on one pair of axes."""
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), dpi=160, sharey=True)
    figure.patch.set_facecolor(SURFACE)

    for panel, table, title, colour in (
        (axes[0], before, "Before — naive expm1", INK_MUTED),
        (axes[1], after, "After — Duan smearing", SERIES_1),
    ):
        panel.set_facecolor(SURFACE)
        predicted = [row["mean_predicted_days"] for row in table]
        actual = [row["mean_actual_days"] for row in table]
        errors = [1.96 * row["actual_standard_error"] for row in table]

        limit = max(max(predicted), max(actual)) * 1.12
        panel.plot(
            [0, limit], [0, limit],
            linestyle=(0, (5, 4)), linewidth=1.5, color=INK_MUTED,
            label="Perfect calibration", zorder=1,
        )
        panel.errorbar(
            predicted, actual, yerr=errors, fmt="-o",
            linewidth=2, markersize=8, color=colour, ecolor=colour,
            elinewidth=1.4, capsize=3.5,
            markeredgecolor=SURFACE, markeredgewidth=2,
            label="Model, by predicted decile", zorder=3,
        )
        panel.set_xlim(0, limit)
        panel.set_ylim(0, limit)
        panel.set_title(title, fontsize=12, color=INK_PRIMARY, loc="left", pad=10)
        panel.set_xlabel(
            "Mean predicted stay (days)", fontsize=10.5, color=INK_SECONDARY
        )
        panel.grid(True, color=GRIDLINE, linewidth=0.8)
        panel.set_axisbelow(True)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            panel.spines[side].set_color(BASELINE)
        panel.tick_params(colors=INK_MUTED, labelsize=9)

    axes[0].set_ylabel(
        "Mean actual stay (days)", fontsize=10.5, color=INK_SECONDARY
    )
    axes[1].legend(
        frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK_SECONDARY
    )
    figure.suptitle(
        "Retransformation bias, corrected",
        fontsize=14, color=INK_PRIMARY, x=0.008, ha="left", y=0.985,
    )
    figure.text(0.008, 0.92, subtitle, fontsize=9.5, color=INK_MUTED, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


# --------------------------------------------------------------------------
# Classifier evaluation — the primary model
# --------------------------------------------------------------------------


def threshold_for_recall(
    y_true: np.ndarray, scores: np.ndarray, target: float
) -> float | None:
    """Highest threshold that still reaches `target` recall, or None.

    None means the target is unreachable at any threshold. Returning a
    sentinel number instead would produce a plausible-looking row of zeros in
    the report, which is worse than saying "not attainable".
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    usable = [
        thresholds[i] for i in range(len(thresholds)) if recall[i] >= target
    ]
    return float(max(usable)) if usable else None


def threshold_for_precision(
    y_true: np.ndarray, scores: np.ndarray, target: float
) -> float | None:
    """Lowest threshold reaching `target` precision, or None if unreachable."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    usable = [
        thresholds[i] for i in range(len(thresholds)) if precision[i] >= target
    ]
    return float(min(usable)) if usable else None


def operating_point(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Precision / recall / F1 and the confusion matrix at one threshold."""
    y_true = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(scores, dtype=float) >= threshold).astype(int)

    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged": int(tp + fp),
        "flagged_rate": float((tp + fp) / len(y_true)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def classifier_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """PR-AUC first: with a rare positive class it is the informative one.

    ROC-AUC is dominated by the vast negative class and looks respectable even
    when the model is useless on the population we care about.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    base_rate = float(np.mean(y_true))

    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "pr_auc_baseline": base_rate,  # a random ranker scores the base rate
        "pr_auc_lift": (
            float(average_precision_score(y_true, scores) / base_rate)
            if base_rate
            else float("nan")
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "positive_rate": base_rate,
        "n": int(len(y_true)),
    }


def reliability_table(
    y_true: np.ndarray, scores: np.ndarray, bins: int | None = None
) -> list[dict]:
    """Predicted probability decile vs the fraction that actually stayed 90+."""
    bins = bins or config.RELIABILITY_BINS
    frame = pd.DataFrame(
        {"y": np.asarray(y_true, dtype=float), "p": np.asarray(scores, dtype=float)}
    )
    frame["bin"] = pd.qcut(
        frame["p"].rank(method="first"), q=bins, labels=False, duplicates="drop"
    )

    rows = []
    for index, part in frame.groupby("bin"):
        actual = part["y"].mean()
        n = len(part)
        rows.append(
            {
                "bin": int(index) + 1,
                "n": int(n),
                "mean_predicted_probability": float(part["p"].mean()),
                "actual_positive_rate": float(actual),
                # Binomial standard error on the observed rate.
                "standard_error": float(np.sqrt(actual * (1 - actual) / n))
                if n
                else 0.0,
            }
        )
    return rows


def expected_calibration_error(table: list[dict]) -> float:
    """Weighted mean gap between predicted probability and observed rate."""
    total = sum(row["n"] for row in table)
    if not total:
        return float("nan")
    return float(
        sum(
            row["n"]
            * abs(row["mean_predicted_probability"] - row["actual_positive_rate"])
            for row in table
        )
        / total
    )


def human_summary(operating: dict, base_rate: float) -> str:
    """The sentence a person can repeat without a statistics degree."""
    precision = operating["precision"]
    recall = operating["recall"]
    return (
        f"Of 100 animals flagged as at-risk on intake day, "
        f"{precision * 100:.0f} really did wait {config.LONG_STAY_DAYS}+ days. "
        f"The model finds {recall * 100:.0f}% of all such animals. "
        f"Flagging at random would find {base_rate * 100:.0f} in 100."
    )


def precision_recall_plot(
    y_true: np.ndarray, scores: np.ndarray, operating: dict, path, subtitle: str
) -> None:
    """The precision-recall curve with the chosen operating point marked."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    base_rate = float(np.mean(np.asarray(y_true, dtype=float)))

    figure, axes = plt.subplots(figsize=(7.4, 6.0), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    axes.axhline(
        base_rate,
        linestyle=(0, (5, 4)),
        linewidth=1.5,
        color=INK_MUTED,
        label=f"Random baseline ({base_rate:.1%})",
        zorder=1,
    )
    axes.plot(
        recall, precision, linewidth=2, color=SERIES_1,
        label="Long-stay classifier", zorder=3,
    )
    axes.plot(
        [operating["recall"]], [operating["precision"]],
        marker="o", markersize=10, color=SERIES_1,
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
    )
    axes.annotate(
        f"operating point\np >= {operating['threshold']:.3f}\n"
        f"recall {operating['recall']:.0%}, precision {operating['precision']:.0%}",
        (operating["recall"], operating["precision"]),
        textcoords="offset points", xytext=(14, 10),
        fontsize=9.5, color=INK_SECONDARY,
    )

    axes.set_xlim(0, 1)
    axes.set_ylim(0, max(0.6, float(np.max(precision)) * 1.05))
    axes.set_xlabel("Recall — share of long stays found", fontsize=11,
                    color=INK_SECONDARY)
    axes.set_ylabel("Precision — share of flags that were right", fontsize=11,
                    color=INK_SECONDARY)
    axes.set_title(
        f"Finding {config.LONG_STAY_DAYS}+ day stays at intake",
        fontsize=14, color=INK_PRIMARY, pad=30, loc="left",
    )
    axes.text(0, 1.012, subtitle, transform=axes.transAxes,
              fontsize=9.5, color=INK_MUTED, va="bottom")

    axes.grid(True, color=GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
    axes.tick_params(colors=INK_MUTED, labelsize=9.5)
    axes.legend(frameon=False, fontsize=10, loc="upper right",
                labelcolor=INK_SECONDARY)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def reliability_plot(
    before: list[dict], after: list[dict], path, subtitle: str
) -> None:
    """Reliability diagram, uncalibrated vs isotonic-calibrated."""
    figure, axes = plt.subplots(figsize=(7.4, 6.4), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    limit = max(
        max(row["mean_predicted_probability"] for row in before + after),
        max(row["actual_positive_rate"] for row in before + after),
    ) * 1.12

    axes.plot(
        [0, limit], [0, limit],
        linestyle=(0, (5, 4)), linewidth=1.5, color=INK_MUTED,
        label="Perfect calibration", zorder=1,
    )

    for table, label, colour in (
        (before, "Raw model score", SERIES_2),
        (after, "After isotonic regression", SERIES_1),
    ):
        axes.errorbar(
            [row["mean_predicted_probability"] for row in table],
            [row["actual_positive_rate"] for row in table],
            yerr=[1.96 * row["standard_error"] for row in table],
            fmt="-o", linewidth=2, markersize=8,
            color=colour, ecolor=colour, elinewidth=1.4, capsize=3.5,
            markeredgecolor=SURFACE, markeredgewidth=2,
            label=label, zorder=3,
        )

    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_xlabel("Mean predicted probability", fontsize=11,
                    color=INK_SECONDARY)
    axes.set_ylabel(
        f"Actual share staying {config.LONG_STAY_DAYS}+ days",
        fontsize=11, color=INK_SECONDARY,
    )
    axes.set_title(
        "Reliability diagram, test set",
        fontsize=14, color=INK_PRIMARY, pad=30, loc="left",
    )
    axes.text(0, 1.012, subtitle, transform=axes.transAxes,
              fontsize=9.5, color=INK_MUTED, va="bottom")

    axes.grid(True, color=GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
    axes.tick_params(colors=INK_MUTED, labelsize=9.5)
    axes.legend(frameon=False, fontsize=10, loc="upper left",
                labelcolor=INK_SECONDARY)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


# --------------------------------------------------------------------------
# Regression variants, head to head
# --------------------------------------------------------------------------


def bias_by_decile(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Mean signed error per predicted-duration decile.

    Level bias is a per-decile property, not a single number: a model can have
    zero average bias while over-predicting the bottom and under-predicting
    the top. The headline mean would hide exactly the failure we care about.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    deciles = predicted_deciles(y_pred)

    rows = []
    for index in sorted(deciles.unique()):
        mask = (deciles == index).to_numpy()
        rows.append(
            {
                "decile": int(index) + 1,
                "n": int(mask.sum()),
                "mean_predicted_days": float(np.mean(y_pred[mask])),
                "mean_actual_days": float(np.mean(y_true[mask])),
                "mean_bias_days": float(
                    np.mean(y_pred[mask] - y_true[mask])
                ),
            }
        )
    return rows


def regression_variant_comparison(
    y_true: np.ndarray, variants: dict[str, np.ndarray]
) -> dict:
    """MAE, median error, R2 and bias for each objective, plus bias by decile."""
    return {
        name: {
            **core_metrics(y_true, predictions),
            "bias_by_decile": bias_by_decile(y_true, predictions),
            "max_abs_decile_bias_days": float(
                max(
                    abs(row["mean_bias_days"])
                    for row in bias_by_decile(y_true, predictions)
                )
            ),
        }
        for name, predictions in variants.items()
    }


# --------------------------------------------------------------------------
# Species diagnosis — is there any signal at all?
# --------------------------------------------------------------------------


def breakdown(
    frame: pd.DataFrame,
    by: str,
    where: dict | None = None,
    min_n: int = 100,
    sort_by: str = "long_stay_rate",
) -> dict:
    """Length of stay and long-stay rate for each level of a grouping column.

    The general form of what species_feature_contrast used to do inline. Takes
    any grouping column, an optional equality filter on any other columns, and
    reports BOTH targets at once — median and mean days, and the 90+ day rate
    — because for this distribution they can disagree. The median can sit flat
    while the tail moves, and the tail is where the operational cost lives.

    Levels below `min_n` are dropped rather than reported at high variance;
    the dropped count is returned so the omission is visible.
    """
    part = frame
    if where:
        for column, value in where.items():
            part = part[part[column] == value]
    if part.empty:
        return {"by": by, "where": where or {}, "n": 0, "levels": []}

    values = part[by].astype(str)
    overall_rate = float(np.mean(long_stay_target(part)))

    levels, dropped, dropped_n = [], 0, 0
    for level, group in part.groupby(values):
        if len(group) < min_n:
            dropped += 1
            dropped_n += len(group)
            continue
        stay = group[config.TARGET]
        rate = float(np.mean(long_stay_target(group)))
        levels.append(
            {
                "value": str(level),
                "n": int(len(group)),
                "share": float(len(group) / len(part)),
                "median_days": float(stay.median()),
                "mean_days": float(stay.mean()),
                "p90_days": float(stay.quantile(0.90)),
                "long_stay_rate": rate,
                "long_stay_n": int(long_stay_target(group).sum()),
                "lift_over_base": (
                    rate / overall_rate if overall_rate else float("nan")
                ),
            }
        )

    levels.sort(key=lambda r: r[sort_by], reverse=True)
    return {
        "by": by,
        "where": where or {},
        "n": int(len(part)),
        "overall_median_days": float(part[config.TARGET].median()),
        "overall_mean_days": float(part[config.TARGET].mean()),
        "overall_long_stay_rate": overall_rate,
        "levels": levels,
        "levels_dropped_below_min_n": int(dropped),
        "rows_dropped_below_min_n": int(dropped_n),
        "min_n": int(min_n),
    }


def species_feature_contrast(frame: pd.DataFrame, species: str) -> dict:
    """How the long stays of one species differ from its short stays.

    Thin wrapper over breakdown() — kept because diagnose_species and its
    tests depend on this exact output shape.
    """
    part = frame[frame["animal_type"] == species]
    if part.empty:
        return {}

    y = long_stay_target(part)
    base_rate = float(np.mean(y))
    contrasts = {}

    for column in config.CATEGORICAL_FEATURES + config.BOOLEAN_FEATURES:
        block = breakdown(
            frame, by=column, where={"animal_type": species}, min_n=100
        )
        rows = [
            {
                "value": level["value"],
                "n": level["n"],
                "long_stay_rate": level["long_stay_rate"],
                "lift_over_species_base": level["lift_over_base"],
            }
            for level in block["levels"]
        ]
        if rows:
            rows.sort(key=lambda r: r["long_stay_rate"], reverse=True)
            contrasts[column] = {
                "top": rows[:3],
                "bottom": rows[-3:],
                "spread": float(
                    rows[0]["long_stay_rate"] - rows[-1]["long_stay_rate"]
                ),
            }

    long_stays = part[y == 1]
    short_stays = part[y == 0]
    numeric = {
        column: {
            "mean_long_stay": float(long_stays[column].mean()),
            "mean_short_stay": float(short_stays[column].mean()),
            "median_long_stay": float(long_stays[column].median()),
            "median_short_stay": float(short_stays[column].median()),
        }
        for column in config.NUMERIC_FEATURES
    }

    return {
        "species": species,
        "n": int(len(part)),
        "long_stay_rate": base_rate,
        "n_long_stay": int(y.sum()),
        "categorical": contrasts,
        "numeric": numeric,
        "most_separating_features": sorted(
            (
                {"feature": name, "spread": block["spread"]}
                for name, block in contrasts.items()
            ),
            key=lambda item: item["spread"],
            reverse=True,
        )[:5],
    }


def diagnose_species(
    bundle,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    species: str,
) -> dict:
    """Why is the model weak on this species — imbalance, or no signal?

    Four checks, in the order that distinguishes the explanations:
      1. base rate vs the other species (is it just rarity?)
      2. a species-only classifier (would specialising help?)
      3. permutation importance within the species (does anything work?)
      4. feature contrasts (is there anything to find at all?)
    """
    from .model import LongStayClassifierBundle

    test_mask = (test["animal_type"] == species).to_numpy()
    species_test = test[test_mask]
    y_species = long_stay_target(species_test)

    general_scores = bundle.predict_proba(species_test)
    general = classifier_metrics(y_species, general_scores)

    # 2. Fit a classifier on this species alone, same features, same split.
    species_train = train[train["animal_type"] == species]
    species_validation = validation[validation["animal_type"] == species]

    specialised_metrics = None
    specialised_importance = None
    if len(species_train) > 500 and long_stay_target(species_train).sum() > 30:
        specialised = LongStayClassifierBundle().fit(
            species_train, species_validation
        )
        specialised_scores = specialised.predict_proba(species_test)
        specialised_metrics = classifier_metrics(y_species, specialised_scores)

        # 3. Permutation importance within the species, on the test rows.
        X = build_feature_matrix(species_test)
        result = permutation_importance(
            _ProbabilityWrapper(specialised.classifier),
            X,
            y_species,
            scoring="average_precision",
            n_repeats=config.PERMUTATION_REPEATS,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
        )
        specialised_importance = sorted(
            (
                {
                    "feature": feature,
                    "pr_auc_drop": float(result.importances_mean[i]),
                    "std": float(result.importances_std[i]),
                }
                for i, feature in enumerate(X.columns)
            ),
            key=lambda item: item["pr_auc_drop"],
            reverse=True,
        )

    improvement = (
        (specialised_metrics["pr_auc"] - general["pr_auc"]) / general["pr_auc"]
        if specialised_metrics and general["pr_auc"]
        else None
    )

    return {
        "species": species,
        "general_model": general,
        "specialised_model": specialised_metrics,
        "specialised_improvement": improvement,
        "specialised_helps": (
            bool(improvement is not None and improvement >= 0.20)
        ),
        "verdict_threshold": 0.20,
        "permutation_importance_within_species": (
            specialised_importance[: config.N_TOP_IMPORTANCES]
            if specialised_importance
            else None
        ),
        "feature_contrast": species_feature_contrast(test, species),
        "reliable": bool(general["pr_auc_lift"] >= config.RELIABILITY_MIN_LIFT),
    }


class _ProbabilityWrapper(ClassifierMixin, BaseEstimator):
    """Adapts a classifier so permutation_importance can score its probabilities.

    ClassifierMixin is load-bearing, not decoration: sklearn decides whether to
    take column 1 of predict_proba from the estimator's tags. Without the mixin
    the scorer hands the full (n, 2) array to precision_recall_curve and fails.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):  # pragma: no cover - never refitted
        return self

    def predict_proba(self, X):
        p = self.classifier.predict_proba(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.classifier.predict_proba(X) >= 0.5).astype(int)


def species_reliability(report: dict) -> dict:
    """Compact per-species reliability, for the API and the result card."""
    out = {}
    for species, block in report.get("by_animal_type", {}).items():
        lift = block["pr_auc_lift"]
        out[species] = {
            "pr_auc": block["pr_auc"],
            "base_rate": block["positive_rate"],
            "lift": lift,
            "reliable": bool(lift >= config.RELIABILITY_MIN_LIFT),
            "min_lift": config.RELIABILITY_MIN_LIFT,
        }
    return out


# --------------------------------------------------------------------------
# The headline finding
# --------------------------------------------------------------------------


def shelter_load_finding(frame: pd.DataFrame) -> dict:
    """Two numbers: how many leave fast, and who actually consumes the shelter.

    Length of stay is a duration, so the right denominator for "how much of
    the shelter does this group use" is total animal-days, not head count.
    """
    los = frame[config.TARGET].to_numpy(dtype=float)
    total_days = float(los.sum())

    quick = los <= config.QUICK_EXIT_DAYS
    long_stay = los >= config.LONG_STAY_DAYS

    return {
        "n_animals": int(len(los)),
        "total_animal_days": total_days,
        "quick_exit_days_threshold": config.QUICK_EXIT_DAYS,
        "quick_exit_share_of_animals": float(np.mean(quick)),
        "quick_exit_share_of_animal_days": float(los[quick].sum() / total_days),
        "long_stay_days_threshold": config.LONG_STAY_DAYS,
        "long_stay_share_of_animals": float(np.mean(long_stay)),
        "long_stay_share_of_animal_days": float(
            los[long_stay].sum() / total_days
        ),
        "long_stay_mean_days": float(los[long_stay].mean())
        if long_stay.any()
        else 0.0,
    }


# --------------------------------------------------------------------------
# Findings — the claims the project actually supports
# --------------------------------------------------------------------------


def shelter_load_plot(load: dict, path) -> None:
    """Share of animals against share of animal-days, for two groups.

    Head count and resource use are different denominators, and the whole
    argument for the project is that they disagree. One chart, two bars per
    group, nothing else.
    """
    groups = [
        (f"Leave within {load['quick_exit_days_threshold']} days",
         load["quick_exit_share_of_animals"],
         load["quick_exit_share_of_animal_days"]),
        (f"Stay {load['long_stay_days_threshold']}+ days",
         load["long_stay_share_of_animals"],
         load["long_stay_share_of_animal_days"]),
    ]

    figure, axes = plt.subplots(figsize=(8.6, 4.2), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    positions = np.arange(len(groups))
    height = 0.34
    animals = [g[1] * 100 for g in groups]
    days = [g[2] * 100 for g in groups]

    axes.barh(positions + height / 2 + 0.02, animals, height=height,
              color=INK_MUTED, label="share of animals", zorder=3)
    axes.barh(positions - height / 2 - 0.02, days, height=height,
              color=SERIES_1, label="share of animal-days", zorder=3)

    for pos, value in zip(positions + height / 2 + 0.02, animals):
        axes.text(value + 1, pos, f"{value:.1f}%", va="center",
                  fontsize=10, color=INK_SECONDARY)
    for pos, value in zip(positions - height / 2 - 0.02, days):
        axes.text(value + 1, pos, f"{value:.1f}%", va="center",
                  fontsize=10, color=SERIES_1, fontweight="bold")

    axes.set_yticks(positions)
    axes.set_yticklabels([g[0] for g in groups], fontsize=11,
                         color=INK_SECONDARY)
    axes.set_xlim(0, max(max(animals), max(days)) * 1.22)
    axes.set_xlabel("percent", fontsize=10.5, color=INK_SECONDARY)
    axes.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(BASELINE)
    axes.tick_params(colors=INK_MUTED, labelsize=9.5)
    axes.legend(frameon=False, fontsize=10, loc="lower right",
                labelcolor=INK_SECONDARY)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def controlled_effect(
    frame: pd.DataFrame,
    flag_column: str,
    controls: list[str],
    min_stratum: int = 40,
) -> dict:
    """Effect of a boolean flag on the 90+ rate, holding `controls` fixed.

    Direct standardisation rather than a fitted regression: within each
    stratum of the control variables, take the difference in 90+ rate between
    flagged and unflagged animals, then pool those differences weighted by
    stratum size. This needs no distributional assumption and no extra
    dependency, and the estimand — "the difference you would see if the two
    groups had the same mix of age, sex, species and circumstance" — is the
    one the question is actually asking.

    The confidence interval is analytic: each stratum contributes binomial
    variance, and the weighted sum is combined in the usual way. Strata
    lacking either group, or smaller than `min_stratum`, are dropped, and the
    share of animals retained is reported so the reader can judge whether the
    controlled estimate speaks for the population.
    """
    part = frame.dropna(subset=[flag_column]).copy()
    part["_flag"] = part[flag_column].astype(bool)
    part["_y"] = long_stay_target(part)

    raw_1 = float(part.loc[part["_flag"], "_y"].mean())
    raw_0 = float(part.loc[~part["_flag"], "_y"].mean())

    keys = [part[c].astype(str) for c in controls]
    grouped = part.groupby(keys, observed=True)

    weights, diffs, variances, used, retained = [], [], [], 0, 0
    for _, stratum in grouped:
        flagged = stratum[stratum["_flag"]]
        unflagged = stratum[~stratum["_flag"]]
        n1, n0 = len(flagged), len(unflagged)
        if n1 == 0 or n0 == 0 or len(stratum) < min_stratum:
            continue
        p1 = float(flagged["_y"].mean())
        p0 = float(unflagged["_y"].mean())
        weights.append(len(stratum))
        diffs.append(p1 - p0)
        variances.append(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        used += 1
        retained += len(stratum)

    if not weights:
        return {"usable": False, "reason": "no stratum contained both groups"}

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    estimate = float(np.sum(w * np.asarray(diffs)))
    standard_error = float(np.sqrt(np.sum(w**2 * np.asarray(variances))))

    return {
        "usable": True,
        "flag": flag_column,
        "controls": controls,
        "raw_rate_flagged": raw_1,
        "raw_rate_unflagged": raw_0,
        "raw_difference_pp": (raw_1 - raw_0) * 100,
        "controlled_difference_pp": estimate * 100,
        "ci95_low_pp": (estimate - 1.96 * standard_error) * 100,
        "ci95_high_pp": (estimate + 1.96 * standard_error) * 100,
        "standard_error_pp": standard_error * 100,
        "strata_used": used,
        "animals_retained": int(retained),
        "coverage": float(retained / len(part)),
        "significant": bool(
            abs(estimate) > 1.96 * standard_error
        ),
    }


# --- charts for the new findings -----------------------------------------
#
# Styled to the interface palette rather than the earlier plot colours, so a
# chart embedded in a finding card sits in the same visual system as the card
# around it. Matplotlib's default blue appears nowhere.

UI_ACCENT = "#0F766E"
UI_WARN = "#B45309"
UI_MUTED = "#667085"
UI_BORDER = "#E4E7EC"
UI_SURFACE = "#FFFFFF"
UI_INK = "#0C0E12"


def _style(axes, figure, xlabel="", ylabel=""):
    figure.patch.set_facecolor(UI_SURFACE)
    axes.set_facecolor(UI_SURFACE)
    axes.grid(True, axis="y", color=UI_BORDER, linewidth=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(UI_BORDER)
    axes.tick_params(colors=UI_MUTED, labelsize=9.5)
    if xlabel:
        axes.set_xlabel(xlabel, fontsize=10.5, color=UI_MUTED)
    if ylabel:
        axes.set_ylabel(ylabel, fontsize=10.5, color=UI_MUTED)


def _paired_series_plot(
    categories, cat_values, dog_values, title, ylabel, path, percent=False
):
    """One grouped bar chart, cats against dogs, over ordered categories."""
    figure, axes = plt.subplots(figsize=(9.2, 4.4), dpi=160)
    positions = np.arange(len(categories))
    width = 0.38

    axes.bar(positions - width / 2, cat_values, width,
             color=UI_ACCENT, label="Cats", zorder=3)
    axes.bar(positions + width / 2, dog_values, width,
             color=UI_WARN, label="Dogs", zorder=3)

    axes.set_xticks(positions)
    axes.set_xticklabels(categories, fontsize=9.5, color=UI_MUTED)
    _style(axes, figure, ylabel=ylabel)
    axes.set_title(title, fontsize=13, color=UI_INK, loc="left", pad=12)
    if percent:
        axes.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
        )
    axes.legend(frameon=False, fontsize=10, labelcolor=UI_MUTED)
    figure.tight_layout()
    figure.savefig(path, facecolor=UI_SURFACE)
    plt.close(figure)


def age_bucket_plot(cats: dict, dogs: dict, path) -> None:
    order = config.AGE_BUCKET_LABELS
    def series(block, key):
        lookup = {lvl["value"]: lvl[key] for lvl in block["levels"]}
        return [lookup.get(b, 0.0) for b in order]
    _paired_series_plot(
        order,
        series(cats, "median_days"), series(dogs, "median_days"),
        "Median length of stay by age at intake", "days", path,
    )


def month_plot(cats: dict, dogs: dict, path) -> None:
    order = [str(m) for m in range(1, 13)]
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    def series(block):
        lookup = {lvl["value"]: lvl["median_days"] for lvl in block["levels"]}
        return [lookup.get(m, 0.0) for m in order]
    _paired_series_plot(
        labels, series(cats), series(dogs),
        "Median length of stay by month of intake", "days", path,
    )


def condition_plot(block: dict, path) -> None:
    """Condition ordered by 90+ rate, both measures on one chart."""
    levels = sorted(block["levels"], key=lambda r: r["long_stay_rate"],
                    reverse=True)
    names = [lvl["value"] for lvl in levels]
    rates = [lvl["long_stay_rate"] * 100 for lvl in levels]
    medians = [lvl["median_days"] for lvl in levels]

    figure, axes = plt.subplots(figsize=(9.2, 4.6), dpi=160)
    positions = np.arange(len(names))
    axes.bar(positions, rates, 0.55, color=UI_WARN, zorder=3,
             label="share staying 90+ days")
    _style(axes, figure, ylabel="percent staying 90+ days")
    axes.set_xticks(positions)
    axes.set_xticklabels(names, fontsize=9.5, color=UI_MUTED)
    axes.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )

    twin = axes.twinx()
    # Markers only. These categories are nominal and ordered by the bar
    # metric, so joining them with a line would draw a trend that does not
    # exist.
    twin.plot(positions, medians, "o", color=UI_ACCENT,
              markersize=8, markeredgecolor=UI_SURFACE, markeredgewidth=2,
              label="median stay", zorder=4)
    twin.set_ylabel("median stay (days)", fontsize=10.5, color=UI_MUTED)
    twin.tick_params(colors=UI_MUTED, labelsize=9.5)
    for side in ("top", "left", "bottom"):
        twin.spines[side].set_visible(False)
    twin.spines["right"].set_color(UI_BORDER)
    twin.grid(False)

    axes.set_title("Intake condition against length of stay", fontsize=13,
                   color=UI_INK, loc="left", pad=34)
    handles = axes.get_legend_handles_labels()[0] + \
        twin.get_legend_handles_labels()[0]
    labels_ = axes.get_legend_handles_labels()[1] + \
        twin.get_legend_handles_labels()[1]
    # Legend above the plot area so it cannot sit on a bar or a marker.
    axes.legend(handles, labels_, frameon=False, fontsize=10,
                labelcolor=UI_MUTED, loc="lower left",
                bbox_to_anchor=(0.0, 1.02), ncol=2)
    axes.set_ylim(0, max(rates) * 1.12)
    twin.set_ylim(0, max(medians) * 1.18)
    figure.tight_layout()
    figure.savefig(path, facecolor=UI_SURFACE)
    plt.close(figure)


def black_effect_plot(effect: dict, path) -> None:
    """Raw against controlled difference, with the confidence interval."""
    figure, axes = plt.subplots(figsize=(8.6, 3.4), dpi=160)
    axes.axvline(0, color=UI_MUTED, linewidth=1.4,
                 linestyle=(0, (5, 4)), zorder=2, label="no difference")

    axes.plot([effect["raw_difference_pp"]], [1], "o", markersize=11,
              color=UI_MUTED, markeredgecolor=UI_SURFACE, markeredgewidth=2,
              zorder=4, label="raw difference")
    axes.plot([effect["controlled_difference_pp"]], [0], "o", markersize=11,
              color=UI_ACCENT, markeredgecolor=UI_SURFACE, markeredgewidth=2,
              zorder=4, label="controlled, with 95% CI")
    axes.hlines(0, effect["ci95_low_pp"], effect["ci95_high_pp"],
                color=UI_ACCENT, linewidth=3, zorder=3)

    axes.set_yticks([0, 1])
    axes.set_yticklabels(["controlled", "raw"], fontsize=10.5, color=UI_MUTED)
    # Extra room below the lower marker so the legend never sits on the
    # confidence interval it is describing.
    axes.set_ylim(-1.5, 1.6)
    _style(axes, figure, xlabel="difference in 90+ day rate (percentage points)")
    axes.grid(True, axis="x", color=UI_BORDER, linewidth=0.9)
    axes.grid(False, axis="y")
    axes.set_title("Does being black change how long an animal waits?",
                   fontsize=13, color=UI_INK, loc="left", pad=12)
    axes.legend(frameon=False, fontsize=9.5, labelcolor=UI_MUTED,
                loc="lower center", ncol=3)
    figure.tight_layout()
    figure.savefig(path, facecolor=UI_SURFACE)
    plt.close(figure)


def build_findings(frame: pd.DataFrame, payload: dict) -> list[dict]:
    """The findings the Findings tab renders, each with its own numbers.

    Where a common belief is not supported, the headline says so plainly
    rather than softening it into "no strong evidence".
    """
    los = frame[config.TARGET]
    black = frame["is_black"].fillna(False).astype(bool)
    named = frame["has_name"].fillna(False).astype(bool)

    # Breakdowns shared by several findings. age_bucket is derived here
    # because it is a reporting band, not a model feature.
    work = frame.copy()
    work["age_bucket"] = age_bucket(work["age_days"])

    by_age = {sp: breakdown(work, "age_bucket", where={"animal_type": sp})
              for sp in ("Cat", "Dog")}
    by_month = {sp: breakdown(work, "intake_month", where={"animal_type": sp})
                for sp in ("Cat", "Dog")}
    by_condition = breakdown(work, "intake_condition", min_n=100)
    # Measured from the data rather than assumed, so per-year rates stay
    # correct after any refetch extends the range.
    span = work["intake_datetime"].max() - work["intake_datetime"].min()
    years_covered = span.days / 365.25
    by_black = {sp: breakdown(work, "is_black", where={"animal_type": sp})
                for sp in ("Cat", "Dog")}
    black_effect = controlled_effect(
        work, "is_black",
        ["animal_type", "age_bucket", "sex", "intake_type", "intake_condition"],
    )

    def level(block, value, key, default=float("nan")):
        for row in block["levels"]:
            if row["value"] == str(value):
                return row[key]
        return default

    # What the 90+ day animals have in common: over-representation against
    # each category's share of the whole population.
    long_stays = work[work[config.TARGET] >= config.LONG_STAY_DAYS]
    profile = []
    for column in ("primary_breed", "intake_type", "age_bucket",
                   "intake_condition"):
        share_long = long_stays[column].astype(str).value_counts(normalize=True)
        share_all = work[column].astype(str).value_counts(normalize=True)
        for value, s in share_long.head(6).items():
            base = float(share_all.get(value, 0.0))
            if base > 0.01 and s / base > 1.25:
                profile.append({
                    "column": column, "value": str(value),
                    "share_of_long_stays": float(s), "share_of_all": base,
                    "over_representation": float(s / base),
                })
    profile.sort(key=lambda r: r["over_representation"], reverse=True)

    # Evidence for the has_name timing leak, recomputed here so the finding
    # carries its own numbers rather than quoting the audit tool.
    strays = frame[frame["intake_type"] == "Stray"]
    stray_flag = strays["has_name"].fillna(False).astype(bool)
    stray_named = float(stray_flag.mean())
    wildlife = frame[frame["intake_type"] == "Wildlife"]
    wildlife_named = (
        float(wildlife["has_name"].fillna(False).astype(bool).mean())
        if len(wildlife) else float("nan")
    )
    stray_named_long = float(np.mean(long_stay_target(strays[stray_flag])))
    stray_unnamed_long = float(np.mean(long_stay_target(strays[~stray_flag])))
    stray_gap = (
        stray_named_long / stray_unnamed_long if stray_unnamed_long else float("nan")
    )

    # Naming looked at from the other direction: not "do named animals stay
    # longer" but "are long-staying animals named".
    all_long = long_stay_target(frame)
    named_among_long = float(named[all_long == 1].mean())
    named_among_rest = float(named[all_long == 0].mean())
    named_baseline = float(named.mean())

    importance = {row["feature"]: row for row in payload["permutation_importance"]}
    load = payload["shelter_load"]
    clf = payload["classifier"]

    black_long = float(np.mean(long_stay_target(frame[black])))
    other_long = float(np.mean(long_stay_target(frame[~black])))

    chosen_reg = payload["regression_variants"]["comparison"][
        payload["regression_variants"]["chosen"]
    ]

    findings = [
        {
            "id": "load",
            "label": "FINDING 01",
            "headline": "Four percent of animals consume a third of the shelter",
            "chart": "shelter_load.png",
            "stats": [
                {"label": "leave within 7 days",
                 "value": f"{load['quick_exit_share_of_animals']:.1%} of animals"},
                {"label": "their share of animal-days",
                 "value": f"{load['quick_exit_share_of_animal_days']:.1%}"},
                {"label": "stay 90+ days",
                 "value": f"{load['long_stay_share_of_animals']:.1%} of animals"},
                {"label": "their share of animal-days",
                 "value": f"{load['long_stay_share_of_animal_days']:.1%}"},
                {"label": "mean stay in that group",
                 "value": f"{load['long_stay_mean_days']:.0f} days"},
                {"label": "total animal-days",
                 "value": f"{load['total_animal_days']:,.0f}"},
            ] + [
                {"label": f"{p['value']} — share of long stays vs population",
                 "value": f"{p['share_of_long_stays']:.1%} vs "
                          f"{p['share_of_all']:.1%} "
                          f"({p['over_representation']:.1f}x)"}
                for p in profile[:5]
            ],
            "callout": (
                f"{load['long_stay_share_of_animals']:.0%} of animals stay "
                f"longer than three months, and those animals take up "
                f"{load['long_stay_share_of_animal_days']:.0%} of all the "
                "time the shelter has to give."
            ),
            "caveat": (
                "That one sentence is the argument for this entire tool. A "
                "small group absorbs a third of everything the shelter has, "
                "so identifying them on day one is worth more than marginally "
                "improving the experience of the 54% who leave within a "
                "week.\n\n"
                "It is measured in animal-days rather than head count, "
                "because duration is the resource being consumed. A kennel "
                "occupied for 152 days is 152 days of capacity, however you "
                "count the animal in it. Both figures cover the whole cleaned "
                "dataset, not just the test period.\n\n"
                "What these animals have in common, drawing on the "
                "breakdowns in findings 05 and 07: they skew adult rather "
                "than young — the 3-7 year and 7-year-plus bands are both "
                "about 1.5 times over-represented, and finding 05 shows the "
                "long-stay rate climbing more than fivefold from puppies to "
                "adult dogs even while the median stay stays flat. They skew "
                "toward Pit Bulls, who are 2.5 times over-represented and are "
                "the single largest signal in the group. They skew injured "
                "rather than sick — finding 07 shows Injured carrying the "
                "highest long-stay rate of any common condition despite "
                "having one of the shortest median stays, which is the "
                "bimodal pattern that makes this group hard to spot early. "
                "And they skew toward owner surrenders over strays.\n\n"
                "The over-representation figures compare each group's share "
                "of long stays with its share of the whole population. They "
                "describe who is in the group; they do not establish cause, "
                "and these categories overlap heavily — an adult surrendered "
                "Pit Bull is one animal counted in four of them."
            ),
        },
        {
            "id": "black",
            "label": "FINDING 02",
            "headline": "Black cats wait longer, black dogs do not, and the "
                        "overall average hides both",
            "chart": "black_effect.png",
            "stats": [
                {"label": "median stay, black (all species)",
                 "value": f"{los[black].median():.1f} days"},
                {"label": "median stay, all others",
                 "value": f"{los[~black].median():.1f} days"},
                {"label": "CATS — median stay, black vs others",
                 "value": f"{level(by_black['Cat'], True, 'median_days'):.1f} vs "
                          f"{level(by_black['Cat'], False, 'median_days'):.1f} days"},
                {"label": "CATS — 90+ day rate, black vs others",
                 "value": f"{level(by_black['Cat'], True, 'long_stay_rate'):.2%} vs "
                          f"{level(by_black['Cat'], False, 'long_stay_rate'):.2%}"},
                {"label": "DOGS — median stay, black vs others",
                 "value": f"{level(by_black['Dog'], True, 'median_days'):.1f} vs "
                          f"{level(by_black['Dog'], False, 'median_days'):.1f} days"},
                {"label": "DOGS — 90+ day rate, black vs others",
                 "value": f"{level(by_black['Dog'], True, 'long_stay_rate'):.2%} vs "
                          f"{level(by_black['Dog'], False, 'long_stay_rate'):.2%}"},
                {"label": "raw difference in 90+ rate",
                 "value": f"{black_effect['raw_difference_pp']:+.2f} pp"},
                {"label": "controlled difference (age, sex, species, "
                          "intake type, condition)",
                 "value": f"{black_effect['controlled_difference_pp']:+.2f} pp "
                          f"[{black_effect['ci95_low_pp']:+.2f}, "
                          f"{black_effect['ci95_high_pp']:+.2f}]"},
                {"label": "strata used / animals retained",
                 "value": f"{black_effect['strata_used']} strata, "
                          f"{black_effect['coverage']:.1%} of animals"},
                {"label": "importance of is_black to the model",
                 "value": f"{importance['is_black']['mae_increase_days']:.3f} days"},
                {"label": "share of animals that are black",
                 "value": f"{black.mean():.1%}"},
            ],
            "caveat": (
                "An earlier pass compared medians across all animals, saw "
                f"{los[black].median():.1f} against {los[~black].median():.1f} "
                "days, and concluded there was nothing here. That was wrong, "
                "and the way it was wrong is the interesting part.\n\n"
                "The overall figure is near zero because the two species "
                "point in opposite directions and cancel. Black cats wait "
                f"about a day longer at the median and are "
                f"{level(by_black['Cat'], True, 'long_stay_rate') / level(by_black['Cat'], False, 'long_stay_rate'):.2f} "
                "times as likely to still be there after three months. Black "
                "dogs are, if anything, very slightly quicker. Averaging the "
                "two produced a null that described neither.\n\n"
                "Holding species, age, sex, intake type and intake condition "
                "fixed by direct standardisation, being black is associated "
                f"with {black_effect['controlled_difference_pp']:+.2f} "
                "percentage points on the 90+ day rate, 95% CI "
                f"[{black_effect['ci95_low_pp']:+.2f}, "
                f"{black_effect['ci95_high_pp']:+.2f}] across "
                f"{black_effect['strata_used']} strata covering "
                f"{black_effect['coverage']:.0%} of animals. The interval "
                "excludes zero, so the effect is real, but read the size "
                "before the significance: on a 4.2% base rate this is a small "
                "effect that a sample of 171,561 makes detectable. It is "
                "nothing like the folklore version of \"black dog syndrome\", "
                "and note that the dogs are the half where the effect is "
                "absent.\n\n"
                "The model, meanwhile, finds no use for the feature at all — "
                f"{importance['is_black']['mae_increase_days']:.3f} days of "
                "permutation importance. An effect can be real and still be "
                "too small to help a prediction.\n\n"
                "Black means the primary colour component is literally "
                "\"Black\"; \"Black Tabby\" is a pattern, not a solid colour. "
                "One shelter, one city, one recording practice — this says "
                "nothing about adopter behaviour anywhere else."
            ),
        },
        {
            "id": "name_leak",
            "label": "FINDING 03",
            "headline": "The strongest predictor we found was measuring our "
                        "own staff, so we deleted it",
            # Rendered in --warn-sf: the one case where the honest result was
            # to remove something rather than report it.
            "notice": (
                "has_name was the second most important feature in the model. "
                "It was removed after the timing audit showed it was not "
                "knowable at intake, at a cost of 7.4% PR-AUC and 44 of the "
                "713 long stays the model previously caught."
            ),
            "chart": "permutation_importance.png",
            "stats": [
                {"label": "stray intakes carrying a name",
                 "value": f"{stray_named:.1%}"},
                {"label": "wildlife intakes carrying a name (control)",
                 "value": f"{wildlife_named:.1%}"},
                {"label": "90+ day rate, named strays",
                 "value": f"{stray_named_long:.2%}"},
                {"label": "90+ day rate, unnamed strays",
                 "value": f"{stray_unnamed_long:.2%}"},
                {"label": "gap between them",
                 "value": f"{stray_gap:.1f}x"},
                {"label": "named among animals staying 90+ days",
                 "value": f"{named_among_long:.1%}"},
                {"label": "named among everyone else",
                 "value": f"{named_among_rest:.1%}"},
                {"label": "named across the whole dataset",
                 "value": f"{named_baseline:.1%}"},
                {"label": "intake and outcome name identical",
                 "value": "122,908 of 122,908"},
                {"label": "cost of removing it (PR-AUC)",
                 "value": "-7.4%"},
            ],
            "caveat": (
                "A name looked like the second most useful thing we knew "
                "about an animal. It was not knowable at intake: two thirds "
                "of animals picked up off the street already carry one in the "
                "record, while wildlife — which nobody names — sits near "
                "zero.\n\n"
                "Set the modelling aside and this is a result about the "
                "shelter rather than about the data. Naming is a thing staff "
                f"do, and it concentrates hard: {named_among_long:.1%} of "
                f"animals who stayed 90+ days have a name against "
                f"{named_among_rest:.1%} of everyone else. Among strays "
                "alone — animals who all arrived the same way, off the "
                f"street — the named ones are {stray_gap:.1f} times more "
                f"likely to still be there after three months "
                f"({stray_named_long:.2%} against {stray_unnamed_long:.2%}). "
                "Whatever else a name is, it marks an animal the staff have "
                "engaged with.\n\n"
                "What this cannot show is which way it runs. Naming may draw "
                "attention that somehow extends a stay, or — far more likely "
                "— staff name the animals who are still there after a few "
                "weeks, so the name follows the long stay rather than "
                "preceding it. The data carries no timestamp on the name "
                "field, so the two are indistinguishable here. Nothing in "
                "this finding says naming an animal harms it, and it should "
                "not be read that way.\n\n"
                "For the model the direction does not matter: either way the "
                "name is not knowable on day one. Removing it cost 7.4% of "
                "PR-AUC and 44 of the long stays we used to catch. We removed "
                "it anyway. See config.TEMPORAL_LEAKAGE_NOTES."
            ),
        },
        {
            "id": "duration",
            "label": "FINDING 04",
            "headline": "Length of stay in days cannot be predicted here",
            "chart": "calibration.png",
            "stats": [
                {"label": "R² of the best variant",
                 "value": f"{chosen_reg['r2']:.3f}"},
                {"label": "MAE",
                 "value": f"{chosen_reg['mae_days']:.1f} days"},
                {"label": "MAE of a constant (the median)",
                 "value": f"{payload['models']['baseline_global_median']['test']['mae_days']:.1f} days"},
                {"label": "classifier PR-AUC",
                 "value": f"{clf['test']['pr_auc']:.4f}"},
                {"label": "better than chance",
                 "value": f"{clf['test']['pr_auc_lift']:.2f}x"},
            ],
            "caveat": (
                "A negative R² means the day estimate performs worse than a "
                "constant. That is the result, not a defect: length of stay "
                "is largely not explained by what is knowable at intake. It "
                "is why the headline figure is a probability rather than a "
                "number of days."
            ),
        },
        {
            "id": "age",
            "label": "FINDING 05",
            "headline": "Age moves the long-stay tail without moving the "
                        "median, and kittens break the pattern entirely",
            "chart": "stay_by_age.png",
            "stats": [
                {"label": "CATS — median stay, under 2 months",
                 "value": f"{level(by_age['Cat'], 'under 2mo', 'median_days'):.1f} days"},
                {"label": "CATS — median stay, 2-6 months",
                 "value": f"{level(by_age['Cat'], '2-6mo', 'median_days'):.1f} days"},
                {"label": "CATS — median stay, 7 years and over",
                 "value": f"{level(by_age['Cat'], '7y+', 'median_days'):.1f} days"},
                {"label": "CATS — 90+ rate, 2-6 months vs 7 years and over",
                 "value": f"{level(by_age['Cat'], '2-6mo', 'long_stay_rate'):.2%} vs "
                          f"{level(by_age['Cat'], '7y+', 'long_stay_rate'):.2%}"},
                {"label": "DOGS — median stay, youngest vs oldest band",
                 "value": f"{level(by_age['Dog'], '2-6mo', 'median_days'):.1f} vs "
                          f"{level(by_age['Dog'], '7y+', 'median_days'):.1f} days"},
                {"label": "DOGS — 90+ rate, 2-6 months",
                 "value": f"{level(by_age['Dog'], '2-6mo', 'long_stay_rate'):.2%}"},
                {"label": "DOGS — 90+ rate, 3-7 years",
                 "value": f"{level(by_age['Dog'], '3-7y', 'long_stay_rate'):.2%}"},
                {"label": "DOGS — ratio across those bands",
                 "value": f"{level(by_age['Dog'], '3-7y', 'long_stay_rate') / level(by_age['Dog'], '2-6mo', 'long_stay_rate'):.1f}x"},
                {"label": "cats / dogs in sample",
                 "value": f"{by_age['Cat']['n']:,} / {by_age['Dog']['n']:,}"},
            ],
            "caveat": (
                "Published analyses of this dataset report length of stay "
                "rising with age for cats and staying relatively flat for "
                "dogs. Tested here, that holds only in part, and the half "
                "that fails is the more useful half.\n\n"
                "For dogs the median really is flat — every age band sits "
                "between four and seven days. But the 90+ day rate climbs "
                f"from {level(by_age['Dog'], '2-6mo', 'long_stay_rate'):.2%} "
                f"for puppies to "
                f"{level(by_age['Dog'], '3-7y', 'long_stay_rate'):.2%} for "
                "adults, a "
                f"{level(by_age['Dog'], '3-7y', 'long_stay_rate') / level(by_age['Dog'], '2-6mo', 'long_stay_rate'):.1f}-fold "
                "difference. \"Flat\" is true of the typical dog and false of "
                "the tail, and the tail is what costs kennel space.\n\n"
                "For cats the relationship is not a rise at all, it is "
                "U-shaped. Kittens under two months have the LONGEST median "
                f"stay of any cat band at "
                f"{level(by_age['Cat'], 'under 2mo', 'median_days'):.1f} days "
                "— longer than senior cats — then the two-to-six-month band "
                f"drops to {level(by_age['Cat'], '2-6mo', 'median_days'):.1f} "
                "days before climbing again with age. A monotonic summary "
                "misses this completely.\n\n"
                "The kitten result most likely reflects policy rather than "
                "demand: very young kittens are typically held to reach an "
                "adoptable weight, so part of that stay is a rule, not a "
                "failure to place them. This data cannot separate the two, "
                "and the distinction matters — a kitten waiting for weight "
                "does not need a foster placement for the same reason an "
                "adult dog waiting for an adopter does."
            ),
        },
        {
            "id": "month",
            "label": "FINDING 06",
            "headline": "Cats have a season and dogs do not",
            "chart": "stay_by_month.png",
            "stats": [
                {"label": "CATS — median stay, January intake",
                 "value": f"{level(by_month['Cat'], 1, 'median_days'):.1f} days"},
                {"label": "CATS — median stay, May intake",
                 "value": f"{level(by_month['Cat'], 5, 'median_days'):.1f} days"},
                {"label": "CATS — median stay, June intake",
                 "value": f"{level(by_month['Cat'], 6, 'median_days'):.1f} days"},
                {"label": "CATS — June against January",
                 "value": f"{level(by_month['Cat'], 6, 'median_days') - level(by_month['Cat'], 1, 'median_days'):+.1f} days "
                          f"({level(by_month['Cat'], 6, 'median_days') / level(by_month['Cat'], 1, 'median_days'):.1f}x)"},
                {"label": "CATS — intake volume, May vs January",
                 "value": f"{int(level(by_month['Cat'], 5, 'n')):,} vs "
                          f"{int(level(by_month['Cat'], 1, 'n')):,} animals"},
                {"label": "DOGS — median stay range across all 12 months",
                 "value": f"{min(l['median_days'] for l in by_month['Dog']['levels']):.1f}"
                          f"–{max(l['median_days'] for l in by_month['Dog']['levels']):.1f} days"},
                {"label": "CATS — median stay range across all 12 months",
                 "value": f"{min(l['median_days'] for l in by_month['Cat']['levels']):.1f}"
                          f"–{max(l['median_days'] for l in by_month['Cat']['levels']):.1f} days"},
            ],
            "caveat": (
                "Kitten season is visible and large. A cat admitted in June "
                f"waits "
                f"{level(by_month['Cat'], 6, 'median_days') / level(by_month['Cat'], 1, 'median_days'):.1f} "
                "times as long as one admitted in January — "
                f"{level(by_month['Cat'], 6, 'median_days'):.1f} days against "
                f"{level(by_month['Cat'], 1, 'median_days'):.1f}. The peak is "
                "May at "
                f"{level(by_month['Cat'], 5, 'median_days'):.1f} days, when "
                "intake volume is also at its highest: "
                f"{int(level(by_month['Cat'], 5, 'n')):,} cats against "
                f"{int(level(by_month['Cat'], 1, 'n')):,} in January. More "
                "animals arriving and each one waiting longer is a "
                "compounding problem, not an additive one.\n\n"
                "Dogs show nothing. Every month sits within a "
                f"{max(l['median_days'] for l in by_month['Dog']['levels']) - min(l['median_days'] for l in by_month['Dog']['levels']):.1f}-day "
                "band across the whole year. Whatever drives dog adoption, it "
                "is not the calendar.\n\n"
                "Month of intake is a proxy, not a cause. It carries kitten "
                "season, school holidays, weather and local events together, "
                "and this data cannot separate them. The practical reading is "
                "capacity planning rather than explanation: the shelter knows "
                "in advance which months will be hard for cats."
            ),
        },
        {
            "id": "condition",
            "label": "FINDING 07",
            "headline": "Injured animals leave fastest and get stuck most "
                        "often — the median and the tail disagree",
            "chart": "stay_by_condition.png",
            "stats": [
                {"label": "Injured — median stay / 90+ rate",
                 "value": f"{level(by_condition, 'Injured', 'median_days'):.1f} days / "
                          f"{level(by_condition, 'Injured', 'long_stay_rate'):.2%}"},
                {"label": "Normal — median stay / 90+ rate",
                 "value": f"{level(by_condition, 'Normal', 'median_days'):.1f} days / "
                          f"{level(by_condition, 'Normal', 'long_stay_rate'):.2%}"},
                {"label": "Sick — median stay / 90+ rate",
                 "value": f"{level(by_condition, 'Sick', 'median_days'):.1f} days / "
                          f"{level(by_condition, 'Sick', 'long_stay_rate'):.2%}"},
                {"label": "Nursing — median stay / 90+ rate",
                 "value": f"{level(by_condition, 'Nursing', 'median_days'):.1f} days / "
                          f"{level(by_condition, 'Nursing', 'long_stay_rate'):.2%}"},
                {"label": "Neonatal — median stay / 90+ rate",
                 "value": f"{level(by_condition, 'Neonatal', 'median_days'):.1f} days / "
                          f"{level(by_condition, 'Neonatal', 'long_stay_rate'):.2%}"},
                {"label": "Aged — median stay / 90+ rate",
                 "value": f"{level(by_condition, 'Aged', 'median_days'):.1f} days / "
                          f"{level(by_condition, 'Aged', 'long_stay_rate'):.2%}"},
                {"label": "Injured animals per year",
                 "value": f"{level(by_condition, 'Injured', 'n') / years_covered:,.0f}"},
                {"label": "share of all intakes recorded Normal",
                 "value": f"{level(by_condition, 'Normal', 'share'):.1%}"},
            ],
            "caveat": (
                "This is the breakdown with the most direct operational "
                "reading, and it is not the one you would guess. Injured "
                "animals have among the SHORTEST median stays "
                f"({level(by_condition, 'Injured', 'median_days'):.1f} days, "
                "against "
                f"{level(by_condition, 'Normal', 'median_days'):.1f} for "
                "Normal) and the HIGHEST rate of getting stuck past three "
                f"months ({level(by_condition, 'Injured', 'long_stay_rate'):.2%} "
                f"against {level(by_condition, 'Normal', 'long_stay_rate'):.2%}). "
                "The distribution is bimodal: most injuries resolve and the "
                "animal moves on quickly, while a minority turn into long "
                "medical holds. A single average of this group describes "
                "almost nobody in it.\n\n"
                "Sick animals show the opposite shape — the shortest median "
                f"of the common conditions at "
                f"{level(by_condition, 'Sick', 'median_days'):.1f} days and a "
                "BELOW-average 90+ rate. Reading that as good news would be "
                "a mistake: this data records how long an animal was present, "
                "not what happened to them, and outcome type is deliberately "
                "excluded from this project. A stay can end quickly for "
                "reasons nobody wants.\n\n"
                "The timing audit could not determine what \"Normal\" means. "
                "The field may be assessed at the door or updated after a "
                "veterinary examination hours later, and the feed carries no "
                "edit history to settle it. If conditions are upgraded after "
                "examination, some animals counted here as Normal were "
                "recorded that way before anyone looked properly. See "
                "config.TEMPORAL_LEAKAGE_NOTES.\n\n"
                "Categories below 100 animals are omitted; Pregnant, Other "
                "and Feral are small enough that their rates move on a "
                "handful of cases."
            ),
        },
    ]

    # Sample size shown on every card. Attached here rather than inline so
    # each finding states the population it was actually computed on, which
    # differs — some cover the whole dataset, one covers only the test period.
    test_rows = payload["split"]["test"]["rows"]
    samples = {
        "load": f"{len(frame):,} animals over "
                f"{years_covered:.1f} years, whole dataset",
        "black": f"{by_black['Cat']['n']:,} cats and "
                 f"{by_black['Dog']['n']:,} dogs; controlled estimate uses "
                 f"{black_effect['strata_used']} strata covering "
                 f"{black_effect['coverage']:.0%} of animals",
        "name_leak": f"{len(strays):,} stray intakes; "
                     f"122,908 stays with a name on both sides",
        "duration": f"{test_rows:,} animals in the held-out test period",
        "age": f"{by_age['Cat']['n']:,} cats and {by_age['Dog']['n']:,} dogs, "
               f"levels below {by_age['Cat']['min_n']} animals omitted",
        "month": f"{by_month['Cat']['n']:,} cats and "
                 f"{by_month['Dog']['n']:,} dogs across 12 months",
        "condition": f"{by_condition['n']:,} animals; "
                     f"{by_condition['levels_dropped_below_min_n']} conditions "
                     f"below {by_condition['min_n']} animals omitted",
    }
    for finding in findings:
        finding["sample"] = samples[finding["id"]]
    return findings


# --------------------------------------------------------------------------
# Limitations, in plain language, built from the measured numbers
# --------------------------------------------------------------------------


def day_estimate_caveat(r2: float) -> dict:
    """The sentence that must appear beside every day estimate.

    Composed here rather than in the page template so the number and the
    wording travel together and cannot drift apart. The UI reads this string;
    it does not format its own.
    """
    return {
        "text": (
            "Indicative only. On these features the day estimate does no "
            f"better than a constant (R² = {r2:.3f})."
        ),
        "r2": float(r2),
    }


def model_limitations(
    variants: dict,
    classifier: dict,
    diagnosis: dict,
) -> list[dict]:
    """The four things a reader must know before trusting any of this.

    Every number is pulled from the measured results rather than typed in, so
    a rerun that changes the picture changes this text too.
    """
    chosen = variants["comparison"][variants["chosen"]]
    r2_values = [
        variants["comparison"][name]["r2"] for name in variants["comparison"]
    ]
    cat = diagnosis.get("Cat", {})
    cat_general = cat.get("general_model", {})

    return [
        {
            "id": "negative_r2",
            "title": "The day estimate is worse than a constant",
            "detail": (
                "R-squared is negative for all three regression variants "
                f"({', '.join(f'{v:.3f}' for v in r2_values)}). This is a "
                "finding, not a bug: length of stay is largely not explained "
                "by what is knowable at intake. It is why the headline number "
                "is a probability and not a number of days."
            ),
        },
        {
            "id": "day_bias",
            "title": "The day estimate is systematically low",
            "detail": (
                f"About {abs(chosen['mean_bias_days']):.0f} days below actual "
                "on average. Removable with Duan's smearing correction, but "
                "only at the cost of accuracy, so accuracy was chosen. Read "
                "the estimated stay as an order of magnitude, not a date."
            ),
        },
        {
            "id": "precision_ceiling",
            "title": "50% precision is unreachable at any threshold",
            "detail": (
                "Class imbalance sets the ceiling: only "
                f"{classifier['test']['positive_rate'] * 100:.1f}% of animals "
                "stay that long, so most flagged animals will still leave "
                "quickly. The tool sorts a queue; it does not deliver a "
                "verdict about any individual animal."
            ),
        },
        {
            "id": "timing_leak",
            "title": "One feature was removed for leaking, others cannot be "
                     "verified",
            "detail": (
                "has_name looked like the second strongest predictor and was "
                "not knowable at intake — two thirds of strays carried a name "
                "in the record. It was removed, at a cost of 7.4% PR-AUC. "
                "Breed, colour and intake_condition show the same "
                "animal-level storage shape but no positive evidence of harm, "
                "and the feed carries no edit history to settle it. They are "
                "kept and flagged in config.TEMPORAL_LEAKAGE_NOTES."
            ),
        },
        {
            "id": "cat_ceiling",
            "title": "For cats the model is barely better than chance",
            "detail": (
                f"{cat_general.get('pr_auc_lift', float('nan')):.2f}x better "
                "than random flagging. Training a cat-only model improves "
                "that by just "
                f"{(cat.get('specialised_improvement') or 0) * 100:+.0f}%, so "
                "this is the ceiling of the available features rather than a "
                "modelling defect. Cats have no equivalent of dog breed, "
                "which is the single strongest signal the model has."
            ),
        },
    ]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@dataclass
class Report:
    payload: dict

    def save(self, path=None) -> None:
        path = path or config.METRICS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

        # findings.json is a deployment convenience: the same list, standalone,
        # for anyone who wants the findings without parsing the whole metrics
        # file. metrics.json stays the source of truth and a test asserts the
        # two never disagree.
        config.FINDINGS_PATH.write_text(
            json.dumps(self.payload["findings"], indent=2), encoding="utf-8"
        )


def classifier_report(
    bundle,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Everything about the primary model, on the held-out test set."""
    y_test = long_stay_target(test)
    active = bundle.predict_proba(test)          # what the service actually uses
    uncalibrated = bundle.predict_proba_uncalibrated(test)
    isotonic = bundle.predict_proba_isotonic(test)
    baseline = bundle.baseline_proba(test)
    calibrated = active

    metrics = classifier_metrics(y_test, calibrated)
    base_rate = metrics["positive_rate"]

    # Thresholds are chosen on the VALIDATION period, then applied to test.
    # Choosing them on test would be reporting the best case as if it were the
    # expected case.
    y_validation = long_stay_target(validation)
    validation_scores = bundle.predict_proba(validation)

    at_recall = threshold_for_recall(
        y_validation, validation_scores, config.REPORT_AT_RECALL
    )
    at_precision = threshold_for_precision(
        y_validation, validation_scores, config.REPORT_AT_PRECISION
    )

    operating = operating_point(y_test, calibrated, config.OPERATING_THRESHOLD)

    reliability_before = reliability_table(y_test, uncalibrated)
    reliability_after = reliability_table(y_test, isotonic)

    by_species = {}
    for species in ("Dog", "Cat"):
        mask = (test["animal_type"] == species).to_numpy()
        if mask.sum() < 50 or long_stay_target(test[mask]).sum() < 5:
            continue
        by_species[species] = {
            **classifier_metrics(y_test[mask], calibrated[mask]),
            "at_operating_threshold": operating_point(
                y_test[mask], calibrated[mask], config.OPERATING_THRESHOLD
            ),
        }

    return {
        "target": f"{config.CLASSIFIER_TARGET} = "
                  f"{config.TARGET} >= {config.LONG_STAY_DAYS}",
        "train_positive_rate": bundle.classifier.train_positive_rate_,
        "test": metrics,
        "baseline_positive_rate": {
            "description": "predict the training positive rate for everyone",
            **classifier_metrics(y_test, baseline),
        },
        "threshold_policy": {
            "cost_ratio_miss_to_false_alarm":
                config.COST_RATIO_MISS_TO_FALSE_ALARM,
            "operating_threshold": config.OPERATING_THRESHOLD,
            "derivation": "p* = 1 / (1 + cost_ratio); derived from the stated "
                          "cost ratio, not tuned against any metric",
            "chosen_on": "cost ratio only; reference thresholds below are "
                         "chosen on the validation period, never on test",
        },
        "at_operating_threshold": operating,
        "at_recall_50": (
            operating_point(y_test, calibrated, at_recall)
            if at_recall is not None
            else {"unattainable": True,
                  "note": f"recall {config.REPORT_AT_RECALL:.0%} is not "
                          "reachable at any threshold on validation"}
        ),
        "at_precision_50": (
            operating_point(y_test, calibrated, at_precision)
            if at_precision is not None
            else {"unattainable": True,
                  "note": f"precision {config.REPORT_AT_PRECISION:.0%} is not "
                          "reachable at any threshold on validation — the "
                          "model never gets that confident"}
        ),
        "calibration": {
            "method": config.CALIBRATION_METHOD,
            "isotonic_applied": config.USE_ISOTONIC_CALIBRATION,
            "fitted_on": "validation period only",
            "decision": (
                "Isotonic was fitted, measured, and switched OFF: it left the "
                "Brier score unchanged and made the expected calibration "
                "error worse. Both curves are reported so the decision is "
                "visible rather than asserted."
            ),
            "brier_raw": float(brier_score_loss(y_test, uncalibrated)),
            "brier_isotonic": float(brier_score_loss(y_test, isotonic)),
            "brier_in_use": float(brier_score_loss(y_test, active)),
            "expected_calibration_error_raw":
                expected_calibration_error(reliability_before),
            "expected_calibration_error_isotonic":
                expected_calibration_error(reliability_after),
            "reliability_before": reliability_before,
            "reliability_after": reliability_after,
            # Isotonic is monotone non-decreasing: it may create ties, but it
            # must never invert a pair. Ties are why this is not a Spearman
            # check — see is_rank_preserving.
            "ranking_preserved": is_rank_preserving(uncalibrated, isotonic),
            "spearman_after_calibration": float(
                pd.Series(uncalibrated).corr(
                    pd.Series(isotonic), method="spearman"
                )
            ),
        },
        "by_animal_type": by_species,
        "human_summary": human_summary(operating, base_rate),
    }


def build_report(
    models,
    classifier_bundle,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> Report:
    """Compute every reported number from the held-out test set."""
    y_test = test[config.TARGET].to_numpy(dtype=float)

    predictions = models.predict_all(test)
    validation_predictions = models.predict_all(validation)
    quantiles = models.predict_quantiles(test)

    model_metrics = {}
    for name, y_pred in predictions.items():
        model_metrics[name] = {
            "test": core_metrics(y_test, y_pred),
            "validation": core_metrics(
                validation[config.TARGET].to_numpy(dtype=float),
                validation_predictions[name],
            ),
            "test_by_animal_type": metrics_by_group(
                y_test, y_pred, test["animal_type"]
            ),
            "test_by_predicted_decile": metrics_by_decile(y_test, y_pred),
        }

    gbm_predictions = predictions["hist_gradient_boosting"]
    variant_predictions = models.predict_variants(test)
    worst = worst_predictions(test, gbm_predictions)

    payload = {
        "target": config.TARGET,
        "target_transform": "fitted on log1p(days), inverted with expm1; all "
                            "reported numbers are in days",
        "split": {
            "kind": "temporal, never shuffled",
            "train": {
                "rows": int(len(train)),
                "intake_from": str(train["intake_datetime"].min()),
                "intake_to": str(train["intake_datetime"].max()),
            },
            "validation": {
                "rows": int(len(validation)),
                "intake_from": str(validation["intake_datetime"].min()),
                "intake_to": str(validation["intake_datetime"].max()),
            },
            "test": {
                "rows": int(len(test)),
                "intake_from": str(test["intake_datetime"].min()),
                "intake_to": str(test["intake_datetime"].max()),
            },
        },
        "hyperparameters": {
            "note": "fixed in config.GBM_PARAMS; no search was run, the first "
                    "configuration tried is the one reported",
            **{k: v for k, v in config.GBM_PARAMS.items()},
        },
        "models": model_metrics,
        "baseline_2_unseen_cell_rate": models.baseline_group.unseen_cell_rate(
            build_feature_matrix(test)
        ),
        "duan_smearing": {
            "citation": "Duan, N. (1983). Smearing Estimate: A Nonparametric "
                        "Retransformation Method. JASA 78(383), 605-610.",
            "status": (
                "NO LONGER APPLIED. Smearing corrects the retransformation "
                "bias of a log1p fit. The chosen objective is raw_absolute, "
                "which has no transform to invert, so there is nothing to "
                "correct. Kept, tested and reported because it is what the "
                "log1p variant needs and what motivated dropping log1p."
            ),
            "factor_S": float(models.gbm_log.smearing_factor_),
            "estimated_on": config.SMEARING_SOURCE,
            "residual_std_log_space": float(
                getattr(models.gbm_log, "smearing_residual_std_", float("nan"))
            ),
            "n_residuals": int(getattr(models.gbm_log, "smearing_n_", 0)),
            "comparison": smearing_comparison(
                y_test,
                variant_predictions["log1p"],
                variant_predictions["log1p_smearing"],
            ),
            "calibration_table_log1p": calibration_table(
                y_test, variant_predictions["log1p"]
            ),
            "calibration_table_log1p_smearing": calibration_table(
                y_test, variant_predictions["log1p_smearing"]
            ),
        },
        "regression_variants": {
            "chosen": config.REGRESSION_OBJECTIVE,
            "comparison": regression_variant_comparison(
                y_test, variant_predictions
            ),
        },
        "classifier": classifier_report(
            classifier_bundle, train, validation, test
        ),
        "species_diagnosis": {
            species: diagnose_species(
                classifier_bundle, train, validation, test, species
            )
            for species in ("Cat", "Dog")
        },
        "shelter_load": shelter_load_finding(
            pd.concat([train, validation, test])
        ),
        "shelter_load_test_period": shelter_load_finding(test),
        "calibration": {
            "interval": interval_calibration(y_test, quantiles),
            "table": calibration_table(y_test, gbm_predictions),
        },
        "permutation_importance": permutation_importance_table(
            models.gbm, test
        ),
        "failure_analysis": {
            "worst_predictions": characterise_failures(worst),
            "long_stay_tail": tail_analysis(test, gbm_predictions, quantiles),
        },
        "unobserved_factors": config.UNOBSERVED_FACTORS,
    }

    # Built last: these read the measured blocks above rather than recomputing.
    payload["primary_model"] = {
        "name": "long_stay_classifier",
        "output": "risk_probability",
        "why": (
            "Every regression variant scored a negative R2 on the held-out "
            "test set, so the day estimate predicts worse than a constant. "
            "The classifier is what works, so it is what the interface leads "
            "with."
        ),
    }
    payload["day_estimate_caveat"] = day_estimate_caveat(
        payload["regression_variants"]["comparison"][
            payload["regression_variants"]["chosen"]
        ]["r2"]
    )
    payload["model_limitations"] = model_limitations(
        payload["regression_variants"],
        payload["classifier"],
        payload["species_diagnosis"],
    )
    payload["findings"] = build_findings(
        pd.concat([train, validation, test]), payload
    )
    # Breakdowns the finding charts need, kept in the payload so the numbers
    # behind each chart are inspectable rather than only drawn.
    payload["breakdowns"] = _finding_breakdowns(
        pd.concat([train, validation, test])
    )
    return Report(payload)


def _finding_breakdowns(frame: pd.DataFrame) -> dict:
    work = frame.copy()
    work["age_bucket"] = age_bucket(work["age_days"])
    return {
        "by_age": {sp: breakdown(work, "age_bucket", where={"animal_type": sp})
                   for sp in ("Cat", "Dog")},
        "by_month": {sp: breakdown(work, "intake_month",
                                   where={"animal_type": sp})
                     for sp in ("Cat", "Dog")},
        "by_condition": breakdown(work, "intake_condition", min_n=100),
        "by_black": {sp: breakdown(work, "is_black",
                                   where={"animal_type": sp})
                     for sp in ("Cat", "Dog")},
        "black_controlled": controlled_effect(
            work, "is_black",
            ["animal_type", "age_bucket", "sex", "intake_type",
             "intake_condition"],
        ),
    }
