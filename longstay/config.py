"""Single source of truth for paths, constants, feature lists and split dates.

Nothing in this list should be hardcoded anywhere else in the project.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

INTAKES_CSV = RAW_DIR / "intakes.csv"
OUTCOMES_CSV = RAW_DIR / "outcomes.csv"

JOINED_PARQUET = PROCESSED_DIR / "joined.parquet"

EVALS_DIR = PROJECT_ROOT / "evals"
RESULTS_DIR = EVALS_DIR / "results"
# Charts live in their own directory so the deployment can serve exactly that
# folder as static assets, rather than exposing the whole results directory.
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_PATH = RESULTS_DIR / "model.joblib"
METRICS_PATH = RESULTS_DIR / "metrics.json"
FINDINGS_PATH = RESULTS_DIR / "findings.json"
REFERENCE_ROW_PATH = RESULTS_DIR / "reference_row.joblib"
CALIBRATION_PLOT_PATH = FIGURES_DIR / "calibration.png"
IMPORTANCE_PLOT_PATH = FIGURES_DIR / "permutation_importance.png"
WORST_PREDICTIONS_CSV = RESULTS_DIR / "worst_predictions.csv"


def ensure_dirs() -> None:
    """Create the data and results directories if they do not already exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Data source — Austin Animal Center, City of Austin open data (Socrata)
# --------------------------------------------------------------------------

INTAKES_URL = "https://data.austintexas.gov/resource/wter-evkm.json"
OUTCOMES_URL = "https://data.austintexas.gov/resource/9t4d-g238.json"

# Socrata caps $limit at 50,000 rows per request for anonymous access.
PAGE_SIZE = 50_000

# Anonymous access is rate limited. Retry with exponential backoff.
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 120

# Socrata pagination is only stable if the result set is explicitly ordered.
# `:id` is the internal row identifier and is guaranteed unique and stable.
SOCRATA_ORDER = ":id"


# --------------------------------------------------------------------------
# Cleaning thresholds
# --------------------------------------------------------------------------

# Length of stay outside [0, 365] days is dropped. Negative values mean the
# join or the source data is wrong; values beyond a year are real but rare
# enough (and operationally different enough) that we exclude them rather than
# let them dominate the error metrics.
MIN_LOS_DAYS = 0.0
MAX_LOS_DAYS = 365.0

# Age text arrives as "2 years", "3 months", "4 weeks", "1 day".
# Calendar-accurate conversions, not approximations.
DAYS_PER_UNIT = {
    "day": 1.0,
    "week": 7.0,
    "month": 365.25 / 12.0,
    "year": 365.25,
}

# Reported by `clean` and `stats`; also the cap used when one-hot encoding
# the high-cardinality breed column.
TOP_BREEDS_N = 30

SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


# --------------------------------------------------------------------------
# Temporal split — never random. See CLAUDE.md principle 2.
# --------------------------------------------------------------------------
#
# The split is on INTAKE date: the model is trained on animals admitted before
# the boundary and tested on animals admitted after it, which is exactly the
# decision it will face in production.
#
# Caveat we must keep saying out loud: because a row only exists once its
# outcome has been recorded, intakes near the end of the data are
# right-censored — long stays that have not resolved yet are simply absent.
# The most recent weeks are therefore biased toward short stays. `clean`
# reports the censoring horizon so this stays visible.

# Three-way split, on INTAKE date, in strict time order. The validation period
# sits between train and test so that nothing chosen on validation can see the
# test period.
#
# The Socrata feed currently ends 2025-05-04 (intakes) / 2025-05-05 (outcomes),
# not "the present" as the dataset description claims, so the test period runs
# 2024-01-01 to 2025-05-04. Re-check after any `fetch --force`: `stats` prints
# all three split sizes and a test asserts none of them is empty.

TRAIN_END_DATE = "2022-12-31"   # train: intakes on or before this date
VAL_START_DATE = "2023-01-01"
VAL_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"  # test: intakes on or after this date


# --------------------------------------------------------------------------
# Features — the leakage boundary. See CLAUDE.md principle 3.
# --------------------------------------------------------------------------
#
# ALLOWED_FEATURES is a whitelist. Only these columns may reach the model.
# Everything else in the processed frame is an identifier, a datetime, or the
# target, and is not an input.

NUMERIC_FEATURES = [
    "age_days",
]

CATEGORICAL_FEATURES = [
    "animal_type",
    "sex",                    # from sex_upon_intake
    "sterilization_status",   # from sex_upon_intake
    "intake_type",
    "intake_condition",
    "primary_breed",
    "primary_color",
    "intake_month",
    "intake_day_of_week",
    "intake_season",
]

BOOLEAN_FEATURES = [
    "is_mix",
    "is_black",
    # has_name was here and was REMOVED. See TEMPORAL_LEAKAGE_NOTES below and
    # tools/audit_timing.py. It is still engineered and still stored in the
    # processed frame, but as a diagnostic column, never as a model input.
]

ALLOWED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Engineered, kept in the parquet, and deliberately NOT modelled. Anything
# listed here failed the timing audit rather than the provenance check.
EXCLUDED_FOR_TIMING = [
    "has_name",
]

TARGET = "length_of_stay_days"

# Columns that are legitimately in the processed frame but are NOT features:
# identifiers, the target, and the two datetimes used to compute it.
NON_FEATURE_COLUMNS = [
    "animal_id",
    "intake_datetime",
    "outcome_datetime",
    TARGET,
] + EXCLUDED_FOR_TIMING

# Recorded at or after the outcome. If any of these reaches the model the
# model is worthless, so they are dropped in clean.py and asserted against in
# features.py. `outcome_datetime` is deliberately absent from this list: it is
# used ONLY to compute the target and is never an input.
FORBIDDEN_COLUMNS = [
    "outcome_type",
    "outcome_subtype",
    "sex_upon_outcome",
    "age_upon_outcome",
    "outcome_monthyear",
    "outcome_name",
    "outcome_breed",
    "outcome_color",
    "outcome_animal_type",
    "date_of_birth",
]

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------

RANDOM_STATE = 20260816

# The target is heavily right-skewed: most animals leave within a week, a long
# tail waits months. We fit on log1p(days) and invert with expm1 for reporting,
# because "0.4 log days" means nothing to a shelter worker.
#
# Note the median is invariant under a monotonic transform, so the median
# baselines are identical whether computed in days or in log space. Quantiles
# are invariant for the same reason, which is why the quantile models can be
# fitted in log space and inverted without distorting the interval.

# Age buckets for baseline 2 (animal_type x age_bucket lookup table).
AGE_BUCKET_EDGES = [0, 60, 180, 365, 1095, 2555]
AGE_BUCKET_LABELS = [
    "under 2mo",
    "2-6mo",
    "6-12mo",
    "1-3y",
    "3-7y",
    "7y+",
]
AGE_BUCKET_UNKNOWN = "unknown"

# HistGradientBoostingRegressor. Native categorical support avoids one-hot
# exploding the breed column. max_bins is 255, and a native categorical feature
# may not exceed that many levels, so high-cardinality columns are capped and
# the tail folded into "Other" — using TRAINING frequencies only.
MAX_MODEL_CATEGORIES = 254

# Fixed, disclosed hyperparameters. No search was run: the first configuration
# tried is the one reported. If it loses to a baseline, that is the result.
GBM_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    # Early stopping would carve out its own validation slice by SHUFFLING,
    # which leaks future rows into the stopping decision. We have a real
    # temporal validation split for that; this stays off.
    "early_stopping": False,
}

# 10th / 50th / 90th percentile models for the calibration interval.
QUANTILES = (0.1, 0.5, 0.9)
INTERVAL_NOMINAL_COVERAGE = 0.80  # what a 10-90 band should contain

# --------------------------------------------------------------------------
# Duan smearing — retransformation bias correction
# --------------------------------------------------------------------------
#
# Fitting squared error on log1p(y) estimates E[log1p(Y)|X]. Inverting that
# with expm1 does NOT give E[Y|X]: by Jensen's inequality it gives something
# closer to a conditional geometric mean, which for a right-skewed conditional
# distribution sits systematically BELOW the arithmetic mean. Our first
# evaluation measured exactly that — every calibration decile above the
# diagonal, mean predicted roughly half of mean actual.
#
# This is a known defect of naive retransformation, and it has a standard
# nonparametric fix:
#
#   Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation
#   Method." Journal of the American Statistical Association, 78(383), 605-610.
#
# Duan's estimator averages the back-transform over the empirical residual
# distribution instead of back-transforming a single point:
#
#   E[Y|X] ~= (1/n) * sum_i g(f(X) + e_i)
#
# where g is the inverse transform and e_i are the fitted residuals in
# transformed space. For g = expm1 this collapses to a closed form, because
# expm1(a + e) = exp(a) * exp(e) - 1:
#
#   E[Y|X] ~= exp(f(X)) * S - 1,   S = mean_i( exp(e_i) )
#
# So the whole correction is one scalar S, applied multiplicatively in exp
# space. S is strictly positive, so the correction is strictly monotone in
# f(X) and CANNOT change the ranking — only the level. A test asserts this.
#
# This is a correction of a known transformation error, not hyperparameter
# tuning: nothing about the fitted model changes, and S is not chosen to
# optimise any test metric.
#
# S is estimated on the VALIDATION period, not the training period. A boosted
# model's in-sample residuals are shrunk by its own fit, which would bias S
# downward and under-correct. The validation split exists for exactly this
# kind of out-of-sample estimate, and it is still strictly earlier than test.
SMEARING_SOURCE = "validation"

# Evaluation constants.
TOLERANCE_DAYS = (7, 14)     # "within tolerance" thresholds
N_DECILES = 10
LONG_STAY_DAYS = 90          # the population the triage tool exists to catch
N_WORST_PREDICTIONS = 50
PERMUTATION_REPEATS = 10
N_TOP_IMPORTANCES = 15
QUICK_EXIT_DAYS = 7          # "left within a week", for the headline finding


# --------------------------------------------------------------------------
# Long-stay classifier — THE PRIMARY MODEL
# --------------------------------------------------------------------------
#
# The regression answers "how many days?". The operational question is "will
# this animal still be here in three months?", and the first evaluation showed
# the regressor answering it badly: of 713 test animals who stayed 90+ days it
# predicted 90+ for exactly none. So the binary target is now the main model
# and the day estimate is secondary context.

CLASSIFIER_TARGET = "stay_90_plus"  # length_of_stay_days >= LONG_STAY_DAYS

CLASSIFIER_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,  # would shuffle; see GBM_PARAMS
}

# --- Operating threshold ---------------------------------------------------
#
# The two errors are NOT symmetric, and the threshold has to say so.
#
#   A MISS  (animal will wait 90+ days, we do not flag them) costs that animal
#           months of additional kennel life. They get no foster placement, no
#           featured listing, no waived fee, and nobody looks at them again
#           until they have already been waiting a season. That is the exact
#           failure this project exists to prevent.
#
#   A FALSE ALARM (animal leaves quickly, we flagged them anyway) costs one
#           foster placement or one marketing slot used on an animal who did
#           not need it. Real, but recoverable within days, and the animal is
#           not harmed by the extra attention.
#
# We judge a miss roughly NINE times as costly as a false alarm. That is a
# value judgement, not a measurement, which is why it is written here in the
# open where it can be argued with rather than buried in a notebook.
#
# Under expected cost, the optimal threshold for a cost ratio r = C_miss/C_fa
# is p* = 1 / (1 + r). With r = 9 that is 0.100. The threshold is therefore
# DERIVED from the stated cost ratio, not tuned until the metrics looked good.
#
# History, because the reasoning matters more than the number: this started at
# 5.0, which gives p* = 0.167. That sounded lenient and was not. This model
# almost never emits a probability above 0.2, so 0.167 sat near the top of its
# range and flagged only 6.1% of intakes, catching 21% of long stays. A policy
# that says "favour recall" and then misses four out of five is not that
# policy. Raising the ratio to 9 is not tuning against a metric: it is the
# stated preference finally being expressed at a threshold the model's score
# distribution can act on.
COST_RATIO_MISS_TO_FALSE_ALARM = 9.0
OPERATING_THRESHOLD = 1.0 / (1.0 + COST_RATIO_MISS_TO_FALSE_ALARM)

# Reference operating points reported alongside, so the trade-off curve is
# visible rather than implied by a single chosen point.
REPORT_AT_RECALL = 0.50
REPORT_AT_PRECISION = 0.50

# Probability calibration.
#
# Boosted-tree scores are often not probabilities, so isotonic regression on
# the validation period is the usual fix. Here it was measured and it did not
# pay: Brier score unchanged at 0.0448, and the expected calibration error got
# WORSE, 0.0104 -> 0.0123. HistGradientBoostingClassifier's raw scores were
# already calibrated to about one percentage point, and refitting on 11,138
# validation rows added more noise than it removed.
#
# So it is off. The code stays, and `evaluate` still reports both curves, so
# the decision is visible as numbers rather than asserted here.
USE_ISOTONIC_CALIBRATION = False
CALIBRATION_METHOD = "isotonic" if USE_ISOTONIC_CALIBRATION else "none (raw scores)"
RELIABILITY_BINS = 10

# Risk bands are computed from the PROBABILITY, not from predicted days.
# "Elevated" starts at the operating threshold by construction, so the band a
# staff member sees and the threshold the model is tuned to are the same
# number rather than two numbers that drift apart.
RISK_BAND_HIGH = 0.40
RISK_BAND_ELEVATED = OPERATING_THRESHOLD

# A fourth band below the elevated one. The test-set base rate is about 4.9%,
# so an animal scoring under 2% is well clear of typical and can be left to
# the normal queue with confidence. Splitting "not flagged" into TYPICAL and
# FAST is a display distinction only — it does not affect the operating
# threshold, which remains RISK_BAND_ELEVATED.
RISK_BAND_FAST = 0.02

# --------------------------------------------------------------------------
# Regression objective
# --------------------------------------------------------------------------
#
# Three variants were measured head to head (see `evaluate`):
#
#   log1p            squared error on log1p(days), naive expm1 back-transform.
#                    Good MAE, but biased low by construction — Jensen.
#   log1p_smearing   the same fit with Duan's correction. Fixes the level,
#                    wrecks MAE and median error, because smearing targets the
#                    conditional MEAN while MAE is minimised by the MEDIAN.
#   raw_absolute     absolute_error loss on raw days, no transform at all.
#                    No retransformation step, so no retransformation bias,
#                    and the loss being optimised is the loss being reported.
#
# REGRESSION_OBJECTIVE selects which one the service and the headline table
# use. The other two are still fitted and reported so the choice is visible.
REGRESSION_OBJECTIVE = "raw_absolute"

REGRESSION_VARIANTS = ("log1p", "log1p_smearing", "raw_absolute")

# absolute_error optimises the conditional median directly. Same tree
# hyperparameters as GBM_PARAMS so the comparison is about the objective and
# not about capacity.
GBM_ABSOLUTE_PARAMS = {**GBM_PARAMS, "loss": "absolute_error"}


# --------------------------------------------------------------------------
# Per-species reliability disclosure
# --------------------------------------------------------------------------
#
# A model that silently emits a useless number is worse than one that says
# "I do not know here". Any species whose PR-AUC lift over its OWN base rate
# falls below this gets an explicit, prominent low-confidence notice in the
# API response and in the result card — not a footnote.
#
# Lift, not raw PR-AUC: a rare positive class drags PR-AUC down on its own,
# and the honest question is whether the model beats that species' own base
# rate, not whether it matches another species' number.
RELIABILITY_MIN_LIFT = 2.0
SPECIES_RELIABILITY_PATH = RESULTS_DIR / "species_reliability.json"

CLASSIFIER_MODEL_PATH = RESULTS_DIR / "classifier.joblib"
PR_CURVE_PLOT_PATH = FIGURES_DIR / "precision_recall.png"
SHELTER_LOAD_PLOT_PATH = FIGURES_DIR / "shelter_load.png"
AGE_BUCKET_PLOT_PATH = FIGURES_DIR / "stay_by_age.png"
MONTH_PLOT_PATH = FIGURES_DIR / "stay_by_month.png"
CONDITION_PLOT_PATH = FIGURES_DIR / "stay_by_condition.png"
BLACK_EFFECT_PLOT_PATH = FIGURES_DIR / "black_effect.png"
RELIABILITY_PLOT_PATH = FIGURES_DIR / "reliability.png"
CALIBRATION_BEFORE_AFTER_PATH = FIGURES_DIR / "calibration_smearing.png"

# --------------------------------------------------------------------------
# TEMPORAL LEAKAGE NOTES
# --------------------------------------------------------------------------
#
# The whitelist above answers "which TABLE did this column come from". That is
# provenance, and it is not the whole question. The question that actually
# matters is "WHEN was this value written" — a field can sit in the intakes
# table and still be edited months into the stay. Provenance is mechanically
# checkable; timing is not, because this feed records no per-field timestamps.
#
# `has_name` passed the provenance check and leaked anyway. These notes are the
# standing record so the next person does not have to rediscover it.
#
# Re-run `python tools/audit_timing.py` after any `fetch --force`.
#
# The diagnostic: compare the intake-side and outcome-side copy of a field for
# the same stay. If they NEVER differ across ~172k stays, the feed stores one
# animal-level value and writes it into both rows — so a later edit silently
# rewrites the intake record. If they do differ, the field is stored per event
# and the intake copy is trustworthy.

TEMPORAL_LEAKAGE_NOTES = {
    "has_name": {
        "status": "LEAKING — removed from the model",
        "evidence": (
            "67.6% of STRAY intakes carry a name, which cannot be true at "
            "intake for animals picked up off the street; Wildlife intakes, "
            "which nobody names, sit at 3.0% and act as the control. The "
            "intake and outcome copies are identical in 122,908 of 122,908 "
            "stays, and has_name never varies across an animal's repeat "
            "visits. Within strays alone the 90+ day rate is 5.38% for named "
            "animals against 0.89% for unnamed — a 6x gap with intake "
            "circumstance held constant."
        ),
        "interpretation": (
            "The name is one animal-level attribute back-filled into every "
            "row. Shelters name animals that stick around, so the field "
            "partly encodes staff attention DURING the stay."
        ),
        "cost_of_removal": (
            "PR-AUC 0.1268 -> 0.1174 (-7.4%), recall at the working threshold "
            "35.3% -> 29.2%, 252 -> 208 of 713 long stays found. Removed "
            "anyway: the premise is worth more than the points."
        ),
    },
    "sex / sterilization_status": {
        "status": "CLEAN — verified per-event",
        "evidence": (
            "70,985 stays show Intact at intake and Fixed at outcome. The "
            "shelter neuters animals during their stay and the feed records "
            "both states separately, so the intake copy is genuinely "
            "intake-time. This is the field we most expected to leak, and it "
            "does not."
        ),
    },
    "age_days": {
        "status": "CLEAN — verified per-event",
        "evidence": (
            "Intake and outcome ages differ for 42,768 stays (24.9%), and age "
            "varies across repeat visits for 56.1% of returning animals, both "
            "consistent with age advancing in real time."
        ),
        "residual_risk": (
            "Age is a coarse text estimate ('2 years'). If it is derived from "
            "a date_of_birth that a vet later revises, the intake age would "
            "move with it. The feed carries no edit history, so this CANNOT "
            "be ruled out from the data."
        ),
    },
    "animal_type": {
        "status": "CLEAN — immutable in practice",
        "evidence": (
            "Present in both tables and identical in every stay, so it shares "
            "the animal-level storage shape. Unlike a name or a breed label, "
            "though, species is not a judgement that gets revised: a dog "
            "recorded at intake is still a dog at outcome, and no plausible "
            "post-intake edit correlates with length of stay. The storage "
            "signature is the same; the risk is not."
        ),
    },
    "primary_breed / primary_color / is_mix / is_black": {
        "status": "UNVERIFIABLE — same storage shape as has_name, no evidence "
                  "of harm",
        "evidence": (
            "Intake and outcome copies are identical in 172,138 of 172,138 "
            "stays and never vary across an animal's repeat visits, which is "
            "the same animal-level storage signature has_name showed. Unlike "
            "has_name there is no positive evidence of post-intake editing: "
            "breed and colour genuinely do not change, so identical copies "
            "are also exactly what a clean per-event field would produce. The "
            "test cannot separate the two cases here."
        ),
        "residual_risk": (
            "If staff refine a breed label after spending time with an animal "
            "— plausible, since long-stay animals get more attention — that "
            "correction would rewrite the intake record. Unquantifiable from "
            "this feed. Kept, and flagged."
        ),
    },
    "intake_type / intake_condition": {
        "status": "UNTESTABLE — no outcome-side copy exists",
        "evidence": (
            "Neither field appears in the outcomes table, so the "
            "intake-versus-outcome comparison cannot be run at all. "
            "intake_condition varies across an animal's repeat visits, which "
            "at least shows it is stored per visit rather than per animal."
        ),
        "residual_risk": (
            "intake_condition is plausibly updated after a veterinary "
            "examination hours or days after arrival. 'Normal' on the form "
            "may mean 'nothing obvious at the door' or 'nothing found on "
            "examination', and the data cannot tell us which. Kept, and "
            "flagged."
        ),
    },
    "intake_month / intake_day_of_week / intake_season": {
        "status": "CLEAN — derived from the intake timestamp itself",
        "evidence": (
            "Computed in clean.py from intake_datetime, which is the moment "
            "the event happened rather than a field describing it. There is "
            "no later value to overwrite it with: an animal admitted in "
            "August was admitted in August whatever anyone edits afterwards. "
            "The only way these could mislead is if intake_datetime itself "
            "were wrong, which would corrupt the target as well and show up "
            "as negative stays — of which the cleaning step found none."
        ),
    },
}


# What this data cannot see, no matter how good the model gets.
# See CLAUDE.md principle 5. Surfaced in the CLI and in the UI.
UNOBSERVED_FACTORS = [
    "photographs and how the animal presents in them",
    "temperament and behaviour notes",
    "medical complexity beyond the coarse intake_condition label",
    "staff effort and volunteer advocacy",
    "local adoption events, promotions and seasonal campaigns",
    "kennel location and visibility to visitors",
]
