# CLAUDE.md — Longstay

Guidance for Claude Code when working in this repository.

---

## What this project is

**Longstay** predicts how long an animal will stay in a shelter, using only
information available on intake day, and ranks today's intakes so staff know
where to spend a limited supply of foster homes, marketing budget, and attention.

**The problem.** A shelter has a fixed number of foster homes, a limited
marketing budget, and finite staff attention. On intake day nobody knows which
animals will be adopted within a week and which will sit for four months. So
attention gets distributed evenly, or by whoever is most visible, rather than by
who needs it. The animals who most need intervention — a foster placement, a
featured listing, a waived fee — are frequently the ones nobody flags until
they have already been waiting ninety days.

**The solution.** A model over intake-day features that estimates the
probability an animal will still be here in three months, plus a triage
interface that ranks today's intakes by that probability and explains each
prediction. **A decision tool, not a dashboard.**

## The framing constraint — read this before proposing anything

This dataset (Austin Animal Center) has been analyzed many times, and almost
always the same way: as a classification problem predicting `outcome_type`
(adopted / transferred / returned-to-owner). We are deliberately not doing that.

| The usual project | Longstay |
|---|---|
| Predict `outcome_type` — *what* happens | **Predict duration — *how long* it takes** |
| Report accuracy / F1 | **Report calibration** — does a stated 20% risk happen 20% of the time? |
| Build a dashboard | **Build a triage tool** — a ranked worklist for today |
| Random train/test split | **Temporal split** — past predicts future |

The axis that matters is **duration, not outcome category**. Whether that
duration is expressed as a regression on days or as a probability of crossing
a 90-day threshold is an engineering choice, and it has already been made
once on evidence — see below.

If a change would slide the project back toward predicting `outcome_type`,
toward accuracy-only reporting, or toward a general-purpose dashboard, do not
make it — say so instead.

## Which model is primary — settled, with evidence

**The long-stay classifier is the primary model.** `stay_90_plus` =
`length_of_stay_days >= 90`. Its calibrated probability is the number the UI
leads with and the number triage is ordered by.

**The day estimate is secondary and must be labelled as such.** Every
regression variant tried scored a NEGATIVE R² on the held-out test set — it
predicts worse than a constant. That is not a bug to tune away: it is the
finding. How long an animal waits is largely not explained by what is
knowable at intake. Do not attempt to fix it with hyperparameter search.

Consequences that are binding:

- `risk_probability` gets the large type and drives `risk_band`.
- `predicted_days` is displayed smaller, with its 10-90 interval, and always
  next to a caveat naming the actual R². The caveat text and the number both
  come from `metrics.json` — never hardcode either.
- Feature drivers are computed from the CLASSIFIER, in percentage points of
  probability.

## Engineering principles

These are not style preferences. Violating any of them makes the project wrong,
not merely untidy.

**1. Every number a user sees comes from a computation we can point to.**
No estimates, no rounded-off guesses, no illustrative placeholder figures. If a
number appears in the UI, in the CLI, or in a report, there is a line of code
that produced it from the data. If we cannot compute it, we do not display it.

**2. The train/test split is temporal, never random.**
We predict the future from the past, so the test set must cover a strictly later
time period than the training set. A random split leaks future information and
inflates the score. The split date lives in `config.py`. Any sklearn helper that
shuffles by default (`train_test_split`, plain `KFold`, `cross_val_score`) is
banned unless shuffling is explicitly disabled and the ordering is temporal.

**3. Only features knowable AT INTAKE may reach the model.**
Anything recorded at or after the outcome is leakage and must be explicitly
excluded. This is the single easiest way to accidentally build a model that
looks brilliant and is worthless. `outcome_type`, `outcome_subtype`,
`sex_upon_outcome`, `age_upon_outcome`, and `outcome_datetime` are all leakage.
`outcome_datetime` is used *only* to compute the target and never as an input.
Enforcement is mechanical, not a matter of care: `config.ALLOWED_FEATURES` is
the whitelist, code asserts against it, and a test fails if a forbidden column
reaches the model.

**4. Report calibration, not just error.**
A stated risk of 20% is only useful if roughly 20% of those animals really do
stay. Report the reliability diagram and Brier score for the classifier, and
interval coverage for the day estimate. Error metrics alone are insufficient.
Where a calibration step was tried and did not pay — isotonic regression was,
and did not — report both curves rather than asserting the decision.

**4a. Disclose per-species reliability, prominently.**
The model works materially better for dogs than for cats, and diagnosis showed
that is the ceiling of the available features rather than a fixable modelling
error. **Do not quote the figures here** — they live in
`evals/results/species_reliability.json` and they move whenever the feature set
or the data does. Any species below `RELIABILITY_MIN_LIFT` gets an
explicit low-confidence block in the UI, at full size, above the number. A
model that silently emits a useless number is worse than one that says "I do
not know here". Never shrink, collapse, or footnote that block.

**5. Be honest about what the model cannot see.**
Photographs, temperament, behavioral notes, medical complexity, staff effort,
volunteer advocacy, and local adoption events are all absent from this data and
all matter enormously to how long an animal waits. State this in the README, in
the UI, and anywhere predictions are presented. A confident-looking number from
a model blind to temperament is a liability if we let it look authoritative.

## Tech stack

- Python 3.11+ — pandas, numpy, scikit-learn, matplotlib
- FastAPI serving one static HTML page
- pytest
- **No Streamlit.**
- **No LLM, no API key, no paid service anywhere in this project.** If a task
  seems to want one, solve it with deterministic code or leave it undone and
  explain why.

## Data

Austin Animal Center, City of Austin open data portal (Socrata):

| Dataset | Endpoint |
|---|---|
| Intakes | `https://data.austintexas.gov/resource/wter-evkm.json` |
| Outcomes | `https://data.austintexas.gov/resource/9t4d-g238.json` |

Both accept SODA query parameters (`$limit`, `$offset`, `$where`, `$select`).
No API key required for anonymous access, but it is rate-limited. Roughly
174,000 outcome rows covering October 2013 to the present.

Join on `animal_id`. `length_of_stay_days = outcome_datetime - intake_datetime`.

**`animal_id` is not unique** — animals are admitted more than once. Both frames
must be sorted by datetime and each intake paired with the *next* outcome for
that animal chronologically. Getting this wrong silently corrupts every
downstream number, so it is covered by a test.

**Be polite to the API.** Raw pulls are cached under `data/raw/` and are not
re-downloaded unless `--force` is passed. Rate limiting is handled with retry
and backoff.

## Layout

```
longstay/
  __init__.py
  config.py      # paths, constants, feature lists, split dates, thresholds
  fetch.py       # download from Socrata, cached, paged, backoff
  clean.py       # join, clean, engineer features
  features.py    # feature whitelist, encoders, temporal split
  model.py       # baselines, regression variants, long-stay classifier
  evaluate.py    # metrics, calibration, diagnosis, plots
  service.py     # scoring for the API — both numbers, drivers, reliability
  api.py         # FastAPI: /api/predict, /api/model-card, static page
data/
  raw/           # intakes.csv, outcomes.csv — cached API pulls
  processed/     # joined.parquet
evals/results/   # model artifacts, metrics.json, plots
static/          # index.html — the single page, inline style and script
tests/
main.py          # CLI: fetch, clean, stats, train, evaluate
```

`config.py` is the single source of truth for paths, split dates, cleaning
thresholds, and feature lists. Do not hardcode any of those elsewhere.

Generated data files are artifacts, not source. Never hand-edit anything under
`data/`.

## Working conventions

- **Report drops, always.** Every cleaning step that removes rows reports how
  many and why. Silent row loss is a bug even when the filter is correct.
- **Assert the leakage boundary in code**, not just in tests. The training frame
  is checked against `ALLOWED_FEATURES` at the moment it is built.
- **Write the test for the join.** The intake→next-outcome pairing and the
  leakage whitelist are the two places where a silent error is most expensive.
- Prefer plain pandas and explicit code over clever abstraction. This is a
  teaching project; readability beats concision.
- Keep the CLI honest: `fetch`, `clean`, `stats`, `train`, `evaluate` do what
  they say and print real computed numbers.

## The interface

One static HTML file, `static/index.html`, with inline `<style>` and
`<script>`. No framework, no build step. The only permitted external request
is Google Fonts.

Three tabs: **Predict** (the triage tool), **Findings** (the findings),
**Model** (the model card). The tool is the default; the other two exist so
the claims can be checked, and their charts are the PNGs in `evals/results/`.

All user-facing copy is in English, sentence case in prose, uppercase only in
the small tracked labels. No emoji.

The charts belong on the Findings and Model tabs. Keep them off the
prediction tab — that one is a worklist, not a report.

Every figure the page shows is fetched from `metrics.json` or the prediction
endpoint. No number is written into the HTML by hand. This is principle 1
applied to the UI, and it is the reason the caveat text lives in the metrics
file rather than in a template string.

## Things that are wrong here even though they are normal elsewhere

- `train_test_split(...)` with default shuffling.
- Adding `outcome_type` "just as a sanity check" on the training frame.
- Filling missing `age_upon_intake` with the mean and not mentioning it.
- Reporting R² or MAE and stopping there.
- Rounding a metric for presentation until it no longer matches the computation.
- Turning the PREDICTION tab into a set of summary charts.
- Leading with `predicted_days`, or showing it without its caveat.
- Hardcoding a metric into the HTML instead of reading it from `metrics.json`.
- Quoting a measured figure in prose — including in this file. Every metric
  moves when the feature set or the data moves, and a number written into
  documentation goes stale silently. Name the artifact that holds it instead.
- Tuning hyperparameters to make R² positive.
