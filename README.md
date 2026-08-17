# Longstay

Predicts which animals will still be in the shelter in three months, using
only what is knowable on intake day, and ranks the day's intakes so staff know
where to spend a limited supply of foster homes, marketing budget and
attention.

A decision tool, not a dashboard.

> **4% of animals stay longer than three months, and those animals take up
> 33% of all the time the shelter has to give.**

That sentence is the argument for the whole project, and it is the one
measured figure quoted anywhere in this file — every other number lives in
`evals/results/metrics.json` and is asserted by a test rather than restated in
prose. Numbers written into documentation go stale silently; this one is
pinned by `test_callout_percentages_match_the_computed_shelter_load`.

Data: [Austin Animal Center](https://data.austintexas.gov), via the City of
Austin's Socrata open-data portal. Roughly 172,000 completed stays from
October 2013 onward.

---

## What it actually does

Two models, and the distinction between them matters more than either:

**The long-stay classifier is primary.** It estimates the probability an
animal stays 90+ days. This is the number the interface leads with and the
number triage is ordered by.

**The day estimate is secondary and labelled as such.** Every regression
variant tried scored a *negative* R² on the held-out test set — it predicts
worse than a constant. That is not a bug awaiting a fix; it is the finding.
How long an animal waits is largely not explained by what is knowable at
intake. The day figure is shown small, with its 10–90 interval, and always
beside a caveat naming the actual R², which the page reads from
`metrics.json` rather than hardcoding.

Three tabs: **Predict** (the triage tool), **Findings** (seven findings, each
with its numbers, sample size and caveat), **Model** (the model card,
calibration curves, feature importance and limitations).

---

## Setup from cold

Requires Python 3.11+ and about 50 MB of disk for the raw pull.

```bash
git clone <your-fork-url>
cd animal-shelter-analyzes

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Serve immediately — the fitted models are committed, so this needs
# no download and no training:
uvicorn longstay.api:app --port 8000
# open http://127.0.0.1:8000
```

To rebuild everything from source data instead:

```bash
python main.py fetch      # downloads from Socrata, cached; --force to refresh
python main.py clean      # join, target, features -> data/processed/
python main.py stats      # describe the processed frame
python main.py train      # fits baselines, regressors and the classifier
python main.py evaluate   # metrics, calibration, figures -> evals/results/
pytest                    # 301 tests
```

Two read-only analysis tools:

```bash
python tools/verify_join.py    # inspect the intake -> outcome pairing
python tools/audit_timing.py   # re-run the timing-leakage audit
```

---

## Architecture

```
longstay/
  config.py     single source of truth: paths, split dates, feature lists,
                thresholds, and TEMPORAL_LEAKAGE_NOTES
  fetch.py      Socrata paging, caching, backoff
  clean.py      intake -> next-outcome pairing, target, feature engineering
  features.py   the whitelist gate and the temporal split
  model.py      baselines, three regression variants, the classifier
  evaluate.py   metrics, calibration, diagnosis, findings, figures
  service.py    scoring for the API — both numbers, drivers, reliability
  api.py        FastAPI: /api/predict, /api/predict/batch, /api/model-card,
                /api/health, and the static page
static/
  index.html    the whole interface: one file, inline style and script
evals/results/  committed artifacts the deployment boots from
```

Three rules the code enforces mechanically rather than by care:

**The split is temporal, never random.** Train ends before validation, which
ends before test. Any sklearn helper that shuffles by default is banned. A
test fails if the periods overlap or if any period is empty.

**Only intake-day features reach the model.** `config.ALLOWED_FEATURES` is a
whitelist; `build_feature_matrix` asserts against it; a test fails if a
forbidden column gets through. `outcome_datetime` is used only to compute the
target and never as an input.

**Every number the interface shows comes from `metrics.json`.** A test scans
the page for hardcoded metric literals, and the renderer validates every key
it reads — a stale key produces a visible error card, not a silently blank
panel.

---

## Limitations

### Temporal leakage — the audit

This is the most important section in this file.

The leakage check was mechanical and it passed, and a feature leaked anyway.
The check verified *provenance*: which table did this column come from. The
question that actually matters is *timing*: when was this value written. A
field can sit in the intakes table and still be edited months into the stay,
and this feed carries no per-field timestamps to tell you.

`has_name` — whether an animal arrived with a name — was the second most
important feature in the model. It came from the intakes table, so it passed
every check. It was also not knowable at intake:

- Two thirds of **stray** intakes carry a name. An animal picked up off the
  street cannot already have one.
- **Wildlife** intakes, which nobody names, sit near zero — the control that
  proves the method rather than the measurement.
- The intake-side and outcome-side copies of the name are identical in every
  one of ~123,000 stays. Shelters name animals that stick around, so a
  genuine intake-time field would show thousands of transitions. Zero means
  one animal-level value written into both rows.
- Among strays alone, holding intake circumstance constant, named animals are
  six times likelier to still be there after three months.

The field was partly recording *staff attention during the stay*. It was
removed. Removal cost real performance — PR-AUC, recall and 44 of the long
stays the model previously caught, all quantified in `metrics.json` — and it
was removed anyway, because a model that predicts the future using the future
is not a model.

The same audit was then run over every other feature:

| Feature | Verdict |
|---|---|
| `sex` / `sterilization_status` | **Clean, verified.** Tens of thousands of Intact→Fixed transitions prove the field is recorded per event. This is the one we most expected to leak. |
| `age_days` | **Clean, verified.** Ages differ between intake and outcome, and vary across repeat visits, consistent with age advancing in real time. |
| `primary_breed`, `primary_color`, `is_mix`, `is_black` | **Unverifiable.** Same animal-level storage shape as `has_name`, but no positive evidence of harm — breed and colour genuinely don't change, so identical copies are also what a clean field would produce. The test cannot separate the two cases. Kept and flagged. |
| `intake_type`, `intake_condition` | **Untestable.** Neither has an outcome-side copy, so the comparison cannot be run. "Normal" may mean "nothing obvious at the door" or "nothing found on examination hours later", and the data cannot say which. Kept and flagged. |
| `animal_type` | Clean. Species is not a judgement that gets revised. |
| `intake_month`, `intake_day_of_week`, `intake_season` | Clean. Derived from the intake timestamp itself. |

The standing record, with the reasoning for each verdict, is
`config.TEMPORAL_LEAKAGE_NOTES`. Tests require every modelled feature to
carry a verdict and every exclusion to carry a documented reason. Re-run
`python tools/audit_timing.py` after any `fetch --force`.

### The day estimate predicts worse than a constant

Negative R² across all three regression variants. Reported on the Model tab,
not hidden. It is also systematically low by roughly two weeks; the Duan
smearing correction removes that bias but costs accuracy, so accuracy was
chosen and the trade-off is documented.

### 50% precision is unreachable at any threshold

Class imbalance sets the ceiling — long stays are a small minority, so most
flagged animals will still leave quickly. The tool sorts a queue; it does not
deliver a verdict about an individual animal.

### The model is barely better than chance for cats

Diagnosis showed this is the ceiling of the available features, not a fixable
modelling error: a cat-only model improves on it by almost nothing. Cats have
no equivalent of dog breed, which is the strongest signal the model has. The
interface shows an explicit, full-size low-confidence block for any species
below the reliability threshold — it is never shrunk or footnoted.

### What the model cannot see, and never will

Photographs and how an animal presents in them. Temperament. Behavioural
notes. Medical complexity beyond one coarse condition label. Staff effort and
volunteer advocacy. Local adoption events. Whether the kennel sits on the
main walkway or in a far corner. All of these matter enormously to how long an
animal waits, and none is in this data.

### One shelter, one city

Everything here describes Austin Animal Center's recording practices and
adopter population. None of it transfers without re-checking.

---

## Deployment

Configured for Render's free tier via `render.yaml`. The service does no work
at boot beyond loading committed artifacts — it never contacts Socrata and
never retrains. `GET /api/health` reports each artifact's presence and size,
whether the models actually loaded, and when the metrics were generated, so a
failed cold start is diagnosable without shell access.

Free-tier instances spin down when idle; expect a 30–60 second first request.

`data/raw/` is gitignored — the source CSVs are ~48 MB and re-downloadable
with `python main.py fetch`.

## Licence and attribution

Data from the City of Austin open data portal under its published terms.
This is coursework, not a production system, and no part of it should be used
to make decisions about an animal without a person in the loop.
