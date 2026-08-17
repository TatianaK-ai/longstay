# Longstay — Project Documentation

Predicting which shelter animals will still be waiting in three months, from
what is knowable on intake day.

| | |
|---|---|
| **Live app** | https://longstay.onrender.com |
| **Source** | https://github.com/TatianaK-ai/longstay |
| **Dataset** | Austin Animal Center, City of Austin open data |
| **Scope** | 171,561 completed stays, October 2013 – May 2025 |

Every figure in this document is produced by `python main.py evaluate` and
stored in `evals/results/metrics.json`. None of it is typed in by hand.

---

## 1. The problem

Animals arrive at a shelter every day — strays found on the street, pets
surrendered by owners who can no longer keep them, animals brought in by the
public. The shelter takes them all and holds each one until it is adopted,
returned to its owner, or transferred to a rescue partner.

A shelter has a fixed number of kennels, a fixed number of volunteer foster
homes, limited staff hours and a limited marketing budget. **Every day an animal
stays is a day that kennel is not available for the next animal.**

More than half of all animals find a home within a week. But four percent stay
longer than three months, and those four percent consume a third of the
shelter's entire capacity.

The difficulty is that on intake day nobody knows which group an animal belongs
to. Extra help — a foster placement, a featured listing, a waived fee — is
limited, so it goes to whoever is most visible. The animals who need it most are
usually the ones nobody flags, until they have already waited ninety days.

---

## 2. What the tool does

A shelter worker enters what is knowable on intake day: species, age, sex,
spay/neuter status, breed, primary colour, intake type, intake condition, and
the month.

It returns:

| Output | What it is |
|---|---|
| **Risk probability** | Chance this animal stays 90+ days. The primary output, and what triage is ordered by. |
| **Risk band** | AT RISK / SLOW / TYPICAL / FAST |
| **Estimated stay** | Median days with a 10–90 range. Secondary, shown small and grey with a warning, because it does not work well. |
| **Reliability** | How the model performs for this species. Prominent orange warning when it is unreliable. |
| **Drivers** | Which facts raised or lowered the risk, in percentage points, against a typical animal of the same species. |

There is also a batch mode: upload a CSV of a full day's intakes and the tool
ranks them by risk, highest first. This is the actual product — a prioritised
work list, not a dashboard.

---

## 3. The data

Two Socrata tables, joined on `animal_id`:

- Intakes — `https://data.austintexas.gov/resource/wter-evkm.json`
- Outcomes — `https://data.austintexas.gov/resource/9t4d-g238.json`

An animal can be admitted more than once, so `animal_id` is not unique. Each
intake is paired with the **next** outcome for that animal chronologically, and
an outcome can settle only one intake. Getting this wrong would silently corrupt
every downstream number, so it is covered by tests (`tests/test_join.py`).

`length_of_stay_days = outcome_datetime − intake_datetime`

**Rows dropped, and why:**

| Reason | Rows |
|---|---:|
| Intake with no matching outcome (still in shelter, or record missing) | 1,586 |
| Two intakes mapped to one outcome — kept the later intake | 88 |
| Stay longer than 365 days — real, but out of scope | 577 |

**Final dataset: 171,561 completed stays.**

One data quirk worth noting: 3,584 outcome timestamps carry an explicit `-05:00`
offset while the rest are naive local time. Both are Austin local. The offset is
stripped rather than converted, because converting only that subgroup to UTC
would shift them five hours and make their length of stay inconsistent with
everyone else's.

---

## 4. Method

### Temporal split — never random

| Period | Dates | Rows |
|---|---|---:|
| Train | 2013-10-01 → 2022-12-31 | 145,832 |
| Validation | 2023-01-01 → 2023-12-31 | 11,138 |
| Test | 2024-01-01 → 2025-05-04 | 14,591 |

The model learns from the past and is tested on the future, which is the
decision it faces in production. A random split would let the model see the
future while learning and would inflate every score. Tests fail if the periods
overlap or if any is empty.

### Two models

**Primary — a classifier** predicting `stay_90_plus` (length of stay ≥ 90 days),
using `HistGradientBoostingClassifier` with native categorical support so the
breed column is not one-hot exploded into thousands of columns.

**Secondary — a regression** on length of stay in days. Three objectives were
compared head to head:

| Variant | MAE | Median AE | R² | Bias |
|---|---:|---:|---:|---:|
| `log1p` (naive expm1) | 18.56 | 6.27 | −0.091 | −14.15 |
| `log1p` + Duan smearing | 24.16 | 17.26 | −0.019 | +5.86 |
| **`raw_absolute`** ✓ | **18.68** | **5.91** | −0.116 | −14.26 |

`raw_absolute` was chosen: best median error, no back-transform to bias, and the
loss being optimised is the loss being reported. Duan's smearing correction
fixes the level but costs 5.4 days of MAE, because smearing targets the
conditional *mean* while MAE is minimised by the *median*. That trade-off is
documented rather than hidden.

### Operating threshold — derived, not tuned

The two errors are not symmetric:

- A **miss** — an animal that will wait 90+ days and is not flagged — costs that
  animal months of kennel life. This is the exact failure the project exists to
  prevent.
- A **false alarm** costs one foster placement used on an animal that did not
  need it. Real, but recoverable in days.

A miss is judged **nine times** as costly. Under expected cost the optimal
threshold is `p* = 1 / (1 + 9) = 0.10`. The threshold is derived from that
stated value judgement, written openly in `config.py` where it can be argued
with, not tuned against a metric.

### Calibration

Isotonic regression was fitted on the validation period and then **switched
off**, because measuring it showed the Brier score unchanged (0.0448 both ways)
and the expected calibration error *worse* (0.0104 → 0.0123). Both curves are
still reported so the decision is visible as numbers rather than asserted.

### No hyperparameter search

Hyperparameters are fixed in `config.GBM_PARAMS` with a comment saying no search
was run. The first configuration tried is the one reported.

---

## 5. Results — held-out test set

| Metric | Classifier | Baseline |
|---|---:|---:|
| **PR-AUC** | **0.117** | 0.049 |
| Better than chance | **2.40×** | 1.00× |
| ROC-AUC | 0.715 | 0.500 |
| Brier score | 0.045 | 0.047 |
| Positive rate | 4.9% | — |

At the operating threshold `p ≥ 0.10`:

| | |
|---|---:|
| Precision | 13.8% |
| Recall | 29.2% |
| Flagged | 10.4% of intakes |

**In plain language:** of 100 animals flagged as at-risk on intake day, about 14
really did wait 90+ days. Random flagging would find 5.

**By species:**

| | PR-AUC | Base rate | Better than chance |
|---|---:|---:|---:|
| Dogs | 0.179 | 6.5% | 2.73× |
| Cats | 0.062 | 3.8% | **1.63×** |

---

## 6. The main finding — a leak that passed every check

The second strongest feature in the model was `has_name` — whether the animal
arrived with a name. It came from the intakes table, so the mechanical leakage
whitelist passed it.

The whitelist answers *"which table did this column come from"*. The question
that actually matters is **"when was this value written"**. A field can sit in
the intakes table and still be edited months into the stay, and this feed
carries no per-field timestamps.

### Evidence that `has_name` was not knowable at intake

| Check | Result |
|---|---|
| Stray intakes carrying a name | **67.6%** — animals picked up off the street cannot already have one |
| Wildlife intakes carrying a name | **3.0%** — the control that validates the method |
| Intake vs outcome copy of the name | **identical in 122,908 of 122,908 stays** |
| Within strays only: 90+ rate, named vs unnamed | **5.38% vs 0.89% — a 6.0× gap** |
| Named among animals staying 90+ days | 94.5%, against 70.3% of everyone else |

Wildlife is the decisive test. Nobody names a raccoon — and wildlife sits where
strays *should* sit. The field behaves correctly for the one category the
shelter never names and wrongly for every category it does. Staff were adding
names **during** the stay; the feature was partly recording staff attention.

### Cost of removal

| | Before | After |
|---|---:|---:|
| PR-AUC | 0.127 | 0.117 (−7.4%) |
| Recall at threshold | 35.3% | 29.2% |
| Long stays found | 252 | 208 (of 713) |

It was removed anyway. **A model that predicts the future using the future is
not a model.**

### What this shows about the shelter, not the data

Naming concentrates hard on animals already staying long. This **cannot
establish direction** — naming may follow a long stay rather than precede it,
and the data carries no timestamp to settle it. Nothing here says that naming an
animal harms it.

### Every other feature was then audited the same way

| Feature | Verdict |
|---|---|
| `sex` / `sterilization_status` | **Clean, verified.** 70,985 stays show Intact at intake → Fixed at outcome, proving the field is recorded per event. This is the one we most expected to leak. |
| `age_days` | **Clean, verified.** Ages differ between intake and outcome for 42,768 stays. |
| `primary_breed`, `primary_color`, `is_mix`, `is_black` | **Unverifiable.** Same animal-level storage shape as `has_name`, but no positive evidence of harm — breed and colour genuinely do not change, so identical copies are also what a clean field would produce. Kept and flagged. |
| `intake_type`, `intake_condition` | **Untestable.** Neither has an outcome-side copy, so the comparison cannot be run. "Normal" may mean *nothing obvious at the door* or *nothing found on examination hours later*. |
| `animal_type` | Clean. Species is not a judgement that gets revised. |
| `intake_month`, `intake_day_of_week`, `intake_season` | Clean. Derived from the intake timestamp itself. |

The standing record is `config.TEMPORAL_LEAKAGE_NOTES`. Tests require every
modelled feature to carry a verdict and every exclusion to carry a documented
reason. Re-run with `python tools/audit_timing.py`.

---

## 7. Other findings

**01 — Four percent of animals consume a third of the shelter.**
54.4% leave within 7 days using 7.5% of animal-days; 4.2% stay 90+ days using
33.1%, averaging 152 days each. Measured in animal-days, because duration *is*
the resource.

**02 — Black cats wait longer, black dogs do not, and the average hides both.**
Cats: 8.1 vs 7.1 days median, 5.10% vs 4.21% long-stay rate. Dogs: 6.0 vs 6.0
days, 4.10% vs 4.49% — slightly faster. The species cancel, producing a raw
difference near zero. Controlled for species, age, sex, intake type and
condition by direct standardisation: **+0.36 percentage points, 95% CI [+0.13,
+0.60]**, across 225 strata covering 96.7% of animals. Real, but small — and the
model assigns `is_black` exactly zero importance.

**03 — The strongest predictor was measuring our own staff, so it was deleted.**
See section 6.

**04 — Length of stay in days cannot be predicted here.**
R² is negative for all three regression variants.

**05 — Age moves the long-stay tail without moving the median.**
Dogs: median flat at 4–7 days across every age band, but the 90+ rate climbs
**5.4×** from puppies to adults. Cats: **U-shaped, not rising** — kittens under
two months have the longest median of any cat band at 11.9 days, longer than
seniors. This partly contradicts published analyses of the same dataset, and is
reported as found.

**06 — Cats have a season and dogs do not.**
A cat admitted in June waits 9.7 days against 5.3 in January — **1.8×**. Peak is
May at 11.1 days, when intake volume is also highest. Dogs sit within a 0.8-day
band all year.

**07 — Injured animals leave fastest and get stuck most often.**
Injured: 4.4-day median (vs 6.2 for Normal) but the highest 90+ rate at 6.19%
(vs 4.07%). Bimodal — most injuries resolve quickly, a minority become long
medical holds.

---

## 8. Limitations

**The day estimate is worse than a constant.** R² is negative for all three
regression variants. This is a finding, not a bug: length of stay is largely not
explained by what is knowable at intake. It is why the headline number is a
probability and not a number of days.

**The day estimate is systematically low** — about 14 days below actual on
average. Removable with Duan's smearing correction, but only at the cost of
accuracy, so accuracy was chosen.

**50% precision is unreachable at any threshold.** Class imbalance sets the
ceiling: only 4.9% of animals stay that long.

**For cats the model is barely better than chance** — 1.63×. A cat-only model
improves that by about 3%, so this is the ceiling of the available features
rather than a modelling defect. Cats have no equivalent of dog breed, which is
the strongest signal the model has.

**One feature removed for leaking, others unverifiable.** See section 6.

**What the model cannot see:** photographs and how the animal presents in them;
temperament; behaviour notes; medical complexity beyond one coarse condition
label; staff effort and volunteer advocacy; local adoption events; whether the
kennel sits on the main walkway or in a far corner. All of these matter
enormously, and none is in this data.

**One shelter, one city.** Everything here describes Austin Animal Center's
recording practices and adopter population. None of it transfers without
re-checking.

---

## 9. Design decisions

**The interface refuses when it should not be trusted.** Any species whose
PR-AUC lift falls below 2.0× gets a prominent, full-size low-confidence block
with the real score, telling the user to rely on their own judgement. It is
never shrunk or footnoted. *A model that silently emits a useless number is
worse than one that says "I do not know here".*

**The weak number looks weak.** The day estimate is deliberately rendered small
and grey, with a caveat naming the actual R², read from the metrics file rather
than hardcoded.

**Every displayed figure comes from the metrics file.** Nothing is typed into
the page by hand. A test scans the page for hardcoded metric literals and fails
if one appears. The renderer validates every key it reads, so a stale key
produces a visible error card rather than a silently blank panel.

---

## 10. Architecture

```
longstay/
  config.py     single source of truth: paths, split dates, feature lists,
                thresholds, TEMPORAL_LEAKAGE_NOTES
  fetch.py      Socrata paging, caching, retry with backoff
  clean.py      intake -> next-outcome pairing, target, feature engineering
  features.py   the whitelist gate and the temporal split
  model.py      baselines, three regression variants, the classifier
  evaluate.py   metrics, calibration, diagnosis, findings, figures
  service.py    scoring for the API
  api.py        FastAPI endpoints and the static page
static/
  index.html    the whole interface: one file, inline style and script
evals/results/  committed model artifacts the deployment boots from
tools/
  verify_join.py    inspect the intake -> outcome pairing
  audit_timing.py   re-run the timing-leakage audit
```

**Stack:** Python 3.11+, FastAPI, scikit-learn, pandas, matplotlib. One HTML
page, no framework, no build step. **310 automated tests.**

**Deployment:** Render, loading pre-trained artifacts — it never downloads or
retrains at startup. `GET /api/health` reports which artifacts loaded and when
the metrics were generated, so a cold-start failure is diagnosable without shell
access.

**API:**

| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | Score one intake |
| `POST /api/predict/batch` | Score and rank a day's intakes |
| `GET /api/model-card` | Everything the Model and Findings tabs display |
| `GET /api/health` | Artifact status and metrics timestamp |

---

## 11. How to run it

```bash
git clone https://github.com/TatianaK-ai/longstay
cd longstay
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
uvicorn longstay.api:app --port 8000
```

Then open http://127.0.0.1:8000

The fitted models are committed, so this needs no download and no training.

To rebuild everything from the source data:

```bash
python main.py fetch      # downloads from Socrata, cached; --force to refresh
python main.py clean      # join, target, features
python main.py stats      # describe the processed frame
python main.py train      # fits baselines, regressors, classifier
python main.py evaluate   # metrics, calibration, figures
pytest                    # 310 tests
```

Two read-only analysis tools:

```bash
python tools/verify_join.py    # inspect the intake -> outcome pairing
python tools/audit_timing.py   # re-run the timing-leakage audit
```
