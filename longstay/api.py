"""FastAPI app. One JSON endpoint, one static page.

    uvicorn longstay.api:app --reload

/api/predict returns BOTH numbers. risk_probability is the primary figure and
drives the band; predicted_days is secondary context. See service.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from . import __version__
from .service import TriageService

app = FastAPI(
    title="Longstay",
    description=(
        "Predicts which animals will still be in the shelter in three months, "
        "from what is knowable on intake day. A triage tool, not a dashboard."
    ),
    version="0.2.0",
)


class IntakeRecord(BaseModel):
    """One animal, as known at the moment of intake."""

    animal_type: str = Field(..., examples=["Dog"])
    breed: str | None = Field(None, examples=["Pit Bull Mix"])
    color: str | None = Field(None, examples=["Black/White"])
    age_upon_intake: str | None = Field(None, examples=["2 years"])
    sex_upon_intake: str | None = Field(None, examples=["Intact Male"])
    intake_type: str | None = Field(None, examples=["Stray"])
    intake_condition: str | None = Field(None, examples=["Normal"])
    name: str | None = Field(None, examples=["Rex"])
    intake_datetime: str | None = None


class DayCaveat(BaseModel):
    text: str
    r2: float | None = None


class PredictedDays(BaseModel):
    """Secondary. Never display without `caveat`."""

    median: float
    p10: float
    p90: float
    point_estimate: float
    caveat: DayCaveat
    is_primary: bool = False


class Driver(BaseModel):
    feature: str
    value: str
    compared_to: str
    effect_percentage_points: float


class ModelReliability(BaseModel):
    """How well the model actually works for this species, measured.

    `reliable` is False when the model barely beats that species' own base
    rate. Clients must surface this prominently, not as small print.
    """

    species: str
    reliable: bool | None
    message: str
    pr_auc: float | None = None
    base_rate: float | None = None
    lift: float | None = None


class Prediction(BaseModel):
    """risk_probability is the headline; predicted_days is reference only."""

    risk_probability: float
    risk_band: str
    primary_metric: str = "risk_probability"
    predicted_days: PredictedDays
    drivers: list[Driver]
    model_reliability: ModelReliability
    caveat: str


@lru_cache(maxsize=1)
def get_service() -> TriageService:
    return TriageService.load()


@app.post("/api/predict", response_model=Prediction)
def predict(record: IntakeRecord) -> dict:
    """Score one intake.

    risk_probability is the number to triage on. predicted_days is there so
    "high risk" has a magnitude attached, not because it is trustworthy on its
    own — see the calibration section of evals/results/metrics.json.
    """
    try:
        service = get_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return service.score(record.model_dump())


@app.post("/api/predict/batch")
def predict_batch(records: list[IntakeRecord]) -> list[dict]:
    """Score a day's intakes at once, ranked by risk descending."""
    if len(records) > 2000:
        raise HTTPException(status_code=413, detail="Max 2000 rows per batch.")
    try:
        service = get_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    scored = [service.score(record.model_dump()) for record in records]
    for record, result in zip(records, scored):
        result["input"] = record.model_dump()
    return sorted(scored, key=lambda r: r["risk_probability"], reverse=True)


@app.get("/api/model-card")
def model_card() -> dict:
    """Everything the Model and Findings tabs display.

    Served straight from metrics.json so the page never holds a number of its
    own. If a rerun changes a metric, the page changes with it.
    """
    if not config.METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No metrics.json yet. Run `python main.py evaluate`.",
        )
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))

    classifier = metrics["classifier"]
    return {
        "primary_model": metrics.get("primary_model"),
        "split": metrics["split"],
        "classifier": {
            "test": classifier["test"],
            "baseline": classifier["baseline_positive_rate"],
            "at_operating_threshold": classifier["at_operating_threshold"],
            "at_recall_50": classifier["at_recall_50"],
            "threshold_policy": classifier["threshold_policy"],
            "human_summary": classifier["human_summary"],
            "by_animal_type": classifier["by_animal_type"],
            "calibration": {
                k: v
                for k, v in classifier["calibration"].items()
                if not k.startswith("reliability_")
            },
        },
        "regression_variants": metrics["regression_variants"],
        "day_estimate_caveat": metrics["day_estimate_caveat"],
        "model_limitations": metrics["model_limitations"],
        "permutation_importance": metrics["permutation_importance"],
        "shelter_load": metrics["shelter_load"],
        "species_diagnosis": {
            name: {
                "general_model": block["general_model"],
                "specialised_improvement": block["specialised_improvement"],
                "specialised_helps": block["specialised_helps"],
                "reliable": block["reliable"],
            }
            for name, block in metrics["species_diagnosis"].items()
        },
        "findings": metrics["findings"],
        "unobserved_factors": metrics["unobserved_factors"],
        "failure_analysis": metrics["failure_analysis"],
        "calibration_interval": metrics["calibration"]["interval"],
    }


@app.get("/api/health")
def health() -> dict:
    """Enough detail to diagnose a cold start without shell access.

    Reports each artifact separately rather than one boolean, because the
    failure modes differ: a missing model means the service cannot score at
    all, while a missing metrics file means it scores but cannot explain
    itself. Also attempts the actual model load, since a file being present
    is not the same as it being loadable under the deployed library versions.
    """
    artifacts = {
        "model": config.MODEL_PATH,
        "classifier": config.CLASSIFIER_MODEL_PATH,
        "reference_row": config.REFERENCE_ROW_PATH,
        "metrics": config.METRICS_PATH,
        "findings": config.FINDINGS_PATH,
        "species_reliability": config.SPECIES_RELIABILITY_PATH,
    }
    present = {
        name: {
            "present": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        for name, path in artifacts.items()
    }

    figures = (
        sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))
        if config.FIGURES_DIR.exists()
        else []
    )

    loaded, load_error = False, None
    try:
        get_service()
        loaded = True
    except Exception as exc:  # surfaced, not swallowed
        load_error = f"{type(exc).__name__}: {exc}"

    metrics_generated_at = None
    if config.METRICS_PATH.exists():
        metrics_generated_at = datetime.fromtimestamp(
            config.METRICS_PATH.stat().st_mtime, tz=timezone.utc
        ).isoformat()

    healthy = loaded and all(a["present"] for a in present.values())
    return {
        "status": "ok" if healthy else "degraded",
        "models_loaded": loaded,
        "load_error": load_error,
        "artifacts": present,
        "figures": figures,
        "metrics_generated_at": metrics_generated_at,
        "long_stay_threshold_days": config.LONG_STAY_DAYS,
        "operating_threshold": config.OPERATING_THRESHOLD,
        "version": __version__,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    page = config.PROJECT_ROOT / "static" / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>Longstay</h1><p>No static page built yet.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


# The plots the Findings and Model tabs display are the same PNG files the
# evaluation wrote. Serving them directly means the page cannot show a chart
# that disagrees with the metrics beside it.
if config.FIGURES_DIR.exists():
    app.mount(
        "/assets", StaticFiles(directory=config.FIGURES_DIR), name="assets"
    )


if __name__ == "__main__":  # pragma: no cover
    # Local convenience only. Render runs uvicorn from render.yaml, which
    # supplies PORT; this fallback is for `python -m longstay.api`.
    import os

    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
