"""README discipline: name the artifact, don't restate the number.

CLAUDE.md forbids quoting measured figures in prose because they go stale
silently. The headline sentence about 4% of animals and 33% of shelter time
is the single documented exception, and it is pinned to the computed values.
"""

from __future__ import annotations

import json
import re

import pytest

from longstay import config

README = config.PROJECT_ROOT / "README.md"

pytestmark = pytest.mark.skipif(
    not README.exists() or not config.METRICS_PATH.exists(),
    reason="README or metrics.json missing",
)


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


def test_headline_sentence_matches_the_computed_figures(readme, metrics):
    """The one exception, and it must stay true."""
    flat = re.sub(r"[\s>*]+", " ", readme)
    load = metrics["shelter_load"]
    animals = f"{load['long_stay_share_of_animals']:.0%}"
    days = f"{load['long_stay_share_of_animal_days']:.0%}"
    assert f"{animals} of animals stay longer than three months" in flat
    assert f"{days} of all the time the shelter has to give" in flat


def test_headline_sentence_matches_the_ui_callout(readme, metrics):
    """README and interface must not drift apart.

    Compared on normalised text: the README wraps the sentence across lines
    inside a bold blockquote, so the formatting differs while the words must
    not.
    """
    def normalise(text: str) -> str:
        return re.sub(r"[\s>*]+", " ", text).strip()

    callout = next(
        f for f in metrics["findings"] if f["id"] == "load"
    )["callout"]
    assert normalise(callout) in normalise(readme)


def test_no_other_measured_figures_are_quoted(readme, metrics):
    """Scan for the actual metric values appearing anywhere in the prose.

    Catches the real failure mode: someone pastes a PR-AUC or an R² into the
    README and it silently goes stale on the next retrain.
    """
    forbidden = {
        "classifier PR-AUC": metrics["classifier"]["test"]["pr_auc"],
        "classifier ROC-AUC": metrics["classifier"]["test"]["roc_auc"],
        "PR-AUC lift": metrics["classifier"]["test"]["pr_auc_lift"],
        "operating precision":
            metrics["classifier"]["at_operating_threshold"]["precision"],
        "operating recall":
            metrics["classifier"]["at_operating_threshold"]["recall"],
        "day-estimate R2": metrics["day_estimate_caveat"]["r2"],
    }
    for label, value in forbidden.items():
        for rendered in (f"{value:.4f}", f"{value:.3f}", f"{value:.2f}",
                         f"{value:.1%}", f"{value:.0%}"):
            assert rendered not in readme, (
                f"README quotes {label} as {rendered}; name the artifact "
                "instead so it cannot go stale"
            )


def test_readme_points_at_the_artifacts_rather_than_the_values(readme):
    assert "evals/results/metrics.json" in readme
    assert "config.TEMPORAL_LEAKAGE_NOTES" in readme


def test_timing_audit_is_its_own_section(readme):
    assert re.search(r"^### Temporal leakage", readme, re.M)
    # and it sits inside Limitations, where a reader will find it
    limitations = readme.index("## Limitations")
    assert readme.index("### Temporal leakage") > limitations


def test_timing_audit_covers_every_feature_group(readme):
    for feature in ("sterilization_status", "age_days", "primary_breed",
                    "intake_condition", "animal_type", "intake_month"):
        assert feature in readme, f"audit section omits {feature}"


def test_readme_documents_setup_from_cold(readme):
    for command in ("pip install -r requirements.txt",
                    "python main.py fetch",
                    "python main.py clean",
                    "python main.py train",
                    "python main.py evaluate",
                    "uvicorn longstay.api:app"):
        assert command in readme


def test_readme_states_the_primary_model_is_the_classifier(readme):
    body = readme.lower()
    assert "classifier is primary" in body
    assert "secondary" in body
