"""Per-species reliability disclosure.

A model that silently emits a useless number is worse than one that says
"I do not know here". These tests hold that promise to the code.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from longstay import config
from longstay.evaluate import (
    bias_by_decile,
    regression_variant_comparison,
    species_feature_contrast,
    species_reliability,
)
from longstay.service import TriageService
from tests.test_classifier import labelled_frame


# ------------------------------------------------------------- disclosure


def test_species_reliability_flags_a_weak_species():
    report = {
        "by_animal_type": {
            "Dog": {"pr_auc": 0.19, "positive_rate": 0.065, "pr_auc_lift": 2.9},
            "Cat": {"pr_auc": 0.076, "positive_rate": 0.038, "pr_auc_lift": 2.0},
        }
    }
    out = species_reliability(report)
    assert out["Dog"]["reliable"] is True
    assert out["Cat"]["reliable"] is (2.0 >= config.RELIABILITY_MIN_LIFT)


def test_reliability_uses_lift_not_raw_pr_auc():
    """A rare positive class drags PR-AUC down on its own.

    The honest question is whether the model beats that species' OWN base
    rate, not whether it matches another species' number.
    """
    report = {
        "by_animal_type": {
            "Rare": {"pr_auc": 0.02, "positive_rate": 0.004, "pr_auc_lift": 5.0},
        }
    }
    assert species_reliability(report)["Rare"]["reliable"] is True


def test_unreliable_message_names_the_number_and_is_not_hedged():
    service = TriageService(
        models=None,
        classifier=None,
        _reliability={
            "Cat": {"pr_auc": 0.0764, "base_rate": 0.0382,
                    "lift": 1.5, "reliable": False, "min_lift": 2.0}
        },
    )
    block = service.reliability_for("Cat")
    assert block["reliable"] is False
    assert "LOW CONFIDENCE" in block["message"]
    assert "0.076" in block["message"]
    assert "1.50x" in block["message"] or "1.5" in block["message"]


def test_reliable_species_gets_a_positive_statement():
    service = TriageService(
        models=None, classifier=None,
        _reliability={
            "Dog": {"pr_auc": 0.19, "base_rate": 0.065,
                    "lift": 2.9, "reliable": True, "min_lift": 2.0}
        },
    )
    block = service.reliability_for("Dog")
    assert block["reliable"] is True
    assert "2.9x" in block["message"]


def test_unmeasured_species_says_so_rather_than_implying_it_is_fine():
    service = TriageService(models=None, classifier=None, _reliability={})
    block = service.reliability_for("Livestock")
    assert block["reliable"] is None
    assert "not been measured" in block["message"]


def test_reliability_file_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "species_reliability.json"
    payload = {
        "Cat": {"pr_auc": 0.076, "base_rate": 0.038,
                "lift": 2.0, "reliable": True, "min_lift": 2.0}
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(config, "SPECIES_RELIABILITY_PATH", path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


# --------------------------------------------------------- decile bias


def test_bias_by_decile_detects_a_pattern_the_mean_hides():
    """Zero overall bias while over-predicting low and under-predicting high."""
    y_true = np.concatenate([np.full(500, 10.0), np.full(500, 100.0)])
    y_pred = np.concatenate([np.full(500, 30.0), np.full(500, 80.0)])

    overall = float(np.mean(y_pred - y_true))
    rows = bias_by_decile(y_true, y_pred)

    assert overall == pytest.approx(0.0)
    assert rows[0]["mean_bias_days"] > 0
    assert rows[-1]["mean_bias_days"] < 0


def test_variant_comparison_covers_every_variant():
    y_true = np.array([5.0, 20.0, 100.0, 3.0] * 30)
    variants = {
        "log1p": y_true * 0.5,
        "log1p_smearing": y_true * 1.3,
        "raw_absolute": y_true * 0.9,
    }
    out = regression_variant_comparison(y_true, variants)
    assert set(out) == set(variants)
    for block in out.values():
        assert "mae_days" in block
        assert "bias_by_decile" in block
        assert "max_abs_decile_bias_days" in block


def test_max_decile_bias_is_the_worst_not_the_average():
    y_true = np.concatenate([np.full(500, 10.0), np.full(500, 100.0)])
    y_pred = np.concatenate([np.full(500, 30.0), np.full(500, 80.0)])
    out = regression_variant_comparison(y_true, {"v": y_pred})["v"]
    assert out["max_abs_decile_bias_days"] == pytest.approx(20.0)
    assert abs(out["mean_bias_days"]) < 1e-9


# ------------------------------------------------------ feature contrast


def test_feature_contrast_finds_a_planted_separation():
    frame = labelled_frame(2000)
    frame["animal_type"] = "Cat"
    contrast = species_feature_contrast(frame, "Cat")

    assert contrast["species"] == "Cat"
    assert contrast["n"] == 2000
    assert contrast["most_separating_features"]
    assert "age_days" in contrast["numeric"]


def test_feature_contrast_ignores_a_species_that_is_absent():
    frame = labelled_frame(200)
    assert species_feature_contrast(frame, "Livestock") == {}


def test_feature_contrast_skips_tiny_levels():
    """A level seen twice cannot tell us anything and must not be reported."""
    frame = labelled_frame(1200)
    frame["animal_type"] = "Cat"
    frame.loc[frame.index[:2], "intake_type"] = "Vanishingly Rare"
    contrast = species_feature_contrast(frame, "Cat")
    reported = [
        row["value"]
        for block in contrast["categorical"].values()
        for row in block["top"] + block["bottom"]
    ]
    assert "Vanishingly Rare" not in reported
