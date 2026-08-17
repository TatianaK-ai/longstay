"""The contract between metrics.json and the page.

A key the UI reads but the evaluation no longer writes produces a silently
blank panel rather than an error, which is the worst possible failure mode for
an honesty-critical block. These tests pin the key names.
"""

from __future__ import annotations

import json
import re

import pytest

from longstay import config

pytestmark = pytest.mark.skipif(
    not config.METRICS_PATH.exists(),
    reason="run `python main.py evaluate` first",
)


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def page() -> str:
    return (config.PROJECT_ROOT / "static" / "index.html").read_text(
        encoding="utf-8"
    )


def test_caveat_uses_the_flat_text_key(metrics):
    caveat = metrics["day_estimate_caveat"]
    assert set(caveat) == {"text", "r2"}
    assert "R²" in caveat["text"]


def test_caveat_quotes_the_actual_r2(metrics):
    """The number in the sentence must be the number that was computed."""
    caveat = metrics["day_estimate_caveat"]
    chosen = metrics["regression_variants"]["chosen"]
    r2 = metrics["regression_variants"]["comparison"][chosen]["r2"]
    assert caveat["r2"] == pytest.approx(r2)
    assert f"{r2:.3f}" in caveat["text"]


def test_limitations_use_flat_title_and_detail(metrics):
    assert len(metrics["model_limitations"]) >= 5
    for item in metrics["model_limitations"]:
        assert set(item) == {"id", "title", "detail"}
        assert item["title"] and item["detail"]


def test_all_required_limitations_are_present(metrics):
    """The four originally required, plus the timing leak we then found."""
    ids = {item["id"] for item in metrics["model_limitations"]}
    assert {
        "negative_r2", "day_bias", "precision_ceiling", "cat_ceiling",
        "timing_leak",
    } <= ids


def test_findings_have_the_keys_the_page_reads(metrics):
    assert len(metrics["findings"]) >= 7
    required = {"id", "label", "headline", "chart", "stats", "caveat", "sample"}
    for finding in metrics["findings"]:
        assert required <= set(finding)
        assert set(finding) <= required | {"callout", "notice"}
        assert finding["label"].startswith("FINDING")
        assert finding["sample"]
        for stat in finding["stats"]:
            assert set(stat) == {"label", "value"}


def test_page_validates_every_key_it_reads(page):
    """The renderer must list exactly the keys it dereferences, so a stale
    metrics.json produces a visible error rather than a blank card."""
    declared = re.search(r"FINDING_KEYS = \[([^\]]+)\]", page).group(1)
    for key in ("id", "label", "headline", "chart", "stats", "caveat",
                "sample"):
        assert f"'{key}'" in declared, f"renderer does not validate {key}"


def test_the_deleted_feature_finding_carries_the_warn_notice(metrics):
    """Amber is reserved for the one honest deletion."""
    leak = next(f for f in metrics["findings"] if f["id"] == "name_leak")
    assert "notice" in leak
    assert "removed" in leak["notice"].lower()


def test_notice_is_reserved_to_a_single_finding(metrics):
    with_notice = [f for f in metrics["findings"] if f.get("notice")]
    assert len(with_notice) == 1
    assert with_notice[0]["id"] == "name_leak"


def test_the_headline_finding_carries_a_plain_language_callout(metrics):
    """The long-tail argument must be stated, not left implicit in a chart."""
    load = next(f for f in metrics["findings"] if f["id"] == "load")
    callout = load["callout"]
    assert "%" in callout
    assert "three months" in callout
    # Both percentages, and no jargon a non-technical reader would trip on.
    assert callout.count("%") >= 2
    for jargon in ("PR-AUC", "R²", "percentile", "animal-days", "90+"):
        assert jargon not in callout


def test_callout_is_reserved_and_not_on_every_finding(metrics):
    """If everything is highlighted, nothing is."""
    with_callout = [f for f in metrics["findings"] if f.get("callout")]
    assert len(with_callout) == 1
    assert with_callout[0]["id"] == "load"


def test_callout_percentages_match_the_computed_shelter_load(metrics):
    load = next(f for f in metrics["findings"] if f["id"] == "load")
    shelter = metrics["shelter_load"]
    animals = f"{shelter['long_stay_share_of_animals']:.0%}"
    days = f"{shelter['long_stay_share_of_animal_days']:.0%}"
    assert animals in load["callout"]
    assert days in load["callout"]


def test_finding_charts_exist_on_disk(metrics):
    """A missing PNG renders as a broken image, not an error.

    Charts live in evals/results/figures/, which is what the API mounts at
    /assets — the finding stores a bare filename.
    """
    for finding in metrics["findings"]:
        assert (config.FIGURES_DIR / finding["chart"]).exists(), finding["chart"]


# ---------------------------------------------------------------- the page


def test_page_reads_only_keys_that_exist(metrics, page):
    """Every metrics key the page dereferences must be in metrics.json."""
    for expression, container in [
        ("l.title", metrics["model_limitations"][0]),
        ("l.detail", metrics["model_limitations"][0]),
        ("d.caveat.text", metrics["day_estimate_caveat"]),
        ("f.headline", metrics["findings"][0]),
        ("f.label", metrics["findings"][0]),
        ("f.chart", metrics["findings"][0]),
        ("f.caveat", metrics["findings"][0]),
    ]:
        assert expression in page, f"page no longer reads {expression}"
        key = expression.rsplit(".", 1)[1]
        assert key in container, f"metrics.json has no key {key!r}"


def test_page_does_not_read_retired_keys(page):
    for retired in ("title_ru", "detail_ru", "caveat.ru", "caveat.en",
                    "mean_smearing_corrected"):
        assert retired not in page, f"page still reads retired key {retired}"


def test_page_has_no_cyrillic_left(page):
    leftovers = re.findall(r"[Ѐ-ӿ]+", page)
    assert not leftovers, f"untranslated text remains: {leftovers[:5]}"


def test_font_import_is_the_first_rule_in_the_style_block(page):
    """@import is ignored by browsers unless it precedes every other rule."""
    style = page.split("<style>", 1)[1]
    first = style.strip().splitlines()[0].strip()
    assert first.startswith("@import"), first
    assert "Plus+Jakarta+Sans" in first
    assert "IBM+Plex+Mono" in first


def test_page_holds_no_hardcoded_metric_numbers(page):
    """Principle 1 applied to the UI: figures come from the API, not markup."""
    body = page.split("<script>", 1)[1]
    for literal in ("0.1268", "0.055", "2.59", "35.3", "14.2"):
        assert literal not in body, f"hardcoded metric {literal} in the page"
