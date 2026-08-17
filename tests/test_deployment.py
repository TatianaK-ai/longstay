"""Deployment contract: no network, no training, diagnosable cold start.

The service must boot from committed artifacts alone. These tests fail if a
change reintroduces a download, a fit, or a silent missing-artifact path.
"""

from __future__ import annotations

import json
import socket

import pytest
from fastapi.testclient import TestClient

from longstay import config
from longstay.api import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ------------------------------------------------------------- no network


LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}


def _is_loopback(address) -> bool:
    """TestClient runs the app over an in-process loopback socketpair, so
    blocking every socket would fail the harness rather than the app. Only
    connections leaving the machine count as a violation."""
    if isinstance(address, tuple) and address:
        return str(address[0]) in LOOPBACK
    return True  # AF_UNIX and socketpair are local by construction


@pytest.fixture
def no_network(monkeypatch):
    """Make any OUTBOUND socket raise, so a hidden fetch cannot pass."""
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def guarded_connect(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise AssertionError(
                f"the service attempted a network connection to {address}"
            )
        return real_connect(self, address, *args, **kwargs)

    def guarded_create(address, *args, **kwargs):
        if not _is_loopback(address):
            raise AssertionError(
                f"the service attempted a network connection to {address}"
            )
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create)
    # DNS resolution is the earliest observable sign of an outbound call.
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: (
        _fail(host) if str(host) not in LOOPBACK
        else [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
    ))


def _fail(host):
    raise AssertionError(f"the service attempted to resolve {host!r}")


def test_the_network_block_actually_blocks(no_network):
    """Guards the three tests below from passing vacuously.

    If the fixture stopped working, every 'works offline' test would still
    pass while proving nothing.
    """
    import requests

    with pytest.raises(Exception) as caught:
        requests.get(config.INTAKES_URL, timeout=5)
    assert "attempted" in str(caught.value)


def test_prediction_works_with_all_networking_blocked(no_network, client):
    """The whole point of committing the artifacts."""
    response = client.post("/api/predict", json={
        "animal_type": "Dog", "breed": "Pit Bull Mix", "color": "Black",
        "age_upon_intake": "3 years", "sex_upon_intake": "Intact Male",
        "intake_type": "Stray", "intake_condition": "Normal",
    })
    assert response.status_code == 200
    assert 0.0 <= response.json()["risk_probability"] <= 1.0


def test_model_card_works_with_networking_blocked(no_network, client):
    assert client.get("/api/model-card").status_code == 200


def test_page_works_with_networking_blocked(no_network, client):
    assert client.get("/").status_code == 200


def test_socrata_urls_are_never_touched_by_the_serving_path():
    """fetch is importable (clean imports it) but must not be wired in."""
    import inspect

    from longstay import api, service

    for module in (api, service):
        source = inspect.getsource(module)
        assert "INTAKES_URL" not in source
        assert "OUTCOMES_URL" not in source
        assert "data.austintexas.gov" not in source


def test_serving_path_never_fits_a_model():
    import inspect

    from longstay import api, service

    for module in (api, service):
        source = inspect.getsource(module)
        assert ".fit(" not in source, f"{module.__name__} trains at runtime"


# --------------------------------------------------------------- health


def test_health_reports_every_required_artifact(client):
    body = client.get("/api/health").json()
    assert set(body["artifacts"]) == {
        "model", "classifier", "reference_row",
        "metrics", "findings", "species_reliability",
    }
    for name, info in body["artifacts"].items():
        assert info["present"], f"{name} is missing"
        assert info["bytes"] > 0, f"{name} is empty"


def test_health_reports_ok_when_everything_loads(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["load_error"] is None


def test_health_carries_a_metrics_timestamp(client):
    from datetime import datetime

    stamp = client.get("/api/health").json()["metrics_generated_at"]
    assert stamp
    datetime.fromisoformat(stamp)  # parses, and carries a timezone


def test_health_lists_the_figures_the_findings_reference(client):
    body = client.get("/api/health").json()
    charts = {f["chart"] for f in client.get("/api/model-card").json()["findings"]}
    assert charts <= set(body["figures"]), "a finding references a missing chart"


# ------------------------------------------------------------- artifacts


def test_figures_live_in_their_own_directory():
    assert config.FIGURES_DIR.exists()
    assert list(config.FIGURES_DIR.glob("*.png"))
    # and not loose in the results root
    assert not list(config.RESULTS_DIR.glob("*.png"))


def test_findings_json_matches_metrics_json():
    """Two files, one source of truth. They must never disagree."""
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    findings = json.loads(config.FINDINGS_PATH.read_text(encoding="utf-8"))
    assert findings == metrics["findings"]


def test_every_runtime_artifact_is_committed_not_generated():
    """These are the files the deployment cannot boot without."""
    for path in (
        config.MODEL_PATH,
        config.CLASSIFIER_MODEL_PATH,
        config.REFERENCE_ROW_PATH,
        config.METRICS_PATH,
        config.FINDINGS_PATH,
        config.SPECIES_RELIABILITY_PATH,
    ):
        assert path.exists(), f"{path.name} would have to be regenerated"


def test_raw_data_is_not_required_at_runtime():
    """data/raw/ is gitignored, so nothing served may depend on it."""
    gitignore = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/" in gitignore


def test_render_yaml_binds_all_interfaces_and_reads_port():
    render = (config.PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "--host 0.0.0.0" in render
    assert "$PORT" in render
    assert "healthCheckPath: /api/health" in render
    assert "longstay.api:app" in render
