from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.models.baseline import train_baseline
from src.models.sequence_model import train_sequence_model
from src.serving import app as app_module
from src.serving.app import create_app
from tests.unit.test_baseline import _learnable_dataset
from tests.unit.test_sequence_model import _learnable_sequence_dataset


def _app_and_client() -> tuple:
    model = train_baseline(_learnable_dataset())
    app = create_app(model=model, version="test_model")
    return app, TestClient(app)


def _client() -> TestClient:
    return _app_and_client()[1]


def _payload(amount: float = 500_000.0, transaction_type: str = "TRANSFER") -> dict:
    return {
        "transaction_id": "txn_1",
        "account_id": "acct_1",
        "recipient_id": "recip_1",
        "amount": amount,
        "timestamp": "2026-07-19T00:00:00",
        "transaction_type": transaction_type,
    }


def test_health() -> None:
    response = _client().get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_returns_a_probability() -> None:
    response = _client().post("/v1/score", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "txn_1"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["model_version"] == "test_model"
    assert body["decision"] in {"allow", "review", "block"}
    assert 0.0 <= body["threshold_used"] <= 1.0
    assert 1 <= len(body["top_contributing_features"]) <= 3
    assert {"feature", "contribution"} <= body["top_contributing_features"][0].keys()


def test_velocity_state_accumulates_across_requests() -> None:
    """Same account scored twice in a row should see its own prior transaction as recent history."""
    app, client = _app_and_client()

    client.post("/v1/score", json=_payload())
    client.post("/v1/score", json=_payload())

    assert app.state.sender_history.count_recent("acct_1", datetime(2026, 7, 19, 0, 0, 1)) == 2


def test_shadow_model_is_scored_but_doesnt_change_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shadow mode: a second model scores silently, response still reflects production only."""
    app, client = _app_and_client()
    shadow_model = train_baseline(_learnable_dataset(seed=2))

    monkeypatch.setattr(app_module.registry, "get_shadow", lambda: "shadow_v1")
    monkeypatch.setattr(app_module.joblib, "load", lambda path: shadow_model)

    response = client.post("/v1/score", json=_payload())

    assert response.status_code == 200
    assert response.json()["model_version"] == "test_model"


def test_sequence_model_is_scored_but_doesnt_change_the_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sequence model (FR5): scores silently alongside production, never in the response."""
    app, client = _app_and_client()
    fake_artifact = tmp_path / "sequence_model.pt"
    fake_artifact.touch()
    gru = train_sequence_model(_learnable_sequence_dataset(n=50), epochs=1, batch_size=32)

    monkeypatch.setattr(app_module.registry, "SEQUENCE_MODEL_PATH", fake_artifact)
    monkeypatch.setattr(app_module.sequence_model_module, "load_sequence_model", lambda path: gru)

    response = client.post("/v1/score", json=_payload())

    assert response.status_code == 200
    assert response.json()["model_version"] == "test_model"
