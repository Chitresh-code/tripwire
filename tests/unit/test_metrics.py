from tests.unit.test_serving_app import _client, _payload


def test_metrics_endpoint_reflects_a_scored_request() -> None:
    client = _client()

    client.post("/v1/score", json=_payload())
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "scoring_latency_seconds" in body
    assert "scoring_decisions_total" in body
    assert "fraud_probability" in body
