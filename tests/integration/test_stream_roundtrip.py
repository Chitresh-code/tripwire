"""Integration test: real Redpanda broker, not a mock.

Needs `docker compose up -d` first. Skips itself if Redpanda isn't reachable
on localhost:9092, so a normal `pytest` run elsewhere doesn't hang or fail.
"""

from __future__ import annotations

import socket

import pytest

from src.ingestion.stream_consumer import make_consumer
from src.ingestion.stream_producer import make_producer

BROKER = ("localhost", 9092)


def _broker_reachable() -> bool:
    try:
        with socket.create_connection(BROKER, timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _broker_reachable(), reason="Redpanda not running (docker compose up -d)"
)


def test_produced_event_is_consumable() -> None:
    marker = "integration_test_txn_unique"
    producer = make_producer()
    event = {
        "transaction_id": marker,
        "account_id": "C_TEST",
        "recipient_id": "M_TEST",
        "amount": 123.45,
        "timestamp": "2026-07-19T00:00:00",
        "transaction_type": "TRANSFER",
    }
    producer.send("transactions", value=event)
    producer.flush()

    consumer = make_consumer()
    consumer.config["consumer_timeout_ms"] = 5000
    found = next((m.value for m in consumer if m.value["transaction_id"] == marker), None)
    consumer.close()

    assert found is not None
    assert found["amount"] == 123.45
