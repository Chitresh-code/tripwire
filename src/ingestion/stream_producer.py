"""Replays PaySim transactions onto a Kafka/Redpanda topic, in timestamp order.

Simulates a live transaction feed (PRD's "streaming simulation") from the
static PaySim file — PaySim itself is not a service, see docs/DECISIONS.md.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import pandas as pd
from kafka import KafkaProducer  # type: ignore[import-untyped]

TOPIC = "transactions"


def transactions_to_events(transactions: pd.DataFrame) -> Iterator[dict[str, str | float]]:
    """Offline rows, in timestamp order, as JSON-ready dicts matching /v1/score's request shape."""
    ordered = transactions.sort_values("timestamp")
    for transaction_id, account_id, recipient_id, amount, timestamp, transaction_type in zip(
        ordered["transaction_id"],
        ordered["account_id"],
        ordered["recipient_id"],
        ordered["amount"],
        ordered["timestamp"],
        ordered["transaction_type"],
    ):
        yield {
            "transaction_id": str(transaction_id),
            "account_id": account_id,
            "recipient_id": recipient_id,
            "amount": float(amount),
            "timestamp": timestamp.isoformat(),
            "transaction_type": transaction_type,
        }


def replay(
    transactions: pd.DataFrame,
    producer: KafkaProducer,
    events_per_second: float = 50.0,
    topic: str = TOPIC,
) -> int:
    """Publish every transaction to `topic`, paced at `events_per_second`. Returns count sent."""
    delay = 1.0 / events_per_second if events_per_second > 0 else 0.0
    count = 0
    for event in transactions_to_events(transactions):
        producer.send(topic, value=event)
        count += 1
        if delay:
            time.sleep(delay)
    producer.flush()
    return count


def make_producer(bootstrap_servers: str = "localhost:9092") -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
