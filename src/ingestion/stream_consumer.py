"""Consumes transaction events from Kafka/Redpanda and scores each one via the live API.

Deliberately calls the real /v1/score endpoint over HTTP rather than
re-implementing scoring here — matches the architecture (ingestion and
scoring are separate components, docs/ARCHITECTURE.md) and guarantees this
path exercises the exact same code as any other caller of the API.
"""

from __future__ import annotations

import json

import httpx
import structlog
from kafka import KafkaConsumer  # type: ignore[import-untyped]

from src.ingestion.stream_producer import TOPIC

log = structlog.get_logger()


def make_consumer(bootstrap_servers: str = "localhost:9092", topic: str = TOPIC) -> KafkaConsumer:
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )


def consume_and_score(
    consumer: KafkaConsumer, scoring_base_url: str = "http://localhost:8000"
) -> None:
    """Blocks forever, scoring each incoming event via the live API and logging the result."""
    with httpx.Client(base_url=scoring_base_url) as client:
        for message in consumer:
            event = message.value
            response = client.post("/v1/score", json=event)
            if response.status_code != 200:
                log.error(
                    "scoring_request_failed",
                    transaction_id=event.get("transaction_id"),
                    status=response.status_code,
                )
                continue
            result = response.json()
            log.info("stream_scored", **result)
