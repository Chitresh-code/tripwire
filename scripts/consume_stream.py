"""Consumes the `transactions` topic and scores each event via the live scoring API.

Run: uv run python scripts/consume_stream.py
Needs: docker compose up -d, and the scoring API running
       (uvicorn src.serving.app:app --port 8000)
"""

from __future__ import annotations

from src.ingestion.stream_consumer import consume_and_score, make_consumer


def main() -> None:
    consumer = make_consumer()
    print("consuming 'transactions' topic, scoring each via http://localhost:8000/v1/score ...")
    consume_and_score(consumer)


if __name__ == "__main__":
    main()
