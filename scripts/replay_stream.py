"""Replays real PaySim transactions onto the `transactions` Redpanda topic.

Run: uv run python scripts/replay_stream.py [events_per_second]
Needs: docker compose up -d  (starts Redpanda on localhost:9092)
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.ingestion.paysim_loader import load_paysim
from src.ingestion.stream_producer import make_producer, replay

RAW_PATH = Path("data/raw/paysim.csv")


def main() -> None:
    events_per_second = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0

    transactions = load_paysim(RAW_PATH)
    producer = make_producer()

    print(f"replaying {len(transactions):,} transactions at {events_per_second}/sec...")
    count = replay(transactions, producer, events_per_second=events_per_second)
    print(f"sent {count:,} events to the 'transactions' topic")


if __name__ == "__main__":
    main()
