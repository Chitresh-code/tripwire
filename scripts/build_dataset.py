"""Loads PaySim, computes features, splits by time, and reports what happened.

Run: uv run python scripts/build_dataset.py
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.paysim_loader import load_paysim
from src.pipelines.build_training_table import build_feature_table, time_based_split

RAW_PATH = Path("data/raw/paysim.csv")


def main() -> None:
    transactions = load_paysim(RAW_PATH)
    print(f"loaded {len(transactions):,} transactions")

    features = build_feature_table(transactions)
    print(f"columns: {list(features.columns)}")
    print(f"is_high_amount rate: {features['is_high_amount'].mean():.4%}")
    print(
        f"sender_txn_count_recent: mean={features['sender_txn_count_recent'].mean():.3f}, "
        f"max={features['sender_txn_count_recent'].max()}"
    )
    print(
        f"recipient_txn_count_recent: mean={features['recipient_txn_count_recent'].mean():.3f}, "
        f"max={features['recipient_txn_count_recent'].max()}"
    )

    train, test = time_based_split(features)
    print(
        f"train: {len(train):,} rows, {train['timestamp'].min()} to {train['timestamp'].max()}, "
        f"fraud rate {train['is_fraud'].mean():.4%}"
    )
    print(
        f"test:  {len(test):,} rows, {test['timestamp'].min()} to {test['timestamp'].max()}, "
        f"fraud rate {test['is_fraud'].mean():.4%}"
    )


if __name__ == "__main__":
    main()
