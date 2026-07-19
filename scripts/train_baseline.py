"""Trains the LightGBM baseline on PaySim and prints an offline evaluation report.

Run: uv run python scripts/train_baseline.py
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.paysim_loader import load_paysim
from src.models.baseline import evaluate, train_baseline
from src.pipelines.build_training_table import build_feature_table, time_based_split

RAW_PATH = Path("data/raw/paysim.csv")


def main() -> None:
    transactions = load_paysim(RAW_PATH)
    features = build_feature_table(transactions)
    train, test = time_based_split(features)
    print(f"train: {len(train):,} rows (fraud rate {train['is_fraud'].mean():.4%})")
    print(f"test:  {len(test):,} rows (fraud rate {test['is_fraud'].mean():.4%})")

    model = train_baseline(train)
    metrics = evaluate(model, test)

    print("\noffline evaluation report (test set, never seen during training):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
