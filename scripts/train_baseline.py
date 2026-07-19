"""Trains the LightGBM baseline on PaySim, evaluates it, and registers it as production.

Run: uv run python scripts/train_baseline.py
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.paysim_loader import load_paysim
from src.models import registry
from src.pipelines.train_and_register import run_training, save_artifacts

RAW_PATH = Path("data/raw/paysim.csv")
VERSION = "baseline_v1"


def main() -> None:
    transactions = load_paysim(RAW_PATH)
    run = run_training(transactions)

    print(f"train: {len(run.train):,} rows (fraud rate {run.train['is_fraud'].mean():.4%})")
    print(f"test:  {len(run.test):,} rows (fraud rate {run.test['is_fraud'].mean():.4%})")
    print("\noffline evaluation report (test set, never seen during training):")
    for name, value in run.metrics.items():
        print(f"  {name}: {value:.4f}")

    save_artifacts(run, VERSION)
    registry.set_production(VERSION)
    print(f"\nsaved and registered '{VERSION}' as production")


if __name__ == "__main__":
    main()
