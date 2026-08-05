"""Trains the sequence model (GRU) in isolation and saves its artifacts.

Deliberately never imports src.models.baseline (LightGBM): running LightGBM
and PyTorch training in the same process crashes under real load — both
bundle their own OpenMP runtime, and the two implementations corrupt each
other's worker-thread pool (confirmed via a macOS crash report: the fault is
inside libomp's own __kmp_fork_barrier/__kmp_suspend, not application code).
Small/synthetic data didn't reproduce it; the full ~6.3M-row PaySim run did,
consistently. See docs/DECISIONS.md's M8 entry. scripts/compare_baselines.py
runs this as a separate subprocess for exactly this reason.

Run: uv run python scripts/train_sequence_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.ingestion.paysim_loader import load_paysim
from src.models import registry, sequence_model
from src.pipelines.build_training_table import build_feature_table, time_based_split

RAW_PATH = Path("data/raw/paysim.csv")


def main() -> None:
    transactions = load_paysim(RAW_PATH)
    features = build_feature_table(transactions)
    train, test = time_based_split(features)

    model = sequence_model.train_sequence_model(train)
    metrics = sequence_model.evaluate(model, test)

    registry.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), registry.SEQUENCE_MODEL_PATH)
    registry.SEQUENCE_METRICS_PATH.write_text(json.dumps(metrics))

    print("sequence model metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"saved to {registry.SEQUENCE_MODEL_PATH}")


if __name__ == "__main__":
    main()
